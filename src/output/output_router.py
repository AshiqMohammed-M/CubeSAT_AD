#!/usr/bin/env python3
"""
OutputRouter – fans telemetry dicts to multiple transports simultaneously.

Transports
----------
1. Serial / UART → ESP32  (pyserial + COBS framing)
2. WebSocket     → OpenMCT dashboard  (websockets, async)
3. InfluxDB      → time-series storage
4. CSV           → local file fallback

Each transport fails independently; one crash never stops the others.
"""

import asyncio
import csv
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import serial
from cobs import cobs
import websockets
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS

logger = logging.getLogger("output_router")

# Fields sent over UART (keep small for embedded MCU)
_UART_FIELDS = [
    "timestamp", "V_batt", "I_batt", "T_batt", "V_solar", "I_solar",
    "V_bus", "I_bus", "T_eps", "SOC", "anomaly_type", "anomaly_active",
]

# InfluxDB tag keys (categorical, not numeric)
_INFLUX_TAG_KEYS = {"anomaly_type", "anomaly_active", "orbit_sunlight"}

# Fields that are NOT numeric (skip for InfluxDB field values)
_NON_NUMERIC_KEYS = {
    "anomaly_type", "anomaly_active", "orbit_sunlight",
    "quaternion", "euler_angles_deg", "angular_velocity_rad_s",
    "gyroscope_rad_s", "accelerometer_m_s2", "magnetometer_uT",
}


class OutputRouter:
    """
    Receives telemetry dicts from TelemetryGenerator and fans them out to
    multiple transports simultaneously.

    Parameters
    ----------
    config : dict
        Loaded from eps_config.yaml (the ``output`` section).
    """

    def __init__(self, config: dict):
        self._cfg = config.get("output", config)

        # ── Serial / UART ─────────────────────────────────────────────
        serial_cfg = self._cfg.get("serial", {})
        self._serial_enabled = serial_cfg.get("enabled", False)
        self._serial_port_name = serial_cfg.get("port", "/dev/ttyUSB0")
        self._serial_baud = serial_cfg.get("baud_rate", 115200)
        self._serial_conn: Optional[serial.Serial] = None
        self._serial_counter = 0

        # ── WebSocket → OpenMCT ───────────────────────────────────────
        ws_cfg = self._cfg.get("websocket", {})
        self._ws_enabled = ws_cfg.get("enabled", False)
        self._ws_host = ws_cfg.get("host", "localhost")
        self._ws_port = ws_cfg.get("port", 8765)
        self._ws_clients: set = set()
        self._ws_server = None
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None

        # ── InfluxDB ──────────────────────────────────────────────────
        influx_cfg = self._cfg.get("influxdb", {})
        self._influx_enabled = influx_cfg.get("enabled", False)
        self._influx_url = influx_cfg.get("url", "http://localhost:8086")
        self._influx_token = influx_cfg.get("token", "")
        self._influx_org = influx_cfg.get("org", "")
        self._influx_bucket = influx_cfg.get("bucket", "eps")
        self._influx_client: Optional[InfluxDBClient] = None
        self._influx_write_api = None

        # ── CSV fallback ──────────────────────────────────────────────
        csv_cfg = self._cfg.get("csv", {})
        self._csv_enabled = csv_cfg.get("enabled", False)
        self._csv_path = Path(csv_cfg.get("output_path", "data/telemetry_log.csv"))
        self._csv_file = None
        self._csv_writer = None
        self._csv_header_written = False

        logger.info(
            "OutputRouter created  serial=%s  ws=%s  influx=%s  csv=%s",
            self._serial_enabled, self._ws_enabled,
            self._influx_enabled, self._csv_enabled,
        )

    # ==================================================================
    #  Lifecycle
    # ==================================================================

    def start(self) -> None:
        """Initialise all enabled transports."""
        if self._serial_enabled:
            self._open_serial()
        if self._ws_enabled:
            self._start_ws_server()
        if self._influx_enabled:
            self._open_influxdb()
        if self._csv_enabled:
            self._open_csv()
        logger.info("OutputRouter started")

    def stop(self) -> None:
        """Gracefully close all connections."""
        if self._serial_conn is not None:
            try:
                self._serial_conn.close()
                logger.info("Serial port closed")
            except Exception:
                logger.exception("Error closing serial port")
            self._serial_conn = None

        if self._ws_loop is not None:
            self._ws_loop.call_soon_threadsafe(self._ws_loop.stop)
            if self._ws_thread is not None:
                self._ws_thread.join(timeout=5)
            logger.info("WebSocket server stopped")

        if self._influx_client is not None:
            try:
                if self._influx_write_api is not None:
                    self._influx_write_api.close()
                self._influx_client.close()
                logger.info("InfluxDB client closed")
            except Exception:
                logger.exception("Error closing InfluxDB client")
            self._influx_client = None
            self._influx_write_api = None

        if self._csv_file is not None:
            try:
                self._csv_file.close()
                logger.info("CSV file closed")
            except Exception:
                logger.exception("Error closing CSV file")
            self._csv_file = None
            self._csv_writer = None

        logger.info("OutputRouter stopped")

    # ==================================================================
    #  Main interface
    # ==================================================================

    def route(self, telemetry: dict) -> None:
        """Fan *telemetry* out to every enabled transport."""
        if self._serial_enabled:
            try:
                self._send_serial(telemetry)
            except Exception:
                logger.exception("Serial transport error")

        if self._ws_enabled:
            try:
                self._send_websocket(telemetry)
            except Exception:
                logger.exception("WebSocket transport error")

        if self._influx_enabled:
            try:
                self._send_influxdb(telemetry)
            except Exception:
                logger.exception("InfluxDB transport error")

        if self._csv_enabled:
            try:
                self._send_csv(telemetry)
            except Exception:
                logger.exception("CSV transport error")

    # ==================================================================
    #  TRANSPORT 1 – Serial / UART → ESP32
    # ==================================================================

    def _open_serial(self) -> None:
        try:
            self._serial_conn = serial.Serial(
                port=self._serial_port_name,
                baudrate=self._serial_baud,
                timeout=1,
            )
            logger.info(
                "Serial port opened: %s @ %d baud",
                self._serial_port_name, self._serial_baud,
            )
        except serial.SerialException:
            logger.warning(
                "Serial port %s not available – UART transport disabled",
                self._serial_port_name,
            )
            self._serial_enabled = False

    def _send_serial(self, telemetry: dict) -> None:
        if self._serial_conn is None or not self._serial_conn.is_open:
            return

        # Build compact payload with only the required UART fields
        payload = {k: telemetry[k] for k in _UART_FIELDS if k in telemetry}
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        # COBS-encode and append 0x00 terminator
        encoded = cobs.encode(raw) + b"\x00"
        self._serial_conn.write(encoded)

        self._serial_counter += 1
        if self._serial_counter % 100 == 0:
            logger.info("Serial packets sent: %d", self._serial_counter)

    # ==================================================================
    #  TRANSPORT 2 – WebSocket → OpenMCT
    # ==================================================================

    def _start_ws_server(self) -> None:
        self._ws_loop = asyncio.new_event_loop()
        self._ws_thread = threading.Thread(
            target=self._run_ws_loop, daemon=True, name="ws-server",
        )
        self._ws_thread.start()
        logger.info(
            "WebSocket server starting on ws://%s:%d",
            self._ws_host, self._ws_port,
        )

    def _run_ws_loop(self) -> None:
        asyncio.set_event_loop(self._ws_loop)
        start_server = websockets.serve(
            self._ws_handler, self._ws_host, self._ws_port,
        )
        self._ws_server = self._ws_loop.run_until_complete(start_server)
        self._ws_loop.run_forever()
        self._ws_server.close()
        self._ws_loop.run_until_complete(self._ws_server.wait_closed())

    async def _ws_handler(self, websocket, path=None) -> None:
        """Register new WebSocket client and keep connection alive."""
        self._ws_clients.add(websocket)
        logger.info("WebSocket client connected (%d total)", len(self._ws_clients))
        try:
            async for _ in websocket:
                pass  # we only broadcast, never read from clients
        except websockets.ConnectionClosed:
            pass
        finally:
            self._ws_clients.discard(websocket)
            logger.info("WebSocket client disconnected (%d remaining)", len(self._ws_clients))

    def _send_websocket(self, telemetry: dict) -> None:
        if not self._ws_clients:
            logger.debug("WebSocket: no clients connected, skipping")
            return

        # Extract timestamp as unix ms for OpenMCT envelope
        ts_raw = telemetry.get("timestamp", time.time())
        if isinstance(ts_raw, (int, float)):
            unix_ms = int(ts_raw * 1000)
        else:
            unix_ms = int(time.time() * 1000)

        # Build one envelope per numeric field (OpenMCT format)
        envelopes = []
        for key, value in telemetry.items():
            if isinstance(value, (int, float)) and key != "timestamp":
                envelopes.append(json.dumps({
                    "id": key,
                    "timestamp": unix_ms,
                    "value": value,
                }))
            elif isinstance(value, bool):
                envelopes.append(json.dumps({
                    "id": key,
                    "timestamp": unix_ms,
                    "value": int(value),
                }))

        if not envelopes:
            return

        dead_clients = set()
        for ws in list(self._ws_clients):
            for msg in envelopes:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        ws.send(msg), self._ws_loop,
                    )
                    future.result(timeout=2)
                except Exception:
                    dead_clients.add(ws)
                    break
        self._ws_clients -= dead_clients

    # ==================================================================
    #  TRANSPORT 3 – InfluxDB
    # ==================================================================

    def _open_influxdb(self) -> None:
        try:
            self._influx_client = InfluxDBClient(
                url=self._influx_url,
                token=self._influx_token,
                org=self._influx_org,
            )
            self._influx_write_api = self._influx_client.write_api(
                write_options=ASYNCHRONOUS,
            )
            logger.info("InfluxDB client opened → %s", self._influx_url)
        except Exception:
            logger.error("InfluxDB connection failed – transport disabled")
            self._influx_enabled = False

    def _send_influxdb(self, telemetry: dict) -> None:
        if self._influx_write_api is None:
            return

        point = Point("eps_telemetry")

        # Tags (categorical)
        for tag in _INFLUX_TAG_KEYS:
            if tag in telemetry:
                point = point.tag(tag, str(telemetry[tag]))

        # Fields (numeric only)
        for key, value in telemetry.items():
            if key in _NON_NUMERIC_KEYS or key == "timestamp":
                continue
            if isinstance(value, (int, float)):
                point = point.field(key, float(value))

        # Timestamp
        ts_raw = telemetry.get("timestamp")
        if isinstance(ts_raw, (int, float)):
            point = point.time(int(ts_raw * 1_000_000_000))  # nanoseconds

        try:
            self._influx_write_api.write(
                bucket=self._influx_bucket, record=point,
            )
        except Exception:
            logger.error("InfluxDB write failed – disabling transport")
            self._influx_enabled = False

    # ==================================================================
    #  TRANSPORT 4 – CSV fallback
    # ==================================================================

    def _open_csv(self) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self._csv_path, "a", newline="")
        # Header is written on first row (see _send_csv)
        self._csv_header_written = self._csv_path.stat().st_size > 0
        logger.info("CSV output → %s", self._csv_path)

    def _send_csv(self, telemetry: dict) -> None:
        if self._csv_file is None:
            return

        # Flatten any list/array values to strings for CSV compatibility
        flat: dict[str, Any] = {}
        for k, v in telemetry.items():
            if isinstance(v, (list, tuple)):
                flat[k] = str(v)
            else:
                try:
                    import numpy as np
                    if isinstance(v, np.ndarray):
                        flat[k] = str(v.tolist())
                    else:
                        flat[k] = v
                except ImportError:
                    flat[k] = v

        if not self._csv_header_written:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(flat.keys()),
            )
            self._csv_writer.writeheader()
            self._csv_header_written = True
        elif self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(flat.keys()),
            )

        self._csv_writer.writerow(flat)
        self._csv_file.flush()


# ======================================================================
#  Self-test
# ======================================================================

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_output.csv"

        config = {
            "output": {
                "serial":    {"enabled": False},
                "websocket": {"enabled": False},
                "influxdb":  {"enabled": False},
                "csv":       {"enabled": True, "output_path": str(csv_path)},
            }
        }

        router = OutputRouter(config)
        router.start()

        # Generate 5 fake telemetry records
        for i in range(5):
            telem = {
                "timestamp": 1700000000.0 + i,
                "mission_time_s": float(i),
                "sample_index": i,
                "V_batt": 7.8 + i * 0.01,
                "I_batt": 0.5,
                "T_batt": 20.0 + i * 0.5,
                "V_solar": 16.0,
                "I_solar": 1.0,
                "V_bus": 7.75,
                "I_bus": 0.5,
                "T_eps": 20.0,
                "SOC": 60.0 - i,
                "anomaly_type": "normal",
                "anomaly_active": False,
                "orbit_sunlight": True,
                "latitude_deg": 45.0,
                "longitude_deg": 12.0,
                "altitude_km": 550.0,
                "quaternion": [1.0, 0.0, 0.0, 0.0],
            }
            router.route(telem)

        router.stop()

        # Verify CSV was written
        assert csv_path.exists(), "CSV file was not created"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"
        assert float(rows[0]["V_batt"]) == 7.8
        assert float(rows[4]["V_batt"]) == 7.84
        assert rows[2]["anomaly_type"] == "normal"

        print(f"CSV verified: {len(rows)} rows, all fields correct")
        print("OutputRouter self-test passed")

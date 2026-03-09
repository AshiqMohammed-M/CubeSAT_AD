#!/usr/bin/env python3
"""
main.py – Master entry point for the nanosatellite EPS simulation pipeline.

Layer 1:  SimulationEngine    – orbital mechanics + IMU
Layer 2:  TelemetryGenerator  – EPS sensor fusion + AnomalyInjector
          MCU_RuleEngine       – deterministic rule evaluation
Layer 3:  OutputRouter         – serial, WebSocket, InfluxDB, CSV

Usage
-----
    python src/main.py [options]
    python src/main.py --mode batch --duration 600 --anomaly-rate 0.10
    python src/main.py --no-serial --no-influx --verbose
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so "src.*" absolute imports resolve.
# Also add src/mcu and src/mcu/mcu_ai so their bare intra-package imports
# (e.g. "from mcu_logger import MCULogger") resolve correctly when the
# modules are loaded as part of the src.mcu package.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
for _p in [
    str(_PROJECT_ROOT),
    str(_SRC / "mcu"),
    str(_SRC / "mcu" / "mcu_ai"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# pylint: disable=wrong-import-position
from src.simulation.simulation_engine import SimulationEngine
from src.telemetry.anomaly_injector import AnomalyInjector
from src.telemetry.telemetry_generator import TelemetryGenerator
from src.output.output_router import OutputRouter
from src.mcu.mcu_rule_engine import MCU_RuleEngine

# OBCSystem is imported for type-awareness; the MCU already communicates with
# OBC internally via its shared obc_interface.
try:
    from src.obc.obc_main import OBCSystem  # noqa: F401
except Exception:
    OBCSystem = None  # OBC optional at pipeline level

# ---------------------------------------------------------------------------
# Counters (updated by the step callback; read by the shutdown handler)
# ---------------------------------------------------------------------------
_stats = {
    "total_steps":      0,
    "anomalies_injected": 0,
    "alerts_triggered": 0,
}

# Global handles populated in main() for use by the signal handler
_output_router: OutputRouter | None = None
_pipeline_logger: logging.Logger | None = None

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="EPS Guardian – nanosatellite telemetry simulation pipeline",
    )
    p.add_argument(
        "--config",
        default="src/config/eps_config.yaml",
        help="Path to eps_config.yaml (default: src/config/eps_config.yaml)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=5400.0,
        help="Simulation duration in seconds (default: 5400)",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Sample rate in Hz (default: 1.0)",
    )
    p.add_argument(
        "--mode",
        choices=["realtime", "batch"],
        default="realtime",
        help="'realtime' (sleep between steps) or 'batch' (as fast as possible)",
    )
    p.add_argument(
        "--anomaly-rate",
        type=float,
        default=None,
        metavar="RATE",
        help="Override anomaly injection rate 0.0-1.0 (default: from config)",
    )
    p.add_argument("--no-serial",    action="store_true", help="Disable serial transport")
    p.add_argument("--no-websocket", action="store_true", help="Disable WebSocket transport")
    p.add_argument("--no-influx",    action="store_true", help="Disable InfluxDB transport")
    p.add_argument("--verbose",      action="store_true", help="Set log level to DEBUG")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> logging.Logger:
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{ts}.log"

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s"

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt))
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)

    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.info("Logging initialised → %s", log_file)
    return pipeline_logger


# ---------------------------------------------------------------------------
# Config loading + CLI overrides
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """Mutate *cfg* in-place to reflect CLI flags."""
    output_cfg = cfg.setdefault("output", {})

    if args.no_serial:
        output_cfg.setdefault("serial", {})["enabled"] = False
    if args.no_websocket:
        output_cfg.setdefault("websocket", {})["enabled"] = False
    if args.no_influx:
        output_cfg.setdefault("influxdb", {})["enabled"] = False

    if args.anomaly_rate is not None:
        cfg.setdefault("anomaly_injection", {})["injection_rate"] = args.anomaly_rate


# ---------------------------------------------------------------------------
# Signal / graceful shutdown
# ---------------------------------------------------------------------------

def _make_signal_handler(logger: logging.Logger):
    def _handler(signum, _frame):
        logger.info("Signal %d received – shutting down…", signum)
        _shutdown(logger)
        sys.exit(0)
    return _handler


def _shutdown(logger: logging.Logger) -> None:
    global _output_router
    if _output_router is not None:
        try:
            _output_router.stop()
        except Exception:
            logger.exception("Error during OutputRouter shutdown")
        _output_router = None

    logger.info(
        "Pipeline summary │ steps: %d │ anomalies: %d │ alerts: %d",
        _stats["total_steps"],
        _stats["anomalies_injected"],
        _stats["alerts_triggered"],
    )
    print(
        f"\n{'='*56}\n"
        f"  Pipeline summary\n"
        f"  Total steps      : {_stats['total_steps']}\n"
        f"  Anomalies injected: {_stats['anomalies_injected']}\n"
        f"  Alerts triggered  : {_stats['alerts_triggered']}\n"
        f"{'='*56}"
    )


# ---------------------------------------------------------------------------
# Per-step processing (shared between realtime and batch modes)
# ---------------------------------------------------------------------------

def _build_step_handler(
    telemetry_generator: TelemetryGenerator,
    mcu_rule_engine: MCU_RuleEngine,
    output_router: OutputRouter,
    logger: logging.Logger,
):
    """Return a callback that processes one SimulationEngine step dict."""

    def on_step(sim_step: dict) -> None:
        # Layer 2: EPS sensor fusion + optional anomaly injection
        telemetry = telemetry_generator.generate(sim_step)

        # MCU deterministic rules; apply_rules returns (actions, message_type, details)
        try:
            _actions, message_type, _details = mcu_rule_engine.apply_rules(telemetry)
        except Exception:
            logger.exception("MCU rule engine error on step %d", _stats["total_steps"])
            message_type = "STATUS_OK"

        # Derive a simplified alert level from the OBC message type
        if "CRITICAL" in message_type:
            alert_level = "CRITICAL"
        elif "WARNING" in message_type or "ALERT" in message_type:
            alert_level = "WARNING"
        else:
            alert_level = "NORMAL"

        telemetry["mcu_alert"] = alert_level

        # Layer 3: fan out to all transports
        output_router.route(telemetry)

        # Update counters
        _stats["total_steps"] += 1
        if telemetry.get("anomaly_active"):
            _stats["anomalies_injected"] += 1
        if alert_level != "NORMAL":
            _stats["alerts_triggered"] += 1

        if _stats["total_steps"] % 100 == 0:
            logger.info(
                "Step %d │ alert=%s │ anomaly=%s",
                _stats["total_steps"],
                alert_level,
                telemetry.get("anomaly_type", "normal"),
            )

    return on_step


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_realtime(
    simulation_engine: SimulationEngine,
    on_step,
    logger: logging.Logger,
) -> None:
    logger.info(
        "Starting REALTIME mode (%.1f Hz, %.0f s)",
        simulation_engine.sample_rate_hz,
        simulation_engine.mission_duration_s,
    )
    simulation_engine.stream(callback=on_step)


def run_batch(
    simulation_engine: SimulationEngine,
    on_step,
    logger: logging.Logger,
) -> None:
    logger.info(
        "Starting BATCH mode (%.1f Hz, %.0f s)",
        simulation_engine.sample_rate_hz,
        simulation_engine.mission_duration_s,
    )
    t0 = time.perf_counter()
    all_steps = simulation_engine.run()
    for step in all_steps:
        on_step(step)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Batch complete: %d steps in %.2f s (%.0f steps/s)",
        len(all_steps), elapsed, len(all_steps) / elapsed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    global _output_router, _pipeline_logger

    args = _parse_args(argv)

    # ── Logging ──────────────────────────────────────────────────────────
    logger = _setup_logging(args.verbose)
    _pipeline_logger = logger

    # ── Config ───────────────────────────────────────────────────────────
    cfg = _load_config(args.config)
    _apply_cli_overrides(cfg, args)
    logger.info("Config loaded from %s", args.config)

    # ── Signal handlers ──────────────────────────────────────────────────
    handler = _make_signal_handler(logger)
    signal.signal(signal.SIGINT,  handler)
    signal.signal(signal.SIGTERM, handler)

    # ── Component instantiation ──────────────────────────────────────────
    inj_cfg    = cfg.get("anomaly_injection", {})
    anomaly_injector = AnomalyInjector(
        enabled=inj_cfg.get("enabled", True),
        injection_rate=inj_cfg.get("injection_rate", 0.05),
        anomaly_weights=inj_cfg.get("weights", None),
    )

    sim_cfg = cfg.get("simulation", {})
    simulation_engine = SimulationEngine(
        orbit_altitude_km=sim_cfg.get("orbit_altitude_km", 550.0),
        inclination_deg=sim_cfg.get("inclination_deg", 45.0),
        sample_rate_hz=args.rate,
        mission_duration_s=args.duration,
    )

    telemetry_generator = TelemetryGenerator(
        sim_engine=simulation_engine,
        anomaly_injector=anomaly_injector,
        config=cfg,
    )

    output_router = OutputRouter(cfg)
    _output_router = output_router

    # MCU_RuleEngine loads thresholds from YAML internally
    mcu_rule_engine = MCU_RuleEngine()

    # ── Start transports ─────────────────────────────────────────────────
    output_router.start()

    logger.info(
        "Pipeline ready │ mode=%s │ duration=%.0f s │ rate=%.1f Hz │ anomaly_rate=%.2f",
        args.mode,
        args.duration,
        args.rate,
        inj_cfg.get("injection_rate", 0.05),
    )

    # ── Step handler ─────────────────────────────────────────────────────
    on_step = _build_step_handler(
        telemetry_generator, mcu_rule_engine, output_router, logger,
    )

    # ── Execute ──────────────────────────────────────────────────────────
    try:
        if args.mode == "realtime":
            run_realtime(simulation_engine, on_step, logger)
        else:
            run_batch(simulation_engine, on_step, logger)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception:
        logger.exception("Unhandled exception in pipeline")
    finally:
        _shutdown(logger)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TelemetryGenerator – streaming wrapper around ProEPSSensorDataGenerator.

Converts one SimulationEngine step dict into a merged telemetry record
(orbital state + EPS sensor readings) without writing any files.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup – locate project root portably
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # src/telemetry/
_SRC_DIR = _THIS_DIR.parent                          # src/
_PROJECT_ROOT = _SRC_DIR.parent                      # project root

sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_SRC_DIR / "dataset_physical"))

from pro_eps_data_generator import ProEPSSensorDataGenerator  # existing class, unchanged

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class-level defaults (override via the config dict)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = {
    "noise": {
        "V_batt": 0.015,
        "I_batt": 0.020,
        "T_batt": 0.100,
        "V_solar": 0.025,
        "I_solar": 0.015,
        "V_bus":   0.012,
        "I_bus":   0.008,
        "T_eps":   0.080,
        "SOC":     0.050,
    }
}

# Schema metadata: (unit, expected_min, expected_max)
_SCHEMA: dict = {
    # --- Orbital fields from SimulationEngine ---
    "timestamp":            ("s",    0.0,    None),
    "mission_time_s":       ("s",    0.0,    None),
    "sample_index":         ("",     0,      None),
    "orbit_sunlight":       ("bool", False,  True),
    "latitude_deg":         ("deg",  -90.0,  90.0),
    "longitude_deg":        ("deg",  -180.0, 180.0),
    "altitude_km":          ("km",   300.0,  800.0),
    "quaternion":           ("",     None,   None),
    "euler_angles_deg":     ("deg",  -180.0, 180.0),
    "angular_velocity_rad_s": ("rad/s", -1.0, 1.0),
    "gyroscope_rad_s":      ("rad/s", -1.0,  1.0),
    "accelerometer_m_s2":   ("m/s²", -15.0, 15.0),
    "magnetometer_uT":      ("µT",   -60.0,  60.0),
    # --- EPS sensor fields ---
    "V_batt":   ("V",   7.2,   8.4),
    "I_batt":   ("A",  -2.5,   2.5),
    "T_batt":   ("°C", -10.0,  60.0),
    "V_solar":  ("V",  15.0,  18.0),
    "I_solar":  ("A",   0.0,   2.0),
    "V_bus":    ("V",   7.6,   8.2),
    "I_bus":    ("A",   0.05,  1.5),
    "T_eps":    ("°C", -10.0,  55.0),
    "SOC":      ("%",  15.0,  95.0),
    "anomaly_type":   ("",   None, None),
    "anomaly_active": ("bool", False, True),
    # --- Derived EPS fields ---
    "P_batt":           ("W",   None, None),
    "P_solar":          ("W",   None, None),
    "P_bus":            ("W",   None, None),
    "converter_ratio":  ("",    None, None),
    "power_balance":    ("W",   None, None),
}


class TelemetryGenerator:
    """
    Streaming wrapper around ProEPSSensorDataGenerator.

    Parameters
    ----------
    sim_engine : SimulationEngine
        The Layer-1 physics engine.  Its step() output is accepted by
        ``generate()``.
    anomaly_injector : AnomalyInjector, optional
        When supplied, ``maybe_inject()`` is called on every EPS record.
    config : dict, optional
        Override noise sigmas and other defaults.  Keys mirror _DEFAULT_CONFIG.
    random_seed : int
        Forwarded to the underlying ProEPSSensorDataGenerator.
    """

    def __init__(
        self,
        sim_engine,
        anomaly_injector=None,
        config: Optional[dict] = None,
        random_seed: int = 42,
    ):
        self._sim_engine = sim_engine
        self._injector = anomaly_injector
        self._cfg = {**_DEFAULT_CONFIG, **(config or {})}

        # Instantiate the existing generator (but we won't use its CSV path code)
        self._eps_gen = ProEPSSensorDataGenerator(random_seed=random_seed)

        logger.info(
            "TelemetryGenerator ready (anomaly_injector=%s)",
            "enabled" if anomaly_injector is not None else "disabled",
        )

    # ------------------------------------------------------------------
    # Primary streaming interface
    # ------------------------------------------------------------------

    def generate(self, sim_step: dict) -> dict:
        """
        Convert one SimulationEngine step dict into a merged telemetry record.

        Parameters
        ----------
        sim_step : dict
            Output of SimulationEngine.step().

        Returns
        -------
        dict
            Flat merge of orbital + EPS fields with optional anomaly injection.
        """
        timestamp_s = sim_step.get("mission_time_s", 0.0)

        # Build a datetime from mission time for use by EPS physics helpers
        epoch = datetime(2025, 1, 1, tzinfo=timezone.utc)
        from datetime import timedelta
        ts_dt = epoch + timedelta(seconds=timestamp_s)

        # Use orbit sunlight flag from the physics engine
        in_sunlight = bool(sim_step.get("orbit_sunlight", True))
        self._eps_gen.system_state["orbit_position"] = (
            (timestamp_s % (90 * 60)) / (90 * 60)
        )

        # Generate base EPS data using existing logic
        eps_data = self._eps_gen.generate_normal_data(ts_dt)
        if eps_data is None:
            # Rare rejection – fall back to minimal plausible values
            eps_data = self._fallback_eps(in_sunlight, ts_dt)

        # Carry the correct sunlight flag (engine is authoritative)
        eps_data["orbit_sunlight"] = in_sunlight

        # Derived power fields
        eps_data["P_batt"] = round(
            eps_data["V_batt"] * eps_data["I_batt"], 6
        )
        eps_data["P_solar"] = round(
            eps_data["V_solar"] * eps_data["I_solar"], 6
        )
        eps_data["P_bus"] = round(
            eps_data["V_bus"] * eps_data["I_bus"], 6
        )
        v_solar = eps_data["V_solar"]
        eps_data["converter_ratio"] = round(
            eps_data["V_bus"] / v_solar if v_solar > 0.1 else 0.0, 6
        )
        eps_data["power_balance"] = round(
            eps_data["P_solar"] - eps_data["P_bus"], 6
        )

        # Remove the datetime timestamp from eps_data (we use mission_time_s)
        eps_data.pop("timestamp", None)

        # Inject anomaly if injector is present
        eps_data.setdefault("anomaly_type", "normal")
        eps_data.setdefault("anomaly_active", False)
        if self._injector is not None:
            eps_data = self._injector.maybe_inject(eps_data)

        # Merge: sim_step fields first, then EPS fields overwrite orbit_sunlight
        merged = {**sim_step, **eps_data}

        return merged

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self) -> dict:
        """
        Return field metadata for all output fields.

        Returns a dict of:
            field_name -> {"unit": str, "min": float|None, "max": float|None}

        Used by OutputRouter for InfluxDB field/tag categorisation.
        """
        return {
            name: {"unit": unit, "min": lo, "max": hi}
            for name, (unit, lo, hi) in _SCHEMA.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_eps(in_sunlight: bool, ts_dt: datetime) -> dict:
        """Minimal plausible EPS record used when normal generation fails."""
        return {
            "V_batt": 7.8,
            "I_batt": 0.5 if in_sunlight else -0.5,
            "T_batt": 20.0,
            "V_solar": 16.0 if in_sunlight else 0.8,
            "I_solar": 1.0 if in_sunlight else 0.05,
            "V_bus": 7.75,
            "I_bus": 0.5,
            "T_eps": 20.0,
            "SOC": 60.0,
            "anomaly_type": "normal",
            "anomaly_active": False,
            "orbit_sunlight": in_sunlight,
        }

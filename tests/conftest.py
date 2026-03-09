"""
Shared pytest fixtures for Phase 6 test suite.

Provides:
    normal_telemetry_dict -- clean baseline EPS telemetry values
    minimal_config        -- all transports disabled except CSV (uses tmp_path)
"""

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure project root and MCU intra-package paths are on sys.path so all
# imports in src/ resolve correctly, matching what main.py does at startup.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"

for _p in [
    str(_PROJECT_ROOT),
    str(_SRC / "mcu"),
    str(_SRC / "mcu" / "mcu_ai"),
    str(_SRC / "dataset_physical"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# NOTE: Do NOT configure logging for "astropy" or "poliastro" at module level
# before those libraries are imported – doing so prevents astropy from patching
# its own AstropyLogger class and causes '_set_defaults' AttributeError.
# Logging suppression is handled via pytest's log_level setting in pyproject.toml.


@pytest.fixture
def normal_telemetry_dict() -> dict:
    """Return a clean baseline EPS telemetry dictionary with all required fields."""
    return {
        # Simulation / orbital
        "timestamp": 0.0,
        "mission_time_s": 0.0,
        "sample_index": 0,
        "orbit_sunlight": True,
        "latitude_deg": 0.0,
        "longitude_deg": 0.0,
        "altitude_km": 550.0,
        "quaternion": [1.0, 0.0, 0.0, 0.0],
        "euler_angles_deg": [0.0, 0.0, 0.0],
        "gyroscope_rad_s": [0.0, 0.0, 0.001],
        "accelerometer_m_s2": [0.0, 0.0, 9.8],
        "magnetometer_uT": [20.0, 0.0, -30.0],
        "angular_velocity_rad_s": [0.0, 0.0, 0.001],
        "position_eci": [6921.0, 0.0, 0.0],
        "velocity_eci": [0.0, 7.6, 0.0],
        # EPS sensor readings (all within normal operating ranges)
        "V_batt": 7.8,
        "I_batt": 0.5,
        "T_batt": 22.0,
        "V_solar": 16.5,
        "I_solar": 1.0,
        "V_bus": 7.9,
        "I_bus": 0.5,
        "T_eps": 25.0,
        "SOC": 65.0,
        # Derived EPS fields
        "P_batt": 3.9,
        "P_solar": 16.5,
        "P_bus": 3.95,
        "converter_ratio": 0.479,
        "power_balance": 12.55,
        # Anomaly metadata
        "anomaly_type": "normal",
        "anomaly_active": False,
        # MCU output
        "mcu_alert": "NORMAL",
    }


@pytest.fixture
def minimal_config(tmp_path) -> dict:
    """Return a config dict with all transports disabled except CSV at tmp_path."""
    csv_path = tmp_path / "telemetry_test.csv"
    return {
        "output": {
            "serial":    {"enabled": False},
            "websocket": {"enabled": False},
            "influxdb":  {"enabled": False},
            "csv": {
                "enabled": True,
                "output_path": str(csv_path),
            },
        }
    }

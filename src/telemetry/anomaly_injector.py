#!/usr/bin/env python3
"""
AnomalyInjector – standalone anomaly perturbation engine.

The 13 original EPS anomaly types plus 8 attitude/IMU anomaly types and
their perturbation logic live here.  The streaming pipeline injects
anomalies without depending on the batch dataset generator.
"""

import random
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default anomaly weight distribution — matches ProEPSSensorDataGenerator
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS: Dict[str, float] = {
    # --- Original 13 EPS anomaly types ---
    "batt_overheat":         0.12,
    "batt_undervoltage":     0.10,
    "batt_overvoltage":      0.08,
    "batt_overcurrent":      0.06,
    "solar_fault":           0.08,
    "converter_failure":     0.06,
    "eps_overheat":          0.06,
    "sensor_fault":          0.04,
    "battery_degradation":   0.06,
    "progressive_overheat":  0.05,
    "oscillation_coupled":   0.04,
    "solar_partial_failure": 0.03,
    "unknown_pattern":       0.02,
    # --- 4 attitude anomaly types ---
    "attitude_drift":        0.04,
    "attitude_tumble":       0.03,
    "attitude_freeze":       0.03,
    "pointing_loss":         0.02,
    # --- 4 IMU anomaly types ---
    "gyro_bias_fault":       0.03,
    "gyro_spike_fault":      0.03,
    "accelerometer_failure": 0.02,
    "magnetometer_anomaly":  0.02,
}

_ANOMALY_TYPES: List[str] = list(_DEFAULT_WEIGHTS.keys())


class AnomalyInjector:
    """
    Injects EPS and attitude/IMU anomalies into a telemetry dict.

    Parameters
    ----------
    enabled : bool
        Master switch.  When False, maybe_inject() is a no-op.
    injection_rate : float
        Fraction of calls to maybe_inject() that actually inject an anomaly.
    anomaly_weights : dict, optional
        Override the per-type sampling probabilities.  Must cover all 21 types.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        enabled: bool = True,
        injection_rate: float = 0.05,
        anomaly_weights: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        self.enabled = enabled
        self.injection_rate = injection_rate
        self.rng = random.Random(seed)
        np.random.seed(seed)

        weights_src = anomaly_weights if anomaly_weights is not None else _DEFAULT_WEIGHTS
        # Normalise in case the supplied weights don't sum to 1
        total = sum(weights_src.values())
        self._types = list(weights_src.keys())
        self._probs = [w / total for w in weights_src.values()]

        logger.info(
            "AnomalyInjector ready (enabled=%s, rate=%.2f)", enabled, injection_rate
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def maybe_inject(self, telemetry: dict) -> dict:
        """
        Probabilistically inject an anomaly.

        Sets ``telemetry['anomaly_type']`` and ``telemetry['anomaly_active']``
        regardless of whether an anomaly is injected.
        """
        if not self.enabled or self.rng.random() >= self.injection_rate:
            telemetry["anomaly_type"] = "normal"
            telemetry["anomaly_active"] = False
            return telemetry

        anomaly_type = self.rng.choices(self._types, weights=self._probs, k=1)[0]
        return self.force_inject(telemetry, anomaly_type)

    def force_inject(self, telemetry: dict, anomaly_type: str) -> dict:
        """
        Force-inject a specific anomaly type.

        Raises ValueError for unknown anomaly types.
        """
        if anomaly_type not in _DEFAULT_WEIGHTS:
            raise ValueError(
                f"Unknown anomaly type '{anomaly_type}'. "
                f"Valid types: {_ANOMALY_TYPES}"
            )
        telemetry = self._apply_perturbation(telemetry, anomaly_type)
        telemetry["anomaly_type"] = anomaly_type
        telemetry["anomaly_active"] = True
        return telemetry

    def get_anomaly_catalog(self) -> List[str]:
        """Return all 21 supported anomaly type names."""
        return list(_ANOMALY_TYPES)

    # ------------------------------------------------------------------
    # Perturbation functions — moved verbatim from ProEPSSensorDataGenerator
    # ------------------------------------------------------------------

    def _apply_perturbation(self, data: dict, anomaly_type: str) -> dict:
        """Dispatch to the per-type perturbation method."""
        data = data.copy()

        if anomaly_type == "batt_overheat":
            # Range starts at 61°C to reliably exceed the 60°C critical threshold
            data["T_batt"] = round(random.uniform(61, 70), 1)
            data["V_batt"] = round(data.get("V_batt", 7.8) * 0.98, 3)
            data["I_batt"] = round(min(abs(data.get("I_batt", 1.0)) * -1.2, -1.8), 3)

        elif anomaly_type == "batt_undervoltage":
            data["V_batt"] = round(random.uniform(6.0, 6.5), 3)
            data["I_batt"] = round(max(-abs(data.get("I_batt", 1.0)) * 1.5, -2.2), 3)
            data["SOC"] = round(random.uniform(10, 20), 1)

        elif anomaly_type == "batt_overvoltage":
            data["V_batt"] = round(random.uniform(8.5, 8.6), 3)
            data["I_batt"] = round(min(abs(data.get("I_batt", 1.0)) * 1.3, 2.3), 3)
            data["T_batt"] = data.get("T_batt", 20.0) + 8

        elif anomaly_type == "batt_overcurrent":
            data["I_batt"] = round(random.uniform(2.6, 3.0), 3)
            data["T_batt"] = data.get("T_batt", 20.0) + 12
            data["V_batt"] = round(data.get("V_batt", 7.8) * 0.96, 3)

        elif anomaly_type == "solar_fault":
            data["V_solar"] = round(random.uniform(2.0, 5.0), 3)
            data["I_solar"] = round(random.uniform(0, 0.1), 3)
            data["I_batt"] = round(max(-abs(data.get("I_batt", 1.0)) * 1.2, -2.0), 3)

        elif anomaly_type == "converter_failure":
            data["V_bus"] = round(random.uniform(4.0, 6.0), 3)
            data["I_bus"] = round(data.get("I_bus", 0.5) * 1.3, 3)
            data["T_eps"] = data.get("T_eps", 20.0) + 15

        elif anomaly_type == "eps_overheat":
            data["T_eps"] = round(random.uniform(60, 75), 1)
            data["V_bus"] = round(data.get("V_bus", 7.9) * 0.9, 3)
            data["I_bus"] = round(data.get("I_bus", 0.5) * 0.8, 3)

        elif anomaly_type == "sensor_fault":
            data["SOC"] = float("nan")
            data["V_batt"] = float("nan")

        elif anomaly_type == "battery_degradation":
            data["V_batt"] = round(data.get("V_batt", 7.8) * 0.85, 3)
            data["I_batt"] = round(data.get("I_batt", 1.0) * 1.3, 3)
            data["SOC"] = round(data.get("SOC", 50.0) * 0.9, 1)

        elif anomaly_type == "progressive_overheat":
            data["T_batt"] = data.get("T_batt", 20.0) + 20
            data["T_eps"] = data.get("T_eps", 20.0) + 15
            data["V_batt"] = round(data.get("V_batt", 7.8) * 0.98, 3)

        elif anomaly_type == "oscillation_coupled":
            orbit_period = 90 * 60
            timestamp = data.get("timestamp")
            if hasattr(timestamp, "hour"):
                total_s = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
            else:
                total_s = float(timestamp) if timestamp is not None else 0.0
            orbit_progress = (total_s % orbit_period) / orbit_period
            data["V_bus"] = round(
                data.get("V_bus", 7.9) + 0.3 * np.sin(orbit_progress * 8 * np.pi), 3
            )
            data["I_bus"] = round(
                data.get("I_bus", 0.5) + 0.2 * np.sin(orbit_progress * 8 * np.pi + np.pi / 4), 3
            )
            data["V_batt"] = round(
                data.get("V_batt", 7.8) + 0.1 * np.sin(orbit_progress * 8 * np.pi), 3
            )

        elif anomaly_type == "solar_partial_failure":
            data["I_solar"] = round(data.get("I_solar", 1.0) * 0.3, 3)
            data["V_solar"] = round(data.get("V_solar", 16.0) * 0.8, 3)

        elif anomaly_type == "unknown_pattern":
            for key in ["V_batt", "I_batt", "V_bus", "I_bus"]:
                if key in data:
                    v = data[key]
                    if isinstance(v, float) and not np.isnan(v):
                        data[key] = round(v + random.uniform(-0.3, 0.3), 3)
            t = data.get("T_batt")
            if t is not None:
                data["T_batt"] = round(t + random.uniform(-8, 8), 1)

        # =============================================================
        # ATTITUDE ANOMALIES (4 types)
        # =============================================================
        elif anomaly_type == "attitude_drift":
            # Gradually increase attitude error; random-walk quaternion
            data["attitude_error_deg"] = data.get("attitude_error_deg", 0.0) + random.uniform(0.4, 0.6)
            q = list(data.get("quaternion", [1, 0, 0, 0])
                     if not isinstance(data.get("quaternion"), list)
                     else data.get("quaternion", [1, 0, 0, 0]))
            for i in range(1, 4):
                q[i] += random.gauss(0, 0.005)
            norm = np.linalg.norm(q)
            if norm > 0:
                q = [c / norm for c in q]
            data["quaternion"] = q
            data["quaternion_w"] = q[0]
            data["quaternion_x"] = q[1]
            data["quaternion_y"] = q[2]
            data["quaternion_z"] = q[3]

        elif anomaly_type == "attitude_tumble":
            # Uncontrolled rotation – high angular velocity
            tumble_vel = [0.5, 0.3, -0.4]
            data["angular_velocity_rad_s"] = tumble_vel
            data["angular_velocity_x"] = tumble_vel[0]
            data["angular_velocity_y"] = tumble_vel[1]
            data["angular_velocity_z"] = tumble_vel[2]
            data["angular_velocity_magnitude"] = float(np.linalg.norm(tumble_vel))
            data["gyroscope_rad_s"] = tumble_vel
            data["gyro_x"] = tumble_vel[0]
            data["gyro_y"] = tumble_vel[1]
            data["gyro_z"] = tumble_vel[2]
            data["gyro_magnitude"] = float(np.linalg.norm(tumble_vel))
            # Rapidly diverge euler angles
            data["roll_deg"] = data.get("roll_deg", 0.0) + random.uniform(5, 15)
            data["pitch_deg"] = data.get("pitch_deg", 0.0) + random.uniform(5, 15)
            data["yaw_deg"] = data.get("yaw_deg", 0.0) + random.uniform(5, 15)
            data["attitude_error_deg"] = data.get("attitude_error_deg", 0.0) + random.uniform(5, 15)

        elif anomaly_type == "attitude_freeze":
            # Sensor freeze – angular velocity exactly zero, values locked
            data["angular_velocity_rad_s"] = [0.0, 0.0, 0.0]
            data["angular_velocity_x"] = 0.0
            data["angular_velocity_y"] = 0.0
            data["angular_velocity_z"] = 0.0
            data["angular_velocity_magnitude"] = 0.0
            data["gyroscope_rad_s"] = [0.0, 0.0, 0.0]
            data["gyro_x"] = 0.0
            data["gyro_y"] = 0.0
            data["gyro_z"] = 0.0
            data["gyro_magnitude"] = 0.0

        elif anomaly_type == "pointing_loss":
            # Sudden large attitude error – single-step discontinuity
            err = random.uniform(45.0, 180.0)
            data["attitude_error_deg"] = err
            # Create a random quaternion that is far from nadir reference
            axis = np.array([random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)])
            axis = axis / (np.linalg.norm(axis) + 1e-9)
            half_angle = np.radians(err) / 2.0
            q = [float(np.cos(half_angle)),
                 float(axis[0] * np.sin(half_angle)),
                 float(axis[1] * np.sin(half_angle)),
                 float(axis[2] * np.sin(half_angle))]
            data["quaternion"] = q
            data["quaternion_w"] = q[0]
            data["quaternion_x"] = q[1]
            data["quaternion_y"] = q[2]
            data["quaternion_z"] = q[3]

        # =============================================================
        # IMU ANOMALIES (4 types)
        # =============================================================
        elif anomaly_type == "gyro_bias_fault":
            bias = [0.05, -0.03, 0.07]
            gyro = data.get("gyroscope_rad_s", [0.0, 0.0, 0.0])
            if isinstance(gyro, np.ndarray):
                gyro = gyro.tolist()
            new_gyro = [g + b for g, b in zip(gyro, bias)]
            data["gyroscope_rad_s"] = new_gyro
            data["gyro_x"] = new_gyro[0]
            data["gyro_y"] = new_gyro[1]
            data["gyro_z"] = new_gyro[2]
            data["gyro_magnitude"] = float(np.linalg.norm(new_gyro))

        elif anomaly_type == "gyro_spike_fault":
            spike_axis = random.randint(0, 2)
            spike_val = random.choice([-2.0, 2.0])
            gyro = list(data.get("gyroscope_rad_s", [0.0, 0.0, 0.0]))
            if isinstance(gyro, np.ndarray):
                gyro = gyro.tolist()
            gyro[spike_axis] += spike_val
            data["gyroscope_rad_s"] = gyro
            data["gyro_x"] = gyro[0]
            data["gyro_y"] = gyro[1]
            data["gyro_z"] = gyro[2]
            data["gyro_magnitude"] = float(np.linalg.norm(gyro))

        elif anomaly_type == "accelerometer_failure":
            if random.random() < 0.5:
                # Total sensor death
                data["accelerometer_m_s2"] = [0.0, 0.0, 0.0]
                data["accel_x"] = 0.0
                data["accel_y"] = 0.0
                data["accel_z"] = 0.0
                data["accel_magnitude"] = 0.0
            else:
                # Saturated noise failure
                noisy = [random.gauss(0, 5.0) for _ in range(3)]
                data["accelerometer_m_s2"] = noisy
                data["accel_x"] = noisy[0]
                data["accel_y"] = noisy[1]
                data["accel_z"] = noisy[2]
                data["accel_magnitude"] = float(np.linalg.norm(noisy))

        elif anomaly_type == "magnetometer_anomaly":
            mag = list(data.get("magnetometer_uT", [20.0, 0.0, -30.0]))
            if isinstance(mag, np.ndarray):
                mag = mag.tolist()
            if random.random() < 0.5:
                # Rotate by 90°: swap and negate one axis
                mag = [mag[2], mag[0], -mag[1]]
            else:
                # Sensor degradation – scale by 0.1
                mag = [m * 0.1 for m in mag]
            data["magnetometer_uT"] = mag
            data["mag_x"] = mag[0]
            data["mag_y"] = mag[1]
            data["mag_z"] = mag[2]

        # Post-perturbation physical plausibility guards (from original code)
        if data.get("I_bus", 0) < 0:
            data["I_bus"] = 0.05
        v_bus = data.get("V_bus")
        v_batt = data.get("V_batt")
        if (
            v_bus is not None
            and v_batt is not None
            and isinstance(v_bus, float)
            and isinstance(v_batt, float)
            and not (np.isnan(v_bus) or np.isnan(v_batt))
            and v_bus > v_batt * 1.05
        ):
            data["V_bus"] = round(v_batt * 0.98, 3)

        return data

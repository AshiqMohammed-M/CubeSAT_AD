#!/usr/bin/env python3
"""
AnomalyInjector – standalone anomaly perturbation engine.

The 13 anomaly types and their perturbation logic are moved here
verbatim from ProEPSSensorDataGenerator.generate_anomaly().
Nothing in that class is deleted; this module is a clean extraction
so the streaming pipeline can inject anomalies without depending on
the batch dataset generator.
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
    "batt_overheat":         0.15,
    "batt_undervoltage":     0.12,
    "batt_overvoltage":      0.10,
    "batt_overcurrent":      0.08,
    "solar_fault":           0.10,
    "converter_failure":     0.08,
    "eps_overheat":          0.07,
    "sensor_fault":          0.05,
    "battery_degradation":   0.08,
    "progressive_overheat":  0.06,
    "oscillation_coupled":   0.05,
    "solar_partial_failure": 0.04,
    "unknown_pattern":       0.02,
}

_ANOMALY_TYPES: List[str] = list(_DEFAULT_WEIGHTS.keys())


class AnomalyInjector:
    """
    Injects EPS anomalies into a telemetry dict.

    Parameters
    ----------
    enabled : bool
        Master switch.  When False, maybe_inject() is a no-op.
    injection_rate : float
        Fraction of calls to maybe_inject() that actually inject an anomaly.
    anomaly_weights : dict, optional
        Override the per-type sampling probabilities.  Must cover all 13 types.
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
        """Return all 13 supported anomaly type names."""
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

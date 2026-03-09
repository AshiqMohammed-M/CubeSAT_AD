"""
test_phase6.py – Complete test suite for the nanosatellite EPS anomaly
detection pipeline.

Pipeline under test:
    SimulationEngine → TelemetryGenerator → AnomalyInjector
      → MCU_RuleEngine → OBC_AI → OutputRouter

Sections
--------
6A  Unit tests        (tests 1-4)
6B  Integration tests (tests 5-6)
6C  Model accuracy    (test 7)
"""

import importlib
import logging
import math
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers – project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "src" / "config" / "eps_config.yaml"

# EPS keys that anomaly injectors are expected to change
_EPS_KEYS = {"V_batt", "I_batt", "T_batt", "V_solar", "I_solar", "V_bus", "I_bus", "T_eps", "SOC"}


def _load_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ============================================================================
# 6A – UNIT TESTS
# ============================================================================

class TestSimulationEngine:
    def test_simulation_engine_step_keys(self):
        """Verify SimulationEngine.step() returns all required keys with correct types."""
        from src.simulation.simulation_engine import SimulationEngine

        engine = SimulationEngine()
        step = engine.step()

        required_keys = {
            "timestamp", "position_eci", "orbit_sunlight", "quaternion",
            "euler_angles_deg", "gyroscope_rad_s", "accelerometer_m_s2",
            "magnetometer_uT", "latitude_deg", "longitude_deg", "altitude_km",
        }
        missing = required_keys - set(step.keys())
        assert not missing, f"Missing keys from step(): {missing}"
        assert isinstance(step["orbit_sunlight"], bool), (
            f"orbit_sunlight should be bool, got {type(step['orbit_sunlight'])}"
        )
        assert len(step["quaternion"]) == 4, (
            f"quaternion length should be 4, got {len(step['quaternion'])}"
        )


class TestAnomalyInjector:
    def test_anomaly_injector_all_types(self, normal_telemetry_dict):
        """Verify every catalog anomaly type sets anomaly_active and changes at least one EPS value."""
        from src.telemetry.anomaly_injector import AnomalyInjector

        injector = AnomalyInjector(enabled=True, seed=42)
        catalog = injector.get_anomaly_catalog()
        assert len(catalog) == 13, f"Expected 13 anomaly types, got {len(catalog)}"

        baseline_eps = {k: normal_telemetry_dict[k] for k in _EPS_KEYS if k in normal_telemetry_dict}

        for anomaly_type in catalog:
            telemetry = normal_telemetry_dict.copy()
            result = injector.force_inject(telemetry, anomaly_type)

            assert result["anomaly_active"] is True, (
                f"{anomaly_type}: anomaly_active should be True"
            )
            assert result["anomaly_type"] == anomaly_type, (
                f"anomaly_type field mismatch: expected '{anomaly_type}', got '{result['anomaly_type']}'"
            )

            # At least one EPS numeric value must differ from baseline
            changed = False
            for k in _EPS_KEYS:
                if k not in result or k not in baseline_eps:
                    continue
                v_new = result[k]
                v_old = baseline_eps[k]
                if isinstance(v_new, float) and math.isnan(v_new):
                    changed = True
                    break
                if isinstance(v_new, (int, float)) and isinstance(v_old, (int, float)):
                    if abs(float(v_new) - float(v_old)) > 1e-9:
                        changed = True
                        break
                elif v_new != v_old:
                    changed = True
                    break
            assert changed, (
                f"{anomaly_type}: no EPS value changed from baseline after force_inject"
            )


class TestOutputRouterTransportIsolation:
    def test_output_router_transport_isolation(self, tmp_path, normal_telemetry_dict):
        """Verify a serial write failure is silently caught and CSV is still written."""
        from src.output.output_router import OutputRouter

        csv_path = tmp_path / "isolation_test.csv"
        config = {
            "output": {
                "serial":    {"enabled": True, "port": "/dev/ttyFAKE_NONEXISTENT_999", "baud_rate": 115200},
                "websocket": {"enabled": False},
                "influxdb":  {"enabled": False},
                "csv":       {"enabled": True, "output_path": str(csv_path)},
            }
        }

        router = OutputRouter(config)
        router.start()  # serial open fails → _serial_enabled becomes False

        # Re-enable serial and attach a mock connection that raises on every write
        router._serial_enabled = True
        fake_conn = mock.MagicMock()
        fake_conn.is_open = True
        fake_conn.write.side_effect = OSError("simulated UART failure")
        router._serial_conn = fake_conn

        # route() must NOT raise even though serial write raises
        router.route(normal_telemetry_dict)
        router.stop()

        # CSV must exist and contain exactly one data row
        assert csv_path.exists(), "CSV file was not created"
        df = pd.read_csv(csv_path)
        assert len(df) == 1, f"Expected 1 CSV row, got {len(df)}"


class TestOBCCriticalThreshold:
    def test_obc_critical_threshold(self):
        """Verify _classify_anomaly returns CRITICAL for reconstruction error above critical_threshold."""
        cfg = _load_cfg()
        critical_threshold = cfg["ai_thresholds"]["complex"]["critical"]

        from src.obc.ai.ai_complex_inference import OBC_AI

        obc = OBC_AI()
        # Pass an error clearly above the critical threshold
        error_above = critical_threshold * 1.1
        level, confidence = obc._classify_anomaly(error_above)

        assert level == "CRITICAL", (
            f"Expected 'CRITICAL' for error={error_above:.4f} "
            f"(threshold={critical_threshold:.4f}), got '{level}'"
        )


# ============================================================================
# 6B – INTEGRATION TESTS
# ============================================================================

class TestFullPipelineBatch60s:
    def test_full_pipeline_batch_60s(self, tmp_path):
        """Verify a 60-second batch run produces 60 CSV rows with valid anomaly and alert data."""
        # Build a temporary config with CSV pointing to tmp_path
        cfg = _load_cfg()
        csv_path = tmp_path / "batch_60s.csv"
        cfg.setdefault("output", {})
        cfg["output"]["serial"]    = {"enabled": False}
        cfg["output"]["websocket"] = {"enabled": False}
        cfg["output"]["influxdb"]  = {"enabled": False}
        cfg["output"]["csv"]       = {"enabled": True, "output_path": str(csv_path)}

        temp_cfg_path = tmp_path / "test_eps_config.yaml"
        with open(temp_cfg_path, "w") as f:
            yaml.dump(cfg, f)

        # Import (or re-import) main and reset module-level counters
        import src.main as main_mod
        importlib.reload(main_mod)  # ensures fresh _stats / _output_router state

        # Clear any logging handlers accumulated from previous calls
        logging.root.handlers = [
            h for h in logging.root.handlers
            if not isinstance(h, logging.FileHandler)
        ]

        main_mod.main([
            "--mode", "batch",
            "--duration", "60",
            "--anomaly-rate", "0.3",
            "--no-serial",
            "--no-websocket",
            "--no-influx",
            "--config", str(temp_cfg_path),
        ])

        assert csv_path.exists(), "Output CSV was not created by batch run"
        df = pd.read_csv(csv_path)

        assert len(df) == 60, f"Expected 60 rows, got {len(df)}"
        assert df["anomaly_type"].notna().all(), "anomaly_type column contains NaN"
        assert (df["mcu_alert"] != "NORMAL").any(), (
            "Expected at least one non-NORMAL mcu_alert in 60-second run with 30% anomaly rate"
        )


class TestForceInjectReachesCSV:
    def test_force_inject_reaches_csv(self, tmp_path):
        """Verify force-injected batt_overheat on step 5 appears in CSV row 5 with CRITICAL alert."""
        from src.simulation.simulation_engine import SimulationEngine
        from src.telemetry.anomaly_injector import AnomalyInjector
        from src.telemetry.telemetry_generator import TelemetryGenerator
        from src.output.output_router import OutputRouter
        from src.mcu.mcu_rule_engine import MCU_RuleEngine

        csv_path = tmp_path / "force_inject.csv"
        config = {
            "output": {
                "serial":    {"enabled": False},
                "websocket": {"enabled": False},
                "influxdb":  {"enabled": False},
                "csv":       {"enabled": True, "output_path": str(csv_path)},
            }
        }

        # Fresh pipeline components (injection disabled – we control step 5 manually)
        sim_engine      = SimulationEngine(mission_duration_s=10.0, noise_seed=77)
        anomaly_injector = AnomalyInjector(enabled=False, seed=77)
        telemetry_gen   = TelemetryGenerator(
            sim_engine=sim_engine,
            anomaly_injector=anomaly_injector,
            config={},
        )
        mcu_engine    = MCU_RuleEngine()
        output_router = OutputRouter(config)
        output_router.start()

        for step_idx in range(10):
            sim_step  = sim_engine.step()
            telemetry = telemetry_gen.generate(sim_step)

            # Step 5 (0-indexed 4) → force-inject battery overheat
            if step_idx == 4:
                telemetry = anomaly_injector.force_inject(telemetry, "batt_overheat")

            # MCU rule evaluation
            try:
                _actions, message_type, _details = mcu_engine.apply_rules(telemetry)
            except Exception:
                message_type = "STATUS_OK"

            if "CRITICAL" in message_type:
                alert_level = "CRITICAL"
            elif "WARNING" in message_type or "ALERT" in message_type:
                alert_level = "WARNING"
            else:
                alert_level = "NORMAL"

            telemetry["mcu_alert"] = alert_level
            output_router.route(telemetry)

        output_router.stop()

        assert csv_path.exists(), "Output CSV was not created"
        df = pd.read_csv(csv_path)
        assert len(df) == 10, f"Expected 10 rows, got {len(df)}"

        row5 = df.iloc[4]   # 0-indexed; step 5 is index 4
        assert str(row5["anomaly_type"]) == "batt_overheat", (
            f"Row 5 anomaly_type: expected 'batt_overheat', got '{row5['anomaly_type']}'"
        )
        assert str(row5["mcu_alert"]) == "CRITICAL", (
            f"Row 5 mcu_alert: expected 'CRITICAL', got '{row5['mcu_alert']}'"
        )
        assert float(row5["T_batt"]) > 60.0, (
            f"Row 5 T_batt should be > 60.0 for batt_overheat, got {row5['T_batt']}"
        )


# ============================================================================
# 6C – MODEL ACCURACY TESTS
# ============================================================================

class TestModelAccuracyCriticalRecall:
    # Anomaly types to evaluate (exact catalog names)
    _CRITICAL_TYPES = ["batt_overheat", "batt_undervoltage", "batt_overcurrent"]

    def _build_anomaly_sequence(self, anomaly_type: str, rng: np.random.Generator) -> np.ndarray:
        """
        Build a (30, 7) EPS feature sequence where every step has the given anomaly
        applied to a normal baseline.  Feature order: V_batt, I_batt, T_batt,
        V_solar, I_solar, V_bus, I_bus  (matches OBC_AI's expected layout).
        """
        from src.telemetry.anomaly_injector import AnomalyInjector

        injector = AnomalyInjector(enabled=True, seed=int(rng.integers(0, 2**31)))
        rows = []
        for _ in range(30):
            base = {
                "V_batt":  float(rng.normal(7.8, 0.1)),
                "I_batt":  float(rng.normal(0.5, 0.1)),
                "T_batt":  float(rng.normal(22.0, 1.0)),
                "V_solar": float(rng.normal(16.5, 0.3)),
                "I_solar": float(rng.normal(1.0, 0.05)),
                "V_bus":   float(rng.normal(7.9, 0.05)),
                "I_bus":   float(rng.normal(0.5, 0.05)),
                "T_eps":   float(rng.normal(25.0, 0.5)),
                "SOC":     float(rng.normal(65.0, 1.0)),
                "timestamp": 0.0,
            }
            perturbed = injector.force_inject(base, anomaly_type)
            rows.append([
                perturbed["V_batt"], perturbed["I_batt"], perturbed["T_batt"],
                perturbed["V_solar"], perturbed["I_solar"],
                perturbed["V_bus"], perturbed["I_bus"],
            ])
        seq = np.array(rows, dtype=np.float32)
        # Replace NaN (sensor_fault type) with 0
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        return seq

    def _build_normal_sequence(self, rng: np.random.Generator) -> np.ndarray:
        """Build a (30, 7) EPS feature sequence with no anomaly injected."""
        rows = []
        for _ in range(30):
            rows.append([
                float(rng.normal(7.8, 0.1)),   # V_batt
                float(rng.normal(0.5, 0.1)),   # I_batt
                float(rng.normal(22.0, 1.0)),  # T_batt
                float(rng.normal(16.5, 0.3)),  # V_solar
                float(rng.normal(1.0, 0.05)),  # I_solar
                float(rng.normal(7.9, 0.05)),  # V_bus
                float(rng.normal(0.5, 0.05)),  # I_bus
            ])
        return np.array(rows, dtype=np.float32)

    def test_model_accuracy_critical_recall(self):
        """Verify >=90% detection recall for batt_overheat, batt_undervoltage, batt_overcurrent."""
        from src.obc.ai.ai_complex_inference import OBC_AI

        obc = OBC_AI()
        rng = np.random.default_rng(seed=2025)

        # Dataset: 100 sequences per critical type + 200 normal
        n_per_type = 100
        n_normal   = 200  # 3 * 100 anomaly + 200 normal = 500 total

        records: list[dict] = []

        for atype in self._CRITICAL_TYPES:
            for _ in range(n_per_type):
                seq = self._build_anomaly_sequence(atype, rng)
                result = obc.analyze_sequence(seq)
                records.append({"true_type": atype, "predicted": result["ai_level"]})

        for _ in range(n_normal):
            seq = self._build_normal_sequence(rng)
            result = obc.analyze_sequence(seq)
            records.append({"true_type": "normal", "predicted": result["ai_level"]})

        assert len(records) == 500

        # ── Confusion matrix (console) ─────────────────────────────────
        all_types  = self._CRITICAL_TYPES + ["normal"]
        pred_levels = ["CRITICAL", "WARNING", "NORMAL"]
        header = f"\n{'Anomaly Type':<22} | {'CRITICAL':>8} | {'WARNING':>8} | {'NORMAL':>8} | {'Recall':>8}"
        print(header)
        print("-" * len(header))

        per_type_recall: dict[str, float] = {}
        for true_type in all_types:
            subset = [r for r in records if r["true_type"] == true_type]
            total  = len(subset)
            counts = {lvl: sum(1 for r in subset if r["predicted"] == lvl) for lvl in pred_levels}
            detected = counts["CRITICAL"] + counts["WARNING"]
            recall   = detected / total if total > 0 else 0.0
            per_type_recall[true_type] = recall
            print(
                f"{true_type:<22} | {counts['CRITICAL']:>8} | {counts['WARNING']:>8} | "
                f"{counts['NORMAL']:>8} | {recall:>7.1%}"
            )

        print("-" * len(header))
        print(f"Total samples: {len(records)}")

        # ── Assertions ────────────────────────────────────────────────
        TARGET_RECALL = 0.90
        for atype in self._CRITICAL_TYPES:
            recall = per_type_recall[atype]
            assert recall >= TARGET_RECALL, (
                f"Recall for '{atype}' = {recall:.1%} < {TARGET_RECALL:.0%} target"
            )

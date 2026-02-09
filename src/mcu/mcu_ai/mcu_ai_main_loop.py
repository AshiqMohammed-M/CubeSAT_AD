#!/usr/bin/env python3
"""
MCU AI MAIN LOOP – Final version with 30s time buffer
"""
#completed
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

# === CORRECTION OF IMPORTS ===
# Move up one level to reach the mcu/ folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_mcu_dir = os.path.dirname(current_dir)  # D:\final_year_project\EPS-Guardian\src\mcu\
sys.path.insert(0, parent_mcu_dir)

# Now import from the mcu/ folder
from mcu_logger import MCULogger
from mcu_rule_engine import MCU_RuleEngine, obc_interface
from mcu_data_interface import DataInterface

class MCUAI_MainLoop:
    def __init__(self):
        self.logger = MCULogger("mcu_ai_validation.log")
        self.logger.info("STARTING HYBRID SYSTEM WITH 30s TIME BUFFER")

        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        data_path = os.path.join(self.base_dir, "data", "dataset", "pro_eps_dataset.csv")

        self.data_interface = DataInterface(source="csv", path=data_path)
        self.rule_engine = MCU_RuleEngine()
        self.obc_interface = obc_interface

        self.model_dir = os.path.join(self.base_dir, "data", "ai_models", "model_simple")
        self.training_data_dir = os.path.join(self.base_dir, "data", "training_data")
        self.output_dir = os.path.join(self.base_dir, "data", "mcu", "outputs", "hybrid_simulation")
        os.makedirs(self.output_dir, exist_ok=True)

        self.load_ai_model()
        self.results = []
        self.cycle_count = 0
        self.ai_alert_buffer = deque(maxlen=5)

        self.logger.info("Hybrid system initialized with a 30s time buffer")
    def load_ai_model(self):
        """Load AI model with calibrated thresholds"""
        try:
            import joblib
            # Correction of paths - using training_data/
            self.scaler = joblib.load(os.path.join(self.training_data_dir, "ai_scaler.pkl"))
            self.feature_names = np.load(os.path.join(self.training_data_dir, "ai_feature_names.npy"), allow_pickle=True).tolist()
            
            # Load training data for simulation
            self.train_data = np.load(os.path.join(self.training_data_dir, "ai_train_data.npy"))

            self.thresholds = {
                "normal": 0.35,
                "warning": 0.55, 
                "critical": 0.75
            }

            self.logger.info(f"AI loaded ({len(self.feature_names)} features, calibrated thresholds)")
        except Exception as e:
            self.logger.error(f"AI loading error: {e}")
            raise

    def prepare_features(self, sensor_data):
        """Prepare features for AI"""
        features = {}
        base_features = ["V_batt", "I_batt", "T_batt", "V_bus", "I_bus", "V_solar", "I_solar"]

        for f in base_features:
            features[f] = sensor_data.get(f, 0.0)

        try:
            features["P_batt"] = features["V_batt"] * features["I_batt"]
            v_solar = features["V_solar"]
            features["converter_ratio"] = features["V_bus"] / v_solar if v_solar > 0.1 else 0.0
        except Exception:
            features["P_batt"], features["converter_ratio"] = 0.0, 0.0

        features["delta_V_batt"] = 0.0
        features["delta_T_batt"] = 0.0
        return features

    def normalize_features(self, features_dict):
        """Normalize features"""
        try:
            df = pd.DataFrame([features_dict], columns=self.feature_names)
            normalized = self.scaler.transform(df)
            return normalized.reshape(-1)
        except Exception as e:
            self.logger.error(f"Normalization error: {e}")
            return None

    def simulate_ai_inference(self, normalized_features):
        """Simulate AI inference"""
        try:
            distances = np.sqrt(np.sum((self.train_data - normalized_features) ** 2, axis=1))
            min_d, avg_d = np.min(distances), np.mean(distances)
            simulated_error = min_d * 0.25 + avg_d * 0.20
            return float(np.clip(simulated_error, 0.001, 1.0))
        except Exception as e:
            self.logger.error(f"AI simulation error: {e}")
            return 0.5

    def ai_anomaly_detection(self, sensor_data):
        """AI anomaly detection"""
        features = self.prepare_features(sensor_data)
        normalized = self.normalize_features(features)
        if normalized is None:
            return {"ai_error": -1, "ai_level": "ERROR"}

        ai_error = self.simulate_ai_inference(normalized)

        if ai_error < self.thresholds["normal"]:
            ai_level = "NORMAL"
        elif ai_error < self.thresholds["warning"]:
            ai_level = "WARNING"
        else:
            ai_level = "CRITICAL"

        self.logger.info(f"[AI] Error={ai_error:.4f} → {ai_level}")
        return {"ai_error": ai_error, "ai_level": ai_level, "features_used": len(self.feature_names)}

    def hybrid_decision_fusion(self, ai_result, rule_result):
        """AI + rules fusion"""
        ai_level = ai_result["ai_level"]
        rule_level = rule_result["max_level"]

        if rule_level == ai_level:
            return ai_level, "AGREEMENT", "HIGH"
        if rule_level == "CRITICAL":
            return "CRITICAL", "RULE_DOMINANT", "HIGH"
        if ai_level == "CRITICAL" and rule_level == "NORMAL":
            return "CRITICAL", "AI_DOMINANT", "HIGH"
        if ai_level == "WARNING" and rule_level == "NORMAL":
            return "WARNING", "AI_DOMINANT", "MEDIUM"
        if rule_level == "WARNING" and ai_level == "NORMAL":
            return "WARNING", "RULE_DOMINANT", "MEDIUM"
        if rule_level == "WARNING" and ai_level == "CRITICAL":
            return "CRITICAL", "FUSION", "HIGH"
        return rule_level, "RULE_DEFAULT", "LOW"

    def hybrid_decision_pipeline(self, sensor_data):
        """Hybrid decision pipeline with OBC transmission"""
        self.cycle_count += 1
        
        # Add data to temporal buffer (via rule_engine)
        self.obc_interface.add_sensor_data(sensor_data)
        
        # 1. MCU rules evaluation
        actions, message_type, rule_details = self.rule_engine.apply_rules(sensor_data)
        rule_result = self._convert_rule_result(actions, message_type, rule_details)
        
        # 2. AI detection
        ai_result = self.ai_anomaly_detection(sensor_data)
        
        # 3. Fusion
        final_decision, decision_source, confidence = self.hybrid_decision_fusion(ai_result, rule_result)
        
        # 4. OBC transmission for AI decisions (with window data)
        if final_decision == "CRITICAL":
            self.obc_interface.send_to_obc("ALERT_CRITICAL", {
                "detection_type": "AI_ANOMALY",
                "final_decision": final_decision,
                "decision_source": decision_source,
                "ai_confidence": ai_result.get("ai_error", 0),
                "ai_level": ai_result.get("ai_level", "UNKNOWN"),
                "rule_triggered": rule_result["details"].get("rule_triggered"),
                "emergency_level": "HIGH"
            }, include_window_data=True)
            
        elif final_decision == "WARNING":
            self.obc_interface.send_to_obc("SUMMARY", {
                "detection_type": "AI_ANOMALY",
                "final_decision": final_decision, 
                "decision_source": decision_source,
                "ai_confidence": ai_result.get("ai_error", 0),
                "ai_level": ai_result.get("ai_level", "UNKNOWN"),
                "rule_triggered": rule_result["details"].get("rule_triggered"),
                "emergency_level": "MEDIUM"
            }, include_window_data=True)
        
        # 5. Log final decision
        if final_decision != "NORMAL":
            status_text = "[CRITICAL]" if final_decision == "CRITICAL" else "[WARNING]"
            self.logger.info(f"{status_text} Decision: {final_decision} (Source: {decision_source})")

        return {
            "cycle": self.cycle_count,
            "timestamp": sensor_data.get("timestamp", datetime.now().isoformat()),
            "final_decision": final_decision,
            "decision_source": decision_source,
            "confidence": confidence,
            "rule_result": rule_result,
            "ai_result": ai_result
        }

    def _convert_rule_result(self, actions, message_type, rule_details):
        """Convert rule result to structured format"""
        if message_type == "ALERT_CRITICAL":
            return {"max_level": "CRITICAL", "actions": actions, "details": rule_details}
        elif message_type == "SUMMARY":
            return {"max_level": "WARNING", "actions": actions, "details": rule_details}
        else:
            return {"max_level": "NORMAL", "actions": actions, "details": rule_details}

    def run_hybrid_simulation(self, max_cycles=50):
        """Run a complete simulation"""
        self.logger.info(f"Démarrage simulation hybride ({max_cycles} cycles)")

        stats = {
            "NORMAL": 0, "WARNING": 0, "CRITICAL": 0,
            "agreement": 0, "ai_dominant": 0, "rules_dominant": 0, "fusion": 0
        }

        for _ in range(max_cycles):
            data = self.data_interface.get_next()
            if data is None:
                break
            result = self.hybrid_decision_pipeline(data)
            self.results.append(result)
            stats[result["final_decision"]] += 1

        self.save_results(stats)
        return stats

    def save_results(self, stats):
        """Save results"""
        # Detailed data
        df = pd.DataFrame([{
            "cycle": r["cycle"],
            "timestamp": r["timestamp"],
            "final_decision": r["final_decision"],
            "decision_source": r["decision_source"],
            "confidence": r["confidence"],
            "ai_level": r["ai_result"]["ai_level"],
            "ai_error": r["ai_result"]["ai_error"],
            "rule_level": r["rule_result"]["max_level"]
        } for r in self.results])

        csv_path = os.path.join(self.output_dir, "mcu_ai_detailed_results.csv")
        df.to_csv(csv_path, index=False)

        # Summary
        total = len(df)
        summary_data = [
            {"Type": "NORMAL", "Number": stats["NORMAL"], "Percentage": f"{(stats['NORMAL']/total)*100:.1f}%"},
            {"Type": "WARNING", "Number": stats["WARNING"], "Percentage": f"{(stats['WARNING']/total)*100:.1f}%"},
            {"Type": "CRITICAL", "Number": stats["CRITICAL"], "Percentage": f"{(stats['CRITICAL']/total)*100:.1f}%"},
        ]
        summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, "simulation_summary.csv")
        summary.to_csv(summary_path, index=False)

        # Save detailed statistics
        stats_path = os.path.join(self.output_dir, "hybrid_simulation_stats.json")
        with open(stats_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_cycles": total,
                "decisions": stats,
                "ai_thresholds": self.thresholds
            }, f, indent=2)

        self.logger.info("=== SIMULATION COMPLETED ===")
        for _, row in summary.iterrows():
            self.logger.info(f"{row['Type']}: {row['Number']} cycles ({row['Percentage']})")

        print(f"\nResults saved in: {self.output_dir}")

def main():
    try:
        system = MCUAI_MainLoop()
        system.run_hybrid_simulation(max_cycles=50)
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

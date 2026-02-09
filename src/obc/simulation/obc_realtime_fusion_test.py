#!/usr/bin/env python3
"""
REAL-WORLD TEST OF OBC AI - WITH YOUR ACTUAL MODEL
Tests the actual LSTM Autoencoder model from the model_complex/ folder
"""
#completed
import os
import sys
import time
import json
import numpy as np
from datetime import datetime, timedelta

# Correction of imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interface.obc_message_handler import OBCMessageHandler
from interface.obc_response_generator import OBCResponseGenerator

class RealAI_FusionTest:
    def __init__(self):
        print(" REAL-WORLD TEST OF OBC AI - LSTM AUTOENCODER MODEL")
        print("=" * 60)
        
        # Check for real AI
        self.check_real_ai()
        
        self.handler = OBCMessageHandler()
        self.response_gen = OBCResponseGenerator()
        self.message_count = 0
        self.critical_detections = 0
        self.warning_detections = 0
        self.normal_detections = 0
        
        print(" Test configured to use the REAL OBC AI model")

    def check_real_ai(self):
        """Checks if the real AI model is available"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_root, "data", "ai_models", "model_complex", "ai_model_lstm_autoencoder.h5")
        thresholds_path = os.path.join(project_root, "data", "ai_models", "model_complex", "ai_thresholds.json")
        
        print(f"  Searching for the real AI model...")
        print(f"   Model: {model_path}")
        print(f"   Thresholds: {thresholds_path}")
        
        if os.path.exists(model_path):
            print(f" REAL MODEL FOUND: {os.path.getsize(model_path)/1024/1024:.1f} MB")
        else:
            print(f" MODEL NOT FOUND - The test will use simulation mode")
            
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r') as f:
                thresholds = json.load(f)
            print(f" THRESHOLDS LOADED: {thresholds['anomaly_thresholds']}")
        else:
            print(" THRESHOLDS NOT FOUND")

    def generate_realistic_sensor_sequence(self, anomaly_type="NONE", progression=0):
        """
        Generates a realistic sequence of 30 points for the LSTM
        Required format: (30, 7) - 30 timesteps, 7 features
        """
        sequence = []
        base_time = datetime.now() - timedelta(seconds=30)
        
        for i in range(30):
            timestamp = base_time + timedelta(seconds=i)
            
            # Base nominal values
            base_values = {
                "V_batt": 7.4,
                "I_batt": 1.2, 
                "T_batt": 35.0,
                "V_bus": 7.8,
                "I_bus": 0.8,
                "V_solar": 15.2,
                "I_solar": 1.5
            }
            
            # Application of realistic anomalies
            if anomaly_type == "OVERHEAT":
                # Realistic progressive overheating
                base_values["T_batt"] = 40.0 + (i * 1.0) + (progression * 10)
                base_values["I_batt"] *= 1.2  # Current increases with temperature
                
            elif anomaly_type == "OVERCURRENT":
                # Realistic current overload
                base_values["I_batt"] = 2.5 + np.random.normal(0, 0.3)
                base_values["V_batt"] *= 0.95  # Voltage slightly decreases
                
            elif anomaly_type == "UNDERVOLTAGE":
                # Realistic deep discharge
                base_values["V_batt"] = 3.5 - (i * 0.05) - (progression * 0.5)
                base_values["I_batt"] = -1.5  # Battery discharges
                
            elif anomaly_type == "CONVERTER_FAULT":
                # Realistic converter fault
                base_values["V_bus"] = 5.0 + np.random.normal(0, 0.5)
                base_values["converter_ratio"] = 0.3  # Abnormal ratio
                
            # Addition of realistic noise
            noisy_data = {
                "timestamp": timestamp.isoformat() + "Z",
                "V_batt": base_values["V_batt"] + np.random.normal(0, 0.02),
                "I_batt": base_values["I_batt"] + np.random.normal(0, 0.05),
                "T_batt": base_values["T_batt"] + np.random.normal(0, 0.1),
                "V_bus": base_values["V_bus"] + np.random.normal(0, 0.01),
                "I_bus": base_values.get("I_bus", 0.8) + np.random.normal(0, 0.02),
                "V_solar": base_values["V_solar"] + np.random.normal(0, 0.1),
                "I_solar": base_values["I_solar"] + np.random.normal(0, 0.02)
            }
            
            sequence.append(noisy_data)
            
        return sequence

    def create_real_test_message(self, sequence_data, anomaly_type="NONE"):
        """Creates a realistic test message for the AI"""
        self.message_count += 1
        
        # Determines the message type based on the anomaly
        if anomaly_type == "NONE":
            message_type = "SUMMARY"
            priority = "MEDIUM"
        else:
            message_type = "ALERT_CRITICAL" 
            priority = "HIGH"
        
        message = {
            "header": {
                "message_id": self.message_count,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "EPS_MCU",
                "message_type": message_type,
                "priority": priority,
                "version": "1.0"
            },
            "payload": {
                "sensor_data": sequence_data[-5:],  # Last measurements
                "temporal_window": {
                    "window_size_seconds": 30,
                    "data_points_count": len(sequence_data),
                    "sensor_data": sequence_data  # Complete sequence for the AI
                },
                "emergency_level": "HIGH" if anomaly_type != "NONE" else "MEDIUM",
                "test_anomaly_type": anomaly_type  # For debugging
            }
        }
        
        return message

    def run_real_ai_test(self, duration_minutes=5):
        """Runs the test with the real AI model"""
        print(f"\n STARTING REAL AI TEST - Duration: {duration_minutes} minutes")
        print("Cycle | Anomaly      | AI Score   | AI Level  | OBC Decision")
        print("-" * 70)
        
        end_time = time.time() + (duration_minutes * 60)
        cycle = 0
        
        # Realistic test scenarios
        test_scenarios = [
            ("NONE", "Nominal system"),
            ("OVERHEAT", "Progressive overheating"), 
            ("OVERCURRENT", "Current overload"),
            ("UNDERVOLTAGE", "Battery discharge"),
            ("CONVERTER_FAULT", "Converter failure"),
            ("NONE", "Return to normal")
        ]
        
        while time.time() < end_time and cycle < len(test_scenarios):
            cycle += 1
            anomaly_type, description = test_scenarios[cycle - 1]
            
            print(f"\n TEST {cycle}: {description}")
            
            # GGeneration of realistic sequence
            sequence = self.generate_realistic_sensor_sequence(
                anomaly_type=anomaly_type, 
                progression=cycle/len(test_scenarios)
            )
            
            # Creation of test message
            test_message = self.create_real_test_message(sequence, anomaly_type)
            
            # Processing by OBC with the REAL AI
            start_time = time.time()
            obc_response = self.handler.process_mcu_message(test_message)
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Extraction of AI results
            ai_analysis = obc_response.get('ai_analysis', {})
            ai_score = ai_analysis.get('ai_score', 0)
            ai_level = ai_analysis.get('ai_level', 'UNKNOWN')
            confidence = ai_analysis.get('confidence', 'LOW')
            simulated = ai_analysis.get('simulated', True)
            
            # Statistics
            if ai_level == 'CRITICAL':
                self.critical_detections += 1
            elif ai_level == 'WARNING':
                self.warning_detections += 1
            else:
                self.normal_detections += 1
            
            # Display detailed results
            print(f"    AI Score: {ai_score:.6f}")
            print(f"    Level: {ai_level} (Confidence: {confidence})")
            print(f"    Model: {'SIMULATION' if simulated else 'REAL LSTM'}")
            print(f"    Processing time: {processing_time:.1f}ms")
            print(f"    OBC Decision: {obc_response['decision']} -> {obc_response['action']}")
            
            # Detection consistency check
            if anomaly_type != "NONE" and ai_level == "NORMAL":
                print(f"   ️  WARNING: Anomaly '{anomaly_type}' not detected!")
            elif anomaly_type == "NONE" and ai_level != "NORMAL":
                print(f"   ️  WARNING: False positive! Normal detected as {ai_level}")
            
            time.sleep(3)  # Pause for analysis
        self._print_real_test_summary()

    def _print_real_test_summary(self):
        """Displays the detailed summary of the real test"""
        print("\n" + "=" * 70)
        print(" FINAL REPORT - OBC REAL AI TEST")
        print("=" * 70)
        
        total_tests = self.critical_detections + self.warning_detections + self.normal_detections
        
        print(f"Tests performed: {total_tests}")
        print(f"CRITICAL detections: {self.critical_detections}")
        print(f"WARNING detections: {self.warning_detections}") 
        print(f"NORMAL detections: {self.normal_detections}")
        
        if total_tests > 0:
            print(f"CRITICAL rate: {(self.critical_detections/total_tests)*100:.1f}%")
            print(f"WARNING rate: {(self.warning_detections/total_tests)*100:.1f}%")
            print(f"NORMAL rate: {(self.normal_detections/total_tests)*100:.1f}%")
        
        # Performance evaluation
        print("\n REAL AI EVALUATION:")
        if self.critical_detections > 0:
            print(" The AI detects critical anomalies")
        else:
            print(" No critical anomalies detected - check the model")
            
        if self.normal_detections > 0:
            print(" The AI recognizes normal behavior")
        else:
            print(" No normal behavior recognized - check the thresholds")

def main():
    """Main entry point"""
    print(" REAL AI TEST SYSTEM - EPS GUARDIAN OBC")
    print("This test uses the REAL LSTM Autoencoder model")
    print("=" * 60)
    
    # Ask the user for the duration
    try:
        duration = int(input("Test duration (minutes, default: 5): ") or "5")
    except:
        duration = 5
    
    # Run the test
    real_tester = RealAI_FusionTest()
    real_tester.run_real_ai_test(duration)
    
    print(f"\n TEST COMPLETED! Check the logs in data/obc/logs/")
if __name__ == "__main__":
    main()

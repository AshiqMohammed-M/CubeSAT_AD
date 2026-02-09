#!/usr/bin/env python3
"""
OBC INPUT DATA SIMULATION
Tests the OBC system with simulated MCU messages
"""
#completed
import os
import sys
import json
import numpy as np
import logging
from datetime import datetime, timedelta

# Configuring paths
current_dir = os.path.dirname(os.path.abspath(__file__))
obc_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(obc_dir)
project_root = os.path.dirname(src_dir)

# ========== CORRECTED PATHS ==========
# All logs in obc/
OBC_LOGS_DIR = os.path.join(project_root, "data", "obc", "logs")
os.makedirs(OBC_LOGS_DIR, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OBC_LOGS_DIR, "obc_simulation_test.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OBC_Simulation_Test")

sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, obc_dir)

from interface.obc_message_handler import OBCMessageHandler
from interface.obc_response_generator import OBCResponseGenerator

def create_sample_mcu_message(message_type="SUMMARY", include_window=True):
    """Creates a simulated MCU message"""
    
    # Simulated sensor data - CORRECTED: exactly 30 points
    sensor_data = []
    base_time = datetime.now() - timedelta(seconds=30)
    
    for i in range(30):  # Exactly 30 points
        timestamp = base_time + timedelta(seconds=i)
        sensor_data.append({
            "timestamp": timestamp.isoformat() + "Z",
            "V_batt": 7.4 + np.random.normal(0, 0.1),
            "I_batt": 1.2 + np.random.normal(0, 0.2),
            "T_batt": 35.0 + np.random.normal(0, 0.5),
            "V_bus": 7.8 + np.random.normal(0, 0.05),
            "I_bus": 0.8 + np.random.normal(0, 0.1),
            "V_solar": 15.2 + np.random.normal(0, 0.3),
            "I_solar": 1.5 + np.random.normal(0, 0.1)
        })
    
    message = {
        "header": {
            "message_id": np.random.randint(1000, 9999),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "EPS_MCU",
            "message_type": message_type,
            "priority": "HIGH" if message_type == "ALERT_CRITICAL" else "MEDIUM",
            "version": "1.0"
        },
        "payload": {
            "rule_triggered": "R4" if message_type == "SUMMARY" else "R1",
            "sensor_data": sensor_data[-10:],  # Latest 10 points
            "actions_taken": ["REDUCE_LOAD", "LED_YELLOW"],
            "emergency_level": "HIGH" if message_type == "ALERT_CRITICAL" else "MEDIUM"
        }
    }
    
    if include_window:
        message["payload"]["temporal_window"] = {
            "window_size_seconds": 30,
            "data_points_count": 30,
            "sensor_data": sensor_data  # All the 30 points
        }
    
    return message

def simulate_critical_anomaly():
    """Creates a message with simulated critical anomaly"""
    sensor_data = []
    base_time = datetime.now() - timedelta(seconds=30)
    
    # Creating a sequence with progressive overheating
    for i in range(30):  # Exactly 30 points
        timestamp = base_time + timedelta(seconds=i)
        # Simulation of progressive overheating
        temp = 40.0 + (i * 1.0)  # From 40°C to 70°C
        
        sensor_data.append({
            "timestamp": timestamp.isoformat() + "Z",
            "V_batt": 7.4 + np.random.normal(0, 0.1),
            "I_batt": 1.2 + np.random.normal(0, 0.2),
            "T_batt": temp + np.random.normal(0, 0.5),
            "V_bus": 7.8 + np.random.normal(0, 0.05),
            "I_bus": 0.8 + np.random.normal(0, 0.1),
            "V_solar": 15.2 + np.random.normal(0, 0.3),
            "I_solar": 1.5 + np.random.normal(0, 0.1)
        })
    
    message = {
        "header": {
            "message_id": np.random.randint(1000, 9999),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "EPS_MCU",
            "message_type": "ALERT_CRITICAL",
            "priority": "HIGH",
            "version": "1.0"
        },
        "payload": {
            "rule_triggered": "R1",
            "sensor_data": sensor_data[-10:],
            "actions_taken": ["CUT_POWER", "LED_RED", "BUZZER_ALARM"],
            "emergency_level": "HIGH",
            "temporal_window": {
                "window_size_seconds": 30,
                "data_points_count": 30,
                "sensor_data": sensor_data  # All the 30 points
            }
        }
    }
    
    return message

def save_test_results(test_name, message, response):
    """Saves test results in data/obc/logs/"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"obc_test_{test_name}_{timestamp}.json"
        filepath = os.path.join(OBC_LOGS_DIR, filename)
        
        test_result = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "message_sent": {
                "message_type": message.get('header', {}).get('message_type'),
                "message_id": message.get('header', {}).get('message_id'),
                "rule_triggered": message.get('payload', {}).get('rule_triggered')
            },
            "response_received": response,
            "test_summary": {
                "decision": response.get('decision'),
                "action": response.get('action'),
                "ai_level": response.get('ai_analysis', {}).get('ai_level'),
                "ai_score": response.get('ai_analysis', {}).get('ai_score')
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(test_result, f, indent=2)
            
        logger.info(f"Backup test result: {filename}")
        
    except Exception as e:
        logger.error(f"Error saving test results: {e}")

def test_obc_system():
    """Complete test of the OBC system"""
    logger.info("STARTING OBC SYSTEM TEST")
    print("TESTING OBC SYSTEM")
    print("=" * 50)
    print(f"Logs: {OBC_LOGS_DIR}")
    print("=" * 50)
    
    handler = OBCMessageHandler()
    response_gen = OBCResponseGenerator()
    
    # Test 1: Message SUMMARY normal
    print("\n1. TEST MESSAGE SUMMARY (normal)")
    logger.info("Test 1: Message SUMMARY normal")
    summary_msg = create_sample_mcu_message("SUMMARY")
    
    # DEBUG: Check the structure of the message
    print(f"DEBUG: Message keys: {list(summary_msg.keys())}")
    print(f"DEBUG: Payload keys: {list(summary_msg['payload'].keys())}")
    if 'temporal_window' in summary_msg['payload']:
        window_data = summary_msg['payload']['temporal_window']['sensor_data']
        print(f"DEBUG: Points in temporal_window: {len(window_data)}")
    
    response = handler.process_mcu_message(summary_msg)
    print(f"   Message: {summary_msg['header']['message_type']}")
    print(f"   Response: {response['decision']} - {response['action']}")
    
    # Safe handling of ai_analysis
    ai_analysis = response.get('ai_analysis', {})
    ai_level = ai_analysis.get('ai_level', 'UNKNOWN')
    ai_score = ai_analysis.get('ai_score', -1)
    error_msg = ai_analysis.get('error', '')
    print(f"   AI Analysis: {ai_level} (score: {ai_score:.6f})")
    if error_msg:
        print(f"   AI Error: {error_msg}")
    
    # Save test results
    save_test_results("summary_normal", summary_msg, response)
    
    # Test 2: Message ALERT_CRITICAL with anomaly
    print("\n2. TEST MESSAGE ALERT_CRITICAL (anomaly)")
    logger.info("Test 2: Message ALERT_CRITICAL with anomaly")
    critical_msg = simulate_critical_anomaly()
    response = handler.process_mcu_message(critical_msg)
    print(f"   Message: {critical_msg['header']['message_type']}")
    print(f"   Response: {response['decision']} - {response['action']}")
    
    ai_analysis = response.get('ai_analysis', {})
    ai_level = ai_analysis.get('ai_level', 'UNKNOWN')
    ai_score = ai_analysis.get('ai_score', -1)
    error_msg = ai_analysis.get('error', '')
    print(f"   Analyse IA: {ai_level} (score: {ai_score:.6f})")
    if error_msg:
        print(f"   AI Error: {error_msg}")
    
    # Saving the result
    save_test_results("critical_anomaly", critical_msg, response)
    
    # Test 3: Message without temporal data
    print("\n3. TEST MESSAGE WITHOUT TEMPORAL DATA")
    logger.info("Test 3: Message without temporal data")
    incomplete_msg = create_sample_mcu_message("SUMMARY", include_window=False)
    response = handler.process_mcu_message(incomplete_msg)
    print(f"   Message: {incomplete_msg['header']['message_type']}")
    print(f"   Response: {response['decision']} - {response.get('error', 'No error')}")
    
    # Saving the result
    save_test_results("no_temporal_data", incomplete_msg, response)
    
    # Test 4: Generating structured response (only if ai_analysis exists)
    print("\n4. TEST GENERATING STRUCTURED RESPONSE")
    logger.info("Test 4: Generating structured response")
    
    # Reuse the answer from test 2 which should have data
    test_response = handler.process_mcu_message(critical_msg)
    
    if 'ai_analysis' in test_response:
        structured_response = response_gen.generate_response(
            critical_msg, 
            test_response['ai_analysis'],
            {"decision": test_response['decision'], "action": test_response['action'], "notes": test_response.get('notes', '')}
        )
        print(f"   Structured response generated: {structured_response['header']['message_type']}")
        print(f"   Priority: {structured_response['header']['priority']}")
        
        # Saving the structured response
        save_test_results("structured_response", critical_msg, structured_response)
    else:
        print("   Impossible to generate structured response - missing AI analysis")
        print(f"   Keys in response: {list(test_response.keys())}")
        logger.warning("Impossible to generate structured response - missing AI analysis")
    
    print("\n" + "=" * 50)
    print("TEST FINISHED")
    logger.info("OBC SYSTEM TEST FINISHED SUCCESSFULLY")
def main():
    """Main entry point with argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OBC system test")
    parser.add_argument("--test", choices=["all", "summary", "critical", "notemporal"], 
                       default="all", help="Type of test to run")
    
    args = parser.parse_args()
    
    logger.info(f"Starting OBC test - Type: {args.test}")
    print(f"Configuration logs: {OBC_LOGS_DIR}")
    
    test_obc_system()

if __name__ == "__main__":
    main()

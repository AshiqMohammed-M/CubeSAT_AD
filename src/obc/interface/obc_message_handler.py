#!/usr/bin/env python3
"""
OBC MESSAGE MANAGER
Receives and processes messages from the MCU
"""
#completed
import json
import logging
import sys
import os
import numpy as np
from datetime import datetime

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
obc_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(obc_dir)
project_root = os.path.dirname(src_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, obc_dir)

try:
    from ai.ai_complex_inference import obc_ai
    AI_AVAILABLE = True
    print("INFO: OBC AI loaded successfully")
except ImportError as e:
    print(f"WARNING: AI not available - {e}")
    AI_AVAILABLE = False

class OBCMessageHandler:
    def __init__(self):
        self.logger = logging.getLogger("OBC_MessageHandler")
        self.logger.info("Initializing the OBC message handler")
        self.ai_available = AI_AVAILABLE

    def process_mcu_message(self, message_json):
        """
        Processes a JSON message received from the MCU
        """
        try:
            # Conversion if necessary
            if isinstance(message_json, str):
                message = json.loads(message_json)
            else:
                message = message_json
            
            self.logger.info(f"Message received: {message.get('message_type', 'UNKNOWN')}")
            
            # Extraction of temporal window data
            temporal_window = self._extract_temporal_data(message)
            
            if temporal_window is None:
                return self._generate_error_response("Missing temporal data")
            
            # Complex AI analysis (if available)
            if self.ai_available:
                ai_result = obc_ai.analyze_sequence(temporal_window)
            else:
                ai_result = self._simulate_ai_analysis(temporal_window)
            
            # GGeneration of the response
            response = self._generate_obc_response(message, ai_result)
            
            self.logger.info(f"Response generated: {response.get('decision', 'UNKNOWN')}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return self._generate_error_response(str(e))

    def _extract_temporal_data(self, message):
        """Extracts temporal window data"""
        try:
            payload = message.get('payload', {})
            
            # Search for temporal data
            temporal_data = None
            
            # Case 1: Data in temporal_window
            if 'temporal_window' in payload:
                window_data = payload['temporal_window'].get('sensor_data', [])
                if window_data and len(window_data) >= 30:
                    temporal_data = np.array([[
                        point.get('V_batt', 0), point.get('I_batt', 0),
                        point.get('T_batt', 0), point.get('V_bus', 0),
                        point.get('I_bus', 0), point.get('V_solar', 0),
                        point.get('I_solar', 0)
                    ] for point in window_data])
                    return temporal_data
            
            # Case 2: Direct data in sensor_data
            elif 'sensor_data' in payload:
                sensor_data = payload['sensor_data']
                if isinstance(sensor_data, list) and len(sensor_data) >= 30:
                    temporal_data = np.array([[
                        point.get('V_batt', 0), point.get('I_batt', 0),
                        point.get('T_batt', 0), point.get('V_bus', 0),
                        point.get('I_bus', 0), point.get('V_solar', 0),
                        point.get('I_solar', 0)
                    ] for point in sensor_data[-30:]])  # Last 30 points
                    return temporal_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting data: {e}")
            return None

    def _simulate_ai_analysis(self, temporal_window):
        """Simulates AI analysis when the model is not available"""
        # simple simulation based on temperature
        avg_temp = np.mean(temporal_window[:, 2])  # T_batt
        
        if avg_temp > 60:
            return {
                "ai_score": 0.95,
                "ai_level": "CRITICAL",
                "confidence": "HIGH",
                "reconstruction_error": 0.95,
                "simulated": True
            }
        elif avg_temp > 45:
            return {
                "ai_score": 0.75,
                "ai_level": "WARNING", 
                "confidence": "MEDIUM",
                "reconstruction_error": 0.75,
                "simulated": True
            }
        else:
            return {
                "ai_score": 0.1,
                "ai_level": "NORMAL",
                "confidence": "HIGH",
                "reconstruction_error": 0.1,
                "simulated": True
            }

    def _generate_obc_response(self, original_message, ai_result):
        """Generates the OBC response"""
        message_type = original_message.get('message_type', 'UNKNOWN')
        ai_level = ai_result.get('ai_level', 'ERROR')
        
        response = {
            "timestamp": datetime.now().isoformat() + "Z",
            "original_message_id": original_message.get('header', {}).get('message_id'),
            "original_type": message_type,
            "ai_analysis": ai_result,
            "decision": "PENDING",
            "action": "NONE",
            "confidence": ai_result.get('confidence', 'LOW'),
            "notes": ""
        }
        
        # Decision logic based on AI analysis
        if ai_level == "CRITICAL":
            response.update({
                "decision": "CONFIRM",
                "action": "ISOLATE_BATTERY",
                "notes": "Critical anomaly confirmed by OBC AI"
            })
        elif ai_level == "WARNING":
            response.update({
                "decision": "MONITOR", 
                "action": "INCREASE_SAMPLING",
                "notes": "Anomaly warning detected - increased monitoring"
            })
        elif ai_level == "NORMAL":
            response.update({
                "decision": "IGNORE",
                "action": "NONE",
                "notes": "No anomaly detected by OBC AI"
            })
        else:  # ERROR
            response.update({
                "decision": "REQUEST_RETRANSMISSION",
                "action": "NONE",
                "notes": "AI analysis error - invalid data"
            })
        
        return response

    def _generate_error_response(self, error_msg):
        """Generates an error response"""
        return {
            "timestamp": datetime.now().isoformat() + "Z",
            "decision": "ERROR",
            "action": "NONE",
            "error": error_msg,
            "notes": "Error processing message"
        }

# Global instance
message_handler = OBCMessageHandler()

def process_incoming_message(message_json):
    """
    Utility function to process an MCU message
    """
    return message_handler.process_mcu_message(message_json)
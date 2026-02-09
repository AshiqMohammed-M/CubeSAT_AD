#!/usr/bin/env python3
"""
OBC RESPONSE GENERATOR
Creates structured responses for the MCU
"""
#completed
import json
import logging
from datetime import datetime

class OBCResponseGenerator:
    def __init__(self):
        self.logger = logging.getLogger("OBC_ResponseGenerator")
        self.response_counter = 0
        self.logger.info("Initializing the OBC response generator")

    def generate_response(self, original_message, ai_analysis, decision_data):
        """
        Generates a structured response for the MCU
        
        Args:
            original_message: Original message from the MCU
            ai_analysis: Result of the AI analysis
            decision_data: Decision data
            
        Returns:
            dict: Structured JSON response
        """
        self.response_counter += 1
        
        response = {
            "header": {
                "message_id": self.response_counter,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "OBC",
                "message_type": "RESPONSE",
                "priority": self._determine_priority(decision_data['decision']),
                "version": "1.0"
            },
            "payload": {
                "original_message": {
                    "id": original_message.get('header', {}).get('message_id'),
                    "type": original_message.get('message_type'),
                    "timestamp": original_message.get('header', {}).get('timestamp')
                },
                "obc_analysis": {
                    "ai_score": ai_analysis.get('ai_score'),
                    "ai_level": ai_analysis.get('ai_level'),
                    "confidence": ai_analysis.get('confidence'),
                    "reconstruction_error": ai_analysis.get('reconstruction_error')
                },
                "decision": decision_data['decision'],
                "action": decision_data['action'],
                "notes": decision_data.get('notes', ''),
                "timestamp_obc": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        self.logger.info(f"Answer #{self.response_counter} generated: {decision_data['decision']}")
        return response

    def _determine_priority(self, decision):
        """DDetermines priority based on the decision"""
        priority_map = {
            "CONFIRM": "HIGH",
            "ISOLATE": "CRITICAL", 
            "MONITOR": "MEDIUM",
            "IGNORE": "LOW",
            "REQUEST_RETRANSMISSION": "MEDIUM",
            "ERROR": "HIGH"
        }
        return priority_map.get(decision, "MEDIUM")

    def generate_heartbeat(self):
        """Generates a heartbeat message from the OBC"""
        heartbeat = {
            "header": {
                "message_id": self.response_counter + 1,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "OBC",
                "message_type": "HEARTBEAT",
                "priority": "LOW",
                "version": "1.0"
            },
            "payload": {
                "system_status": "OPERATIONAL",
                "ai_model_status": "LOADED",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "processed_messages": self.response_counter
            }
        }
        
        self.logger.debug("Heartbeat OBC generated")
        return heartbeat

    def generate_error_response(self, original_message, error_description):
        """Generates an error response"""
        self.response_counter += 1
        
        error_response = {
            "header": {
                "message_id": self.response_counter,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "OBC",
                "message_type": "ERROR",
                "priority": "HIGH",
                "version": "1.0"
            },
            "payload": {
                "original_message_id": original_message.get('header', {}).get('message_id'),
                "error_type": "PROCESSING_ERROR",
                "error_description": error_description,
                "suggested_action": "RETRY_OR_CHECK_DATA",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        self.logger.error(f"Error response generated: {error_description}")
        return error_response

# Global instance
response_generator = OBCResponseGenerator()
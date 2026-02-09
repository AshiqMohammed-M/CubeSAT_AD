#!/usr/bin/env python3
"""
MAIN ENTRY POINT OF THE OBC SYSTEM
Main brain of the On-Board Computer module
"""
#completed
import os
import sys
import time
import json
import logging
from datetime import datetime

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

from interface.obc_message_handler import OBCMessageHandler
from interface.obc_response_generator import OBCResponseGenerator
from ai.ai_complex_inference import obc_ai

# ========== CORRECTED PATHS ==========
# All outputs in obc/
OBC_LOGS_DIR = os.path.join(project_root, "data", "obc", "logs")
OBC_OUTPUTS_DIR = os.path.join(project_root, "data", "obc", "outputs")
OBC_SYSTEM_DIR = os.path.join(project_root, "data", "obc", "system")

# Create OBC directories
os.makedirs(OBC_LOGS_DIR, exist_ok=True)
os.makedirs(OBC_OUTPUTS_DIR, exist_ok=True)
os.makedirs(OBC_SYSTEM_DIR, exist_ok=True)

# Logging configuration - now in obc/logs/
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OBC_LOGS_DIR, "obc_main_system.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OBC_Main")

class OBCSystem:
    def __init__(self):
        self.logger = logger
        self.is_running = False
        self.processed_messages = 0
        self.start_time = None
        self.system_id = f"OBC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Import components
        try:
            self.message_handler = OBCMessageHandler()
            self.response_generator = OBCResponseGenerator()
            self.ai_system = obc_ai
            
            # Check AI status
            ai_status = self.ai_system.get_model_info()
            self.logger.info(f"OBC system initialized successfully")
            self.logger.info(f"Logs directory: {OBC_LOGS_DIR}")
            self.logger.info(f"Outputs directory: {OBC_OUTPUTS_DIR}")
            self.logger.info(f"AI status: {ai_status['status']}")
            
        except ImportError as e:
            self.logger.error(f"Component import error: {e}")
            raise

    def start_system(self):
        """Starts the OBC system"""
        self.is_running = True
        self.start_time = datetime.now()
        self.processed_messages = 0
        
        ai_status = self.ai_system.get_model_info()
        
        self.logger.info("OBC SYSTEM STARTUP")
        self.logger.info(f"System ID: {self.system_id}")
        self.logger.info(f"Start time: {self.start_time}")
        self.logger.info(f"AI status: {ai_status['status']}")
        self.logger.info("Waiting for MCU messages...")
        
        # Save startup status
        self._save_system_status()
        
        return True

    def process_single_message(self, message_json):
        """
        Processes a single message (manual mode)
        """
        if not self.is_running:
            self.logger.warning("OBC system not started - Automatic startup")
            self.start_system()
        
        try:
            # Message processing
            response = self.message_handler.process_mcu_message(message_json)
            self.processed_messages += 1
            
            # Generate structured response
            structured_response = self.response_generator.generate_response(
                message_json,
                response['ai_analysis'],
                {
                    "decision": response['decision'],
                    "action": response['action'],
                    "notes": response.get('notes', '')
                }
            )
            
            self.logger.info(f"Message #{self.processed_messages} processed - Decision: {response['decision']}")
            
            # Save response in obc/outputs/
            self._save_response(structured_response, message_json)
            
            return structured_response
            
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
            error_response = self.response_generator.generate_error_response(
                message_json if isinstance(message_json, dict) else {"header": {}},
                str(e)
            )
            self._save_response(error_response, message_json, is_error=True)
            return error_response

    def _save_response(self, response, original_message, is_error=False):
        """Saves the response in obc/outputs/"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "error" if is_error else "response"
            filename = f"obc_{prefix}_{timestamp}_{self.processed_messages:06d}.json"
            filepath = os.path.join(OBC_OUTPUTS_DIR, filename)
            
            response_data = {
                "system_id": self.system_id,
                "timestamp": datetime.now().isoformat(),
                "processed_count": self.processed_messages,
                "response": response,
                "original_message_summary": {
                    "message_id": original_message.get('header', {}).get('message_id', 'unknown'),
                    "message_type": original_message.get('header', {}).get('message_type', 'unknown'),
                    "source": original_message.get('header', {}).get('source', 'unknown')
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(response_data, f, indent=2)
                
            self.logger.debug(f"Response saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Response save error: {e}")

    def _save_system_status(self):
        """Saves the system status in obc/system/"""
        try:
            status = self.get_system_status()
            filename = f"obc_system_status_{self.system_id}.json"
            filepath = os.path.join(OBC_SYSTEM_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(status, f, indent=2)
                
            self.logger.debug(f"System status saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Status save error: {e}")

    def run_continuous_mode(self, message_callback=None):
        """
        Executes continuous mode (permanent listening)
        """
        if not self.start_system():
            return
        
        self.logger.info("CONTINUOUS MODE ACTIVE")
        
        try:
            cycle_count = 0
            while self.is_running:
                cycle_count += 1
                
                # Periodic status save
                if cycle_count % 10 == 0:
                    self._log_system_status()
                    self._save_system_status()
                    
                    # Simulation: generate a test message periodically
                    from simulation.obc_simulate_incoming_data import create_sample_mcu_message
                    test_message = create_sample_mcu_message("SUMMARY")
                    self.process_single_message(test_message)
                
                # Wait before next check
                time.sleep(2)
                
        except KeyboardInterrupt:
            self.logger.info("Stop requested by user")
        except Exception as e:
            self.logger.error(f"Continuous mode error: {e}")
        finally:
            self.stop_system()

    def stop_system(self):
        """Stops the OBC system"""
        self.is_running = False
        end_time = datetime.now()
        runtime = end_time - self.start_time if self.start_time else None
        
        # Final status save
        final_status = self.get_system_status()
        final_status["shutdown_time"] = end_time.isoformat()
        final_status["total_runtime_seconds"] = runtime.total_seconds() if runtime else 0
        
        final_filename = f"obc_shutdown_report_{self.system_id}.json"
        final_filepath = os.path.join(OBC_SYSTEM_DIR, final_filename)
        
        try:
            with open(final_filepath, 'w') as f:
                json.dump(final_status, f, indent=2)
        except Exception as e:
            self.logger.error(f"Final report save error: {e}")
        
        self.logger.info("OBC SYSTEM SHUTDOWN")
        self.logger.info(f"System ID: {self.system_id}")
        self.logger.info(f"Messages processed: {self.processed_messages}")
        if runtime:
            self.logger.info(f"Execution time: {runtime}")
        self.logger.info(f"Final report: {final_filename}")

    def get_system_status(self):
        """Returns the system status"""
        ai_info = self.ai_system.get_model_info() if hasattr(self, 'ai_system') else {"status": "UNKNOWN"}
        
        status = {
            "system_id": self.system_id,
            "is_running": self.is_running,
            "ai_loaded": self.ai_system.is_loaded if hasattr(self, 'ai_system') else False,
            "ai_status": ai_info['status'],
            "processed_messages": self.processed_messages,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "current_time": datetime.now().isoformat(),
            "directories": {
                "logs": OBC_LOGS_DIR,
                "outputs": OBC_OUTPUTS_DIR,
                "system": OBC_SYSTEM_DIR
            }
        }
        
        if hasattr(self, 'ai_system') and self.ai_system.is_loaded:
            status.update({
                "ai_model_info": self.ai_system.get_model_info()
            })
            
        return status

    def _log_system_status(self):
        """Logs the periodic system status"""
        status = self.get_system_status()
        self.logger.info(f"System status - Messages: {status['processed_messages']}, "
                        f"AI: {status['ai_status']}")

# Instance globale
obc_system = OBCSystem()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OBC System EPS Guardian")
    parser.add_argument("--mode", choices=["single", "continuous"], default="single",
                       help="Execution mode (single/continuous)")
    parser.add_argument("--message", type=str, help="JSON message to process (single mode)")
    parser.add_argument("--message-file", type=str, help="JSON file containing the message")
    
    args = parser.parse_args()
    
    system = OBCSystem()
    
    try:
        if args.mode == "single":
            # Single processing mode
            message_data = None
            
            if args.message_file:
                # Load from file
                with open(args.message_file, 'r') as f:
                    message_data = json.load(f)
            elif args.message:
                # Load from JSON string
                message_data = json.loads(args.message)
            else:
                # Default message
                from simulation.obc_simulate_incoming_data import create_sample_mcu_message
                message_data = create_sample_mcu_message("SUMMARY")
            
            print("OBC SINGLE PROCESSING MODE")
            print(f"Outputs directory: {OBC_OUTPUTS_DIR}")
            response = system.process_single_message(message_data)
            print("\nGenerated response:")
            print(json.dumps(response, indent=2))
            
        elif args.mode == "continuous":
            # Continuous mode
            print("OBC CONTINUOUS MODE")
            print(f"System ID: {system.system_id}")
            print(f"Logs directory: {OBC_LOGS_DIR}")
            print(f"Outputs directory: {OBC_OUTPUTS_DIR}")
            print("Ctrl+C to stop")
            system.run_continuous_mode()
            
        else:
            # Demo mode
            print("OBC SYSTEM EPS GUARDIAN")
            print("=" * 50)
            print(f"Logs directory: {OBC_LOGS_DIR}")
            print(f"Outputs directory: {OBC_OUTPUTS_DIR}")
            print(f"System directory: {OBC_SYSTEM_DIR}")
            print("=" * 50)
            
            # Test with example message
            from simulation.obc_simulate_incoming_data import create_sample_mcu_message
            test_message = create_sample_mcu_message("SUMMARY")
            
            print("Automatic test with example message:")
            response = system.process_single_message(test_message)
            print("\nGenerated response:")
            print(json.dumps(response, indent=2))
            
            print("\nComplete system status:")
            print(json.dumps(system.get_system_status(), indent=2))
            
    except Exception as e:
        logger.error(f"Execution error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

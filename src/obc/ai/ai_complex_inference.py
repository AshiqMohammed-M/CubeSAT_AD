#!/usr/bin/env python3
"""
AI COMPLEX OBC INFERENCE
Anomaly detection with the trained LSTM Autoencoder model
"""
#completed
import os
import numpy as np
import json
import logging
import joblib

# Configuring paths
current_dir = os.path.dirname(os.path.abspath(__file__))
obc_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(obc_dir)
project_root = os.path.dirname(src_dir)

# ========== CORRECTED PATHS ==========
# All outputs in obc/
OBC_AI_DIR = os.path.join(project_root, "data", "ai_models", "model_complex")
OBC_LOGS_DIR = os.path.join(project_root, "data", "obc", "logs")
OBC_OUTPUTS_DIR = os.path.join(project_root, "data", "obc", "outputs")
DATA_DIR = os.path.join(project_root, "data", "ai_training_base")  # Corrected

# Creating OBC folders
os.makedirs(OBC_LOGS_DIR, exist_ok=True)
os.makedirs(OBC_OUTPUTS_DIR, exist_ok=True)

# Logging - now in obc/logs/
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OBC_LOGS_DIR, "obc_ai_inference.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OBC_AI_Inference")

class OBC_AI:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.thresholds = None
        self.is_loaded = False
        self.simulation_mode = True
        self.model_available = False
        
        logger.info("Initializing OBC AI")
        logger.info(f"Logs: {OBC_LOGS_DIR}")
        logger.info(f"Outputs: {OBC_OUTPUTS_DIR}")
        self.load_model_and_thresholds()

    def load_model_and_thresholds(self):
        """Load the model and thresholds"""
        try:
            # Loading thresholds
            thresholds_path = os.path.join(OBC_AI_DIR, "ai_thresholds.json")
            logger.info(f"Searching for thresholds: {thresholds_path}")
            
            if not os.path.exists(thresholds_path):
                logger.error(f"Thresholds not found: {thresholds_path}")
                # Create default thresholds
                self.thresholds = {
                    "normal_threshold": 0.001,
                    "warning_threshold": 0.002,
                    "critical_threshold": 0.003,
                    "training_stats": {
                        "mean_error": 0.0005,
                        "std_error": 0.0005,
                        "min_error": 0.0001,
                        "max_error": 0.005
                    }
                }
                logger.info("Default thresholds created")
            else:
                with open(thresholds_path, 'r') as f:
                    thresholds_data = json.load(f)
                self.thresholds = thresholds_data["anomaly_thresholds"]
                logger.info("Anomaly thresholds loaded")
            
            # Loading model with compile=False
            model_path = os.path.join(OBC_AI_DIR, "ai_model_lstm_autoencoder.h5")
            
            if os.path.exists(model_path):
                try:
                    from tensorflow.keras.models import load_model
                    
                    print(f"Attempting to load model: {model_path}")
                    print(f"Size: {os.path.getsize(model_path)} bytes")
                    
                    # SOLUTION 1: Load without compilation
                    try:
                        self.model = load_model(model_path, compile=False)
                        self.simulation_mode = False
                        self.model_available = True
                        print("SUCCESS: LSTM model loaded (compile=False)")
                        logger.info("LSTM Autoencoder model successfully loaded (compile=False)")
                        
                    except Exception as e:
                        print(f"Error with compile=False: {e}")
                        self._fallback_to_simulation()
                            
                except ImportError as e:
                    print(f"TensorFlow not available: {e}")
                    self._fallback_to_simulation()
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    self._fallback_to_simulation()
            else:
                print("Model not found on disk")
                self._fallback_to_simulation()
            
            # Loading or creating scaler - CORRECTED PATH
            scaler_path = os.path.join(DATA_DIR, "ai_sequence_scaler.pkl")
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    logger.info("Scaler loaded")
                except Exception as e:
                    logger.warning(f"Error loading scaler: {e} - creating default scaler")
                    self._create_default_scaler()
            else:
                logger.warning("Scaler not found - creating default scaler")
                self._create_default_scaler()
            
            self.is_loaded = True
            logger.info("OBC AI successfully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error loading AI: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _fallback_to_simulation(self):
        """Switch to simulation mode"""
        self.model = None
        self.simulation_mode = True
        self.model_available = False
        print("USING: Simulation mode active")
        logger.warning("Simulation mode active - TensorFlow model not available")

    def _create_default_scaler(self):
        """Create a default scaler"""
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        # Fit with realistic simulated data
        dummy_data = np.array([
            [7.4, 1.2, 35.0, 7.8, 0.8, 15.2, 1.5]  # Typical values
        ] * 100) + np.random.randn(100, 7) * 0.1
        self.scaler.fit(dummy_data)

    def analyze_sequence(self, sequence_data):
        """
        Analyzes a time sequence and returns the anomaly level
        """
        if not self.is_loaded:
            return self._simulate_analysis(sequence_data)
        
        # CORRECTION: Use the real model if available
        if self.model is not None:
            try:
                # Data verification and preparation
                if sequence_data.shape != (30, 7):
                    logger.warning(f"Incorrect shape: {sequence_data.shape}, expected: (30, 7)")
                    return self._simulate_analysis(sequence_data)
                
                # Data cleaning
                sequence_data = np.nan_to_num(sequence_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # CORRECTION: Normalization - corrected method
                print(f"Shape before normalization: {sequence_data.shape}")
                
                # Corrected method: Proper reshape for scaler
                sequence_normalized = self.scaler.transform(sequence_data.reshape(-1, 7))
                sequence_normalized = sequence_normalized.reshape(1, 30, 7)
                print(f"Shape after normalization: {sequence_normalized.shape}")
                
                # Prediction with LSTM model
                print("USING THE REAL LSTM MODEL...")
                reconstructed = self.model.predict(sequence_normalized, verbose=0)
                
                # Calculation of reconstruction error
                reconstruction_error = np.mean(np.square(sequence_normalized - reconstructed))
                
                print(f"LSTM reconstruction error: {reconstruction_error:.6f}")
                print(f"Normal threshold: {self.thresholds['normal_threshold']:.6f}")
                print(f"Warning threshold: {self.thresholds['warning_threshold']:.6f}")
                print(f"Critical threshold: {self.thresholds['critical_threshold']:.6f}")
                
                # DDetermination of anomaly level
                ai_level, confidence = self._classify_anomaly(reconstruction_error)
                
                result = {
                    "ai_score": float(reconstruction_error),
                    "ai_level": ai_level,
                    "confidence": confidence,
                    "reconstruction_error": float(reconstruction_error),
                    "thresholds_used": {
                        "normal": self.thresholds["normal_threshold"],
                        "warning": self.thresholds["warning_threshold"],
                        "critical": self.thresholds["critical_threshold"]
                    },
                    "simulated": False,
                    "model_used": "LSTM Autoencoder"
                }
                
                logger.info(f"Analyze AI LSTM - Score: {reconstruction_error:.6f}, Level: {ai_level}")
                
                # SAVE RESULT IN OBC/OUTPUTS
                self._save_analysis_result(result, sequence_data)
                
                return result
                
            except Exception as e:
                logger.error(f"Error AI LSTM analysis: {e} - switching to simulation")
                print(f"LSTM ERROR: {e}")
                import traceback
                traceback.print_exc()
                return self._simulate_analysis(sequence_data)
        else:
            # Simulation mode
            print("USING SIMULATION MODE")
            return self._simulate_analysis(sequence_data)

    def _save_analysis_result(self, result, sequence_data):
        """Saves the analysis results in obc/outputs/"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Result JSON file
            result_filename = f"obc_ai_analysis_{timestamp}.json"
            result_path = os.path.join(OBC_OUTPUTS_DIR, result_filename)
            
            result_data = {
                "timestamp": datetime.now().isoformat(),
                "analysis_result": result,
                "sequence_info": {
                    "shape": sequence_data.shape,
                    "data_points": len(sequence_data)
                }
            }
            
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            logger.info(f"Result saved: {result_path}")
            
        except Exception as e:
            logger.error(f"Error saving result: {e}")

    def _simulate_analysis(self, sequence_data):
        """Simulates AI analysis when the model is not available"""
        logger.info("Using simulated analysis")
        
        try:
            # Simulation based on average temperature and other metrics
            avg_temp = np.mean(sequence_data[:, 2])  # T_batt
            avg_current = np.mean(sequence_data[:, 1])  # I_batt
            avg_voltage = np.mean(sequence_data[:, 0])  # V_batt
            
            # Simulated anomaly detection logic
            if avg_temp > 60 or avg_current > 3.0:
                result = {
                    "ai_score": 0.95,
                    "ai_level": "CRITICAL",
                    "confidence": "HIGH",
                    "reconstruction_error": 0.95,
                    "simulated": True,
                    "model_used": "Simulation Rules",
                    "metrics": {
                        "avg_temp": float(avg_temp),
                        "avg_current": float(avg_current),
                        "avg_voltage": float(avg_voltage)
                    }
                }
            elif avg_temp > 45 or avg_current > 2.0 or avg_voltage < 3.2:
                result = {
                    "ai_score": 0.75,
                    "ai_level": "WARNING",
                    "confidence": "MEDIUM", 
                    "reconstruction_error": 0.75,
                    "simulated": True,
                    "model_used": "Simulation Rules",
                    "metrics": {
                        "avg_temp": float(avg_temp),
                        "avg_current": float(avg_current),
                        "avg_voltage": float(avg_voltage)
                    }
                }
            else:
                result = {
                    "ai_score": 0.1,
                    "ai_level": "NORMAL",
                    "confidence": "HIGH",
                    "reconstruction_error": 0.1,
                    "simulated": True,
                    "model_used": "Simulation Rules",
                    "metrics": {
                        "avg_temp": float(avg_temp),
                        "avg_current": float(avg_current),
                        "avg_voltage": float(avg_voltage)
                    }
                }
            
            # Save even in simulation mode
            self._save_analysis_result(result, sequence_data)
            return result
            
        except Exception as e:
            logger.error(f"Error in simulated analysis: {e}")
            # Fallback if error
            result = {
                "ai_score": 0.1,
                "ai_level": "NORMAL",
                "confidence": "MEDIUM",
                "reconstruction_error": 0.1,
                "simulated": True,
                "model_used": "Simulation Rules",
                "error": str(e)
            }
            self._save_analysis_result(result, sequence_data)
            return result

    def _classify_anomaly(self, error):
        """Classifies the anomaly based on the thresholds"""
        if error <= self.thresholds["normal_threshold"]:
            return "NORMAL", "HIGH"
        elif error <= self.thresholds["warning_threshold"]:
            return "WARNING", "MEDIUM"
        else:
            return "CRITICAL", "HIGH"

    def get_model_info(self):
        """Returns the model information"""
        if not self.is_loaded:
            return {"status": "NOT_LOADED"}
        
        if self.model is not None:
            status = "LOADED"
            model_arch = "LSTM Autoencoder"
        else:
            status = "SIMULATION"
            model_arch = "Simulation Rules"
        
        return {
            "status": status,
            "model_architecture": model_arch,
            "input_shape": [30, 7],
            "thresholds": self.thresholds,
            "training_stats": self.thresholds.get("training_stats", {}),
            "scaler_available": self.scaler is not None,
            "model_available": self.model is not None,
            "simulation_mode": self.model is None
        }

# Global instance for easy use
obc_ai = OBC_AI()

def analyze_sensor_sequence(sequence_data):
    """
    Utility function to analyze a sensor sequence
    """
    return obc_ai.analyze_sequence(sequence_data)

if __name__ == "__main__":
    # Test the AI
    ai = OBC_AI()
    print(f"AI loaded: {ai.is_loaded}")
    print(f"Model available: {ai.model is not None}")
    print(f"Simulation mode: {ai.model is None}")
    print(f"Full status: {json.dumps(ai.get_model_info(), indent=2)}")
    
    # Test with realistic data
    print("\nTest with simulated data:")
    test_sequence = np.array([
        [7.4, 1.2, 35.0, 7.8, 0.8, 15.2, 1.5]  # Basic values
    ] * 30) + np.random.randn(30, 7) * 0.1
    
    result = ai.analyze_sequence(test_sequence)
    print("Test result:")
    print(json.dumps(result, indent=2))

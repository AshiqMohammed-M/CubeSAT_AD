#!/usr/bin/env python3
"""
AI Inference System for EPS GUARDIAN - CORRECTED VERSION
"""
#completed
import tensorflow as tf
import numpy as np
import os
import json
import joblib
import pandas as pd
from datetime import datetime

class AIModelInference:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.model_dir = os.path.join(self.base_dir, "data", "ai_models", "model_simple")
        self.training_dir = os.path.join(self.base_dir, "data", "training_data")
        self.output_dir = os.path.join(self.base_dir, "data", "mcu", "outputs", "ai_inference_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.scaler = None
        self.feature_names = None
        self.thresholds = None
        self.inference_results = []
        
        print("AIModelInference initialized")
    
    def load_artifacts(self):
        """Load all AI artifacts"""
        try:
            # Load the scaler
            scaler_path = os.path.join(self.training_dir, "ai_scaler.pkl")
            self.scaler = joblib.load(scaler_path)
            
            # Load feature names
            features_path = os.path.join(self.training_dir, "ai_feature_names.npy")
            self.feature_names = np.load(features_path, allow_pickle=True).tolist()
            
            # Load thresholds
            thresholds_path = os.path.join(self.model_dir, "ai_thresholds.json")
            with open(thresholds_path, 'r') as f:
                thresholds_data = json.load(f)
                self.thresholds = thresholds_data["thresholds"]
            
            # Load the TFLite model
            tflite_path = os.path.join(self.model_dir, "ai_autoencoder.tflite")
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print("All artifacts loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            return False
    
    def predict_from_normalized(self, normalized_features):
        """Direct prediction with already normalized features"""
        try:
            # Prepare input
            input_data = normalized_features.astype(np.float32)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Calculate reconstruction error
            reconstruction_error = np.mean(np.square(input_data - output_data))
            
            # DDetermine anomaly level
            anomaly_level = "NORMAL"
            if reconstruction_error >= self.thresholds["critical"]:
                anomaly_level = "CRITICAL"
            elif reconstruction_error >= self.thresholds["warning"]:
                anomaly_level = "WARNING"
            
            return reconstruction_error, anomaly_level
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None
    
    def test_real_scenarios(self):
        """Test with REAL data from the training dataset"""
        try:
            # Load normalized training data
            train_data_path = os.path.join(self.training_dir, "ai_train_data.npy")
            X_train = np.load(train_data_path)
            
            print(f"Training data loaded: {X_train.shape}")
            
            # Test some real samples
            test_indices = [0, 100, 500, 1000, 2000]  # Different samples
            
            for i, idx in enumerate(test_indices):
                if idx < len(X_train):
                    sample = X_train[idx:idx+1]  # Take a sample
                    
                    # Make prediction
                    error, level = self.predict_from_normalized(sample)
                    
                    if error is not None:
                        print(f"\nTest {i+1} (sample {idx}):")
                        print(f"  Reconstruction error: {error:.6f}")
                        print(f"  Anomaly level: {level}")
                        print(f"  Thresholds: Normal<{self.thresholds['normal']:.6f}")
                        
                        # Save the result
                        result = {
                            "timestamp": datetime.now().isoformat(),
                            "sample_index": idx,
                            "reconstruction_error": float(error),
                            "anomaly_level": level,
                            "is_training_sample": True
                        }
                        self.inference_results.append(result)
            
            return True
            
        except Exception as e:
            print(f"Error testing real-world scenarios: {e}")
            return False
    
    def test_synthetic_normal(self):
        """Test with SYNTHETIC NORMAL data"""
        try:
            # Create typical normalized features (close to the mean)
            normal_features = np.random.normal(0.5, 0.1, (1, len(self.feature_names)))
            normal_features = np.clip(normal_features, 0, 1)  # Keep between 0 and 1
            
            error, level = self.predict_from_normalized(normal_features)
            
            if error is not None:
                print(f"\nTest SYNTHETIC NORMAL:")
                print(f"  Reconstruction error: {error:.6f}")
                print(f"  Anomaly level: {level}")
                
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "sample_type": "synthetic_normal",
                    "reconstruction_error": float(error),
                    "anomaly_level": level,
                    "is_training_sample": False
                }
                self.inference_results.append(result)
            
            return True
            
        except Exception as e:
            print(f"Error testing synthetic data: {e}")
            return False
    
    def save_results(self):
        """Save all results """
        try:
            # CSV
            csv_path = os.path.join(self.output_dir, "ai_inference_summary.csv")
            df = pd.DataFrame(self.inference_results)
            df.to_csv(csv_path, index=False)
            
            # JSON
            json_path = os.path.join(self.output_dir, "ai_inference_test_cases.json")
            with open(json_path, 'w') as f:
                json.dump(self.inference_results, f, indent=2)
            
            print(f"\nRResults saved:")
            print(f"  - CSV: {csv_path}")
            print(f"  - JSON: {json_path}")
            
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    inference = AIModelInference()
    
    if inference.load_artifacts():
        print("\nAI INFERENCE SYSTEM READY")
        print("=" * 40)
        
        # 1. Test with REAL training data
        print("\n1. TESTS WITH REAL DATA:")
        inference.test_real_scenarios()
        
        # 2. Test with SYNTHETIC data
        print("\n2. TESTS WITH SYNTHETIC DATA:")
        inference.test_synthetic_normal()
        
        # 3. Save results
        inference.save_results()
        
        print("\nInference tests completed successfully!")
        
    else:
        print("Unable to load AI artifacts")

if __name__ == "__main__":
    main()

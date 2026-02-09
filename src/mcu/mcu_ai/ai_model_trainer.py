#!/usr/bin/env python3
"""
AI model trainer for EPS GUARDIAN
Trains a lightweight autoencoder on normal data for anomaly detection
"""
#completed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# Disable TensorFlow warnings for a cleaner output
tf.get_logger().setLevel('ERROR')

class AIModelTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # CORRECTED PATHS
        self.model_dir = os.path.join(self.base_dir, "data", "ai_models", "model_simple")
        self.training_dir = os.path.join(self.base_dir, "data", "training_data")  # ← CORRECTED HERE
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)
        
        self.thresholds = {
            "normal": 0.05,
            "warning": 0.15,
            "critical": 0.25
        }
        print(f"Base directory: {self.base_dir}")
        print(f"Model directory: {self.model_dir}")
        print(f"Training directory: {self.training_dir}")
        
    def load_training_data(self):
        """Load prepared training data"""
        data_path = os.path.join(self.training_dir, "ai_train_data.npy")
        feature_path = os.path.join(self.training_dir, "ai_feature_names.npy")
        
        print(f"Looking for data: {data_path}")
        
        if not os.path.exists(data_path):
            # List files for debugging
            parent_dir = os.path.dirname(data_path)
            if os.path.exists(parent_dir):
                print(f"Files in {parent_dir}:")
                for f in os.listdir(parent_dir):
                    print(f"   - {f}")
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        print("Loading training data...")
        X_train = np.load(data_path)
        feature_names = np.load(feature_path, allow_pickle=True)
        
        print(f"Data loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Features: {list(feature_names)}")
        
        return X_train, feature_names
    
    def create_autoencoder(self, input_dim):
        """
        Creates a lightweight autoencoder for ESP32
        Architecture: 10 → 8 → 4 → 2 → 4 → 8 → 10
        """
        print(f" Creating autoencoder (input_dim: {input_dim})")
        
        # Encoder
        encoder = keras.Sequential([
            layers.Dense(8, activation='relu', input_shape=(input_dim,)),
            layers.Dense(4, activation='relu'),
            layers.Dense(2, activation='relu', name='bottleneck')  # Strong compression
        ], name='encoder')
        
        # Decoder
        decoder = keras.Sequential([
            layers.Dense(4, activation='relu', input_shape=(2,)),
            layers.Dense(8, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')  # Sigmoid because normalized data [0,1]
        ], name='decoder')
        
        # Full Autoencoder
        autoencoder = keras.Sequential([
            encoder,
            decoder
        ], name='autoencoder')
        
        # Compilation
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean Squared Error for reconstruction
            metrics=['mae']
        )
        
        # Display architecture
        autoencoder.summary()
        
        return autoencoder, encoder, decoder
    
    def train_model(self, X_train, validation_split=0.2, epochs=100, batch_size=32):
        """Train the autoencoder model"""
        print("\n Starting training...")
        print(f"   - Samples: {X_train.shape[0]}")
        print(f"   - Features: {X_train.shape[1]}")
        print(f"   - Validation split: {validation_split}")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        
        # Creating the model
        self.model, self.encoder, self.decoder = self.create_autoencoder(X_train.shape[1])
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Training
        self.history = self.model.fit(
            X_train, X_train,  # Autoencoder: input = output
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        print("Training completed!")
        return self.history
    
    def calculate_anomaly_threshold(self, X_train):
        """Calculate the anomaly threshold based on reconstruction error"""
        print("\n Calculating anomaly threshold...")
        
        # Predictions on training data
        predictions = self.model.predict(X_train, verbose=0)
        
        # Reconstruction error (MSE)
        reconstruction_errors = np.mean(np.square(X_train - predictions), axis=1)
        
        # Error statistics
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        
        # Thresholds based on normal distribution
        self.thresholds = {
            "normal": mean_error + std_error,      # 1 sigma
            "warning": mean_error + 2 * std_error, # 2 sigma
            "critical": mean_error + 3 * std_error # 3 sigma
        }
        
        print(f"Reconstruction error statistics:")
        print(f"   - Mean: {mean_error:.6f}")
        print(f"   - Standard deviation: {std_error:.6f}")
        print(f"   - Min: {np.min(reconstruction_errors):.6f}")
        print(f"   - Max: {np.max(reconstruction_errors):.6f}")
        print(f"Anomaly thresholds:")
        print(f"   - Normal: < {self.thresholds['normal']:.6f}")
        print(f"   - Warning: {self.thresholds['normal']:.6f} - {self.thresholds['warning']:.6f}")
        print(f"   - Critical: > {self.thresholds['warning']:.6f}")
        
        return reconstruction_errors
    
    def save_model(self, feature_names):
        """Save the model and artifacts in the dedicated folder"""
        print(f"Model directory: {self.model_dir}")
        
        # 1. Save the complete Keras model
        model_path = os.path.join(self.model_dir, "ai_autoencoder.h5")
        self.model.save(model_path)
        print(f"Keras model saved: {model_path}")
        
        # 2. Save the encoder (useful for feature extraction)
        encoder_path = os.path.join(self.model_dir, "ai_encoder.h5")
        self.encoder.save(encoder_path)
        print(f"Encoder saved: {encoder_path}")
        
        # 3. Conversion to TensorFlow Lite
        tflite_path = os.path.join(self.model_dir, "ai_autoencoder.tflite")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimizations for ESP32
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Precision reduction
        
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Check model size
        model_size_kb = len(tflite_model) / 1024
        print(f"TFLite model saved: {tflite_path}")
        print(f"TFLite model size: {model_size_kb:.1f} KB")
        
        # 4. Save thresholds
        thresholds_path = os.path.join(self.model_dir, "ai_thresholds.json")
        thresholds_data = {
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
            "calculation_date": datetime.now().isoformat(),
            "model_size_kb": float(model_size_kb),
            "features": list(feature_names)
        }
        
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds_data, f, indent=2)
        print(f"Thresholds saved: {thresholds_path}")
        
        # 5. Save model information
        model_info_path = os.path.join(self.model_dir, "ai_model_info.json")
        model_info = {
            "model_name": "EPS_Guardian_Autoencoder",
            "version": "1.0.0",
            "creation_date": datetime.now().isoformat(),
            "architecture": {
                "input_dim": self.model.input_shape[1],
                "encoder_layers": [8, 4, 2],
                "decoder_layers": [4, 8, self.model.input_shape[1]],
                "total_parameters": self.model.count_params()
            },
            "training_config": {
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "loss_function": "mse",
                "metrics": ["mae"]
            },
            "features_used": list(feature_names),
            "model_size_kb": float(model_size_kb),
            "esp32_compatible": model_size_kb <= 50
        }
        
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"Model information saved: {model_info_path}")
        
        # 6. Save model summary (FIXED)
        summary_path = os.path.join(self.model_dir, "ai_model_summary.txt")
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:  # ← ENCODING UTF-8 ADDED
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            print(f"Model summary saved: {summary_path}")
        except Exception as e:
            print(f"Note: Summary not saved (encoding error: {e})")
            # Create a simplified summary
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"EPS Guardian Autoencoder\n")
                f.write(f"Input: {self.model.input_shape[1]} features\n")
                f.write(f"Architecture: 18 -> 8 -> 4 -> 2 -> 4 -> 8 -> 18\n")
                f.write(f"Parameters: {self.model.count_params()}\n")
                f.write(f"TFLite size: {model_size_kb:.1f} KB\n")
            print(f"Simplified summary saved: {summary_path}")
        
        return model_path, tflite_path
    
    def plot_training_history(self):
        """Generate training plots"""
        if self.history is None:
            print("No training history available")
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss
            ax1.plot(self.history.history['loss'], label='Train Loss')
            ax1.plot(self.history.history['val_loss'], label='Val Loss')
            ax1.set_title('Model Loss')
            ax1.set_ylabel('Loss')
            ax1.set_xlabel('Epoch')
            ax1.legend()
            ax1.grid(True)
            
            # MAE
            ax2.plot(self.history.history['mae'], label='Train MAE')
            ax2.plot(self.history.history['val_mae'], label='Val MAE')
            ax2.set_title('Model MAE')
            ax2.set_ylabel('MAE')
            ax2.set_xlabel('Epoch')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save
            plot_path = os.path.join(self.model_dir, "ai_training_history.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Training plots saved: {plot_path}")
        except Exception as e:
            print(f"Error creating training plots: {e}")
    
    def evaluate_model_size(self):
        """Evaluate if the model is suitable for ESP32"""
        tflite_path = os.path.join(self.model_dir, "ai_autoencoder.tflite")
        
        if not os.path.exists(tflite_path):
            print("TFLite model not found for size evaluation")
            return False
        
        model_size_kb = os.path.getsize(tflite_path) / 1024
        target_size_kb = 50  # ESP32 target
        
        print(f"\n MODEL SIZE EVALUATION:")
        print(f"   - Current size: {model_size_kb:.1f} KB")
        print(f"   - ESP32 target: {target_size_kb} KB")
        
        if model_size_kb <= target_size_kb:
            print("  MODEL SUITABLE FOR ESP32! ")
            return True
        else:
            print(f" Model too large: {model_size_kb - target_size_kb:.1f} KB above target")
            print("Suggestions: Reduce architecture or use INT8 optimization")
            return False
    
    def run(self, epochs=100, batch_size=32):
        """Run the complete training"""
        print("Starting AI model training")
        print("=" * 60)
        
        try:
            # 1. Load data
            X_train, feature_names = self.load_training_data()
            
            # 2. Train model
            history = self.train_model(X_train, epochs=epochs, batch_size=batch_size)
            
            # 3. Calculate anomaly thresholds
            reconstruction_errors = self.calculate_anomaly_threshold(X_train)
            
            # 4. Save models
            model_path, tflite_path = self.save_model(feature_names)
            
            # 5. Visualization
            self.plot_training_history()
            
            # 6. Size evaluation
            esp32_compatible = self.evaluate_model_size()
            
            print("\n AI training completed successfully!")
            
            # Final report
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            print(f"\n FINAL REPORT:")
            print(f"   - Final loss: {final_loss:.6f}")
            print(f"   - Validation loss: {final_val_loss:.6f}")
            print(f"   - Features used: {len(feature_names)}")
            print(f"   - Model saved in: {self.model_dir}")
            print(f"   - ESP32 compatible: {'YES' if esp32_compatible else 'NO'}")
            
            return True
            
        except Exception as e:
            print(f" Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    trainer = AIModelTrainer()
    
    # Training configuration (adjustable)
    success = trainer.run(
        epochs=100,        # Number of epochs
        batch_size=32      # Batch size
    )
    
    if success:
        print("\n AI training completed successfully!")
        print(" All files are organized in: data/ai_models/model_simple/")
        print(" You can now test inference with ai_model_inference.py")
    else:
        print("\n AI training failed.")

if __name__ == "__main__":
    main()

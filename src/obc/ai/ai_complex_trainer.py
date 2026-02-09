#!/usr/bin/env python3
"""
OBC Complex AI Model Training
LSTM Autoencoder for Temporal Anomaly Detection
"""
#completed
import os
import numpy as np
import json
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib

def find_project_root():
    """Find the project root directory robustly"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different depth levels
    possible_roots = [
        os.path.dirname(os.path.dirname(os.path.dirname(current_dir))),  # src/obc/ai → root
        os.path.join(current_dir, "..", "..", ".."),  # Alternative
        current_dir  
    ]
    
    for root in possible_roots:
        root = os.path.abspath(root)
        # Check if data/ai_training_base exists
        data_training = os.path.join(root, "data", "ai_training_base")
        if os.path.exists(data_training):
            return root
    
    # If nothing found, use the current directory
    return current_dir

# Determining paths
PROJECT_ROOT = find_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ai_training_base")  # CORRECTED: ai_training_base instead of mcu/training
OBC_AI_DIR = os.path.join(PROJECT_ROOT, "data", "ai_models", "model_complex")
OBC_LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "mcu", "logs")

# Creation of files
os.makedirs(OBC_AI_DIR, exist_ok=True)
os.makedirs(OBC_LOGS_DIR, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OBC_LOGS_DIR, "obc_ai_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OBC_AI_Trainer")

logger.info(f"Detected root directory: {PROJECT_ROOT}")
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"OBC AI directory: {OBC_AI_DIR}")

class OBCAITrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.training_history = None
        self.anomaly_thresholds = {}
        
        # Initialization of attributes to avoid errors
        self.sequences = None
        self.labels = None
        self.dataset_config = None
        self.training_sequences = None
        self.x_train = None
        self.x_val = None
        
        logger.info("Initialization of OBC AI Trainer")

    def load_training_data(self):
        """Load training data from ai_training_base"""
        logger.info("Loading training data...")
        
        try:
            # Check if the directory exists
            if not os.path.exists(DATA_DIR):
                logger.error(f"Data directory not found: {DATA_DIR}")
                logger.info("Contents of the root directory:")
                for item in os.listdir(PROJECT_ROOT):
                    logger.info(f"  - {item}")
                return False
            
            # Check files in the directory
            data_files = os.listdir(DATA_DIR)
            logger.info(f"Files found in {DATA_DIR}: {data_files}")
            
            # Loading normalized sequences
            sequences_path = os.path.join(DATA_DIR, "ai_sequence_data.npy")
            if not os.path.exists(sequences_path):
                logger.error(f"File not found: {sequences_path}")
                return False
                
            self.sequences = np.load(sequences_path)
            logger.info(f"Sequences loaded: {self.sequences.shape}")
            
            # Loading labels
            labels_path = os.path.join(DATA_DIR, "ai_sequence_labels.npy")
            if not os.path.exists(labels_path):
                logger.error(f"File not found: {labels_path}")
                return False
                
            self.labels = np.load(labels_path)
            logger.info(f"Labels loaded: {len(self.labels)}")
            
            # Analysis of labels
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            logger.info("Distribution of labels:")
            for label, count in zip(unique_labels, counts):
                logger.info(f"  {label}: {count} sequences ({count/len(self.labels)*100:.1f}%)")
            
            # Loading scaler
            scaler_path = os.path.join(DATA_DIR, "ai_sequence_scaler.pkl")
            if not os.path.exists(scaler_path):
                logger.error(f"File not found: {scaler_path}")
                return False
                
            self.scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded")
            
            # Loading configuration
            config_path = os.path.join(DATA_DIR, "dataset_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.dataset_config = json.load(f)
                logger.info("Dataset configuration loaded")
            else:
                logger.warning("Dataset configuration not found")
                self.dataset_config = {}
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def prepare_data(self):
        """Prepare data for training"""
        logger.info("Preparing data...")
        
        # Verify that data is loaded
        if self.sequences is None or self.labels is None:
            logger.error("Data not loaded - Call load_training_data() first")
            return False
        
        # Use only NORMAL sequences for autoencoder training
        normal_indices = np.where(self.labels == "NORMAL")[0]
        
        if len(normal_indices) == 0:
            logger.warning("No NORMAL sequences found, using all data")
            normal_indices = np.arange(len(self.sequences))
        
        self.training_sequences = self.sequences[normal_indices]
        logger.info(f"Training sequences (normal): {self.training_sequences.shape}")
        
        # Train/validation split (80/20)
        split_idx = int(0.8 * len(self.training_sequences))
        self.x_train = self.training_sequences[:split_idx]
        self.x_val = self.training_sequences[split_idx:]
        
        logger.info(f"Train: {self.x_train.shape}, Validation: {self.x_val.shape}")
        return True

    def build_lstm_autoencoder(self):
        """Build the model LSTM Autoencoder"""
        logger.info("Building LSTM Autoencoder model...")
        
        # Model parameters
        timesteps = self.sequences.shape[1]
        n_features = self.sequences.shape[2]
        encoding_dim = 32  # Latent space dimension
        
        logger.info(f"Architecture: {timesteps} timesteps x {n_features} features")
        logger.info(f"Latent dimension: {encoding_dim}")
        
        # Encoder
        inputs = Input(shape=(timesteps, n_features))
        encoded = LSTM(64, activation='relu', return_sequences=True, name="encoder_lstm1")(inputs)
        encoded = LSTM(32, activation='relu', return_sequences=False, name="encoder_lstm2")(encoded)
        encoded = Dense(encoding_dim, activation='relu', name="bottleneck")(encoded)
        
        # Decoder
        decoded = RepeatVector(timesteps, name="repeat_vector")(encoded)
        decoded = LSTM(32, activation='relu', return_sequences=True, name="decoder_lstm1")(decoded)
        decoded = LSTM(64, activation='relu', return_sequences=True, name="decoder_lstm2")(decoded)
        decoded = TimeDistributed(Dense(n_features), name="output")(decoded)
        
        # Complete model
        self.model = Model(inputs, decoded, name="lstm_autoencoder")
        
        # Compilation
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Model compiled successfully")
        
        # Model summary
        total_params = self.model.count_params()
        logger.info(f"Total number of parameters: {total_params:,}")
        
        return True

    def train_model(self, epochs=100, batch_size=32):
        """Train the model"""
        logger.info("Starting training...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
        ]
        
        logger.info(f"Training parameters: {epochs} epochs, batch_size={batch_size}")
        
        # Training
        self.training_history = self.model.fit(
            self.x_train, self.x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.x_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Analyze results
        final_train_loss = self.training_history.history['loss'][-1]
        final_val_loss = self.training_history.history['val_loss'][-1]
        final_epochs = len(self.training_history.history['loss'])
        
        logger.info(f"Training completed after {final_epochs} epochs")
        logger.info(f"Final loss - Train: {final_train_loss:.6f}, Validation: {final_val_loss:.6f}")
        
        return True

    def calculate_anomaly_thresholds(self):
        """Calculate anomaly thresholds based on reconstruction error"""
        logger.info("Calculating anomaly thresholds...")
        
        # Predictions on training data
        logger.info("Calculating reconstruction errors...")
        train_predictions = self.model.predict(self.x_train, verbose=0)
        train_errors = np.mean(np.square(self.x_train - train_predictions), axis=(1, 2))
        
        # Calculate statistics
        mean_error = np.mean(train_errors)
        std_error = np.std(train_errors)
        min_error = np.min(train_errors)
        max_error = np.max(train_errors)
        
        logger.info(f"Error statistics - Mean: {mean_error:.6f}, Std: {std_error:.6f}")
        logger.info(f"Error range - Min: {min_error:.6f}, Max: {max_error:.6f}")
        
        # Calculate thresholds
        self.anomaly_thresholds = {
            "normal_threshold": float(mean_error + std_error),
            "warning_threshold": float(mean_error + 2 * std_error),
            "critical_threshold": float(mean_error + 3 * std_error),
            "training_stats": {
                "mean_error": float(mean_error),
                "std_error": float(std_error),
                "min_error": float(min_error),
                "max_error": float(max_error),
                "training_samples": len(train_errors)
            }
        }
        
        logger.info("Thresholds calculated:")
        logger.info(f"  NORMAL  < {self.anomaly_thresholds['normal_threshold']:.6f}")
        logger.info(f"  WARNING < {self.anomaly_thresholds['warning_threshold']:.6f}") 
        logger.info(f"  CRITICAL >= {self.anomaly_thresholds['critical_threshold']:.6f}")
        
        return True

    def save_model_and_thresholds(self):
        """Save the model and thresholds"""
        logger.info("Saving the model and thresholds...")
        
        try:
            # Save the model
            model_path = os.path.join(OBC_AI_DIR, "ai_model_lstm_autoencoder.h5")
            self.model.save(model_path)
            logger.info(f"Model saved: {model_path}")
            
            # Save the thresholds
            thresholds_path = os.path.join(OBC_AI_DIR, "ai_thresholds.json")
            thresholds_data = {
                "anomaly_thresholds": self.anomaly_thresholds,
                "training_date": datetime.now().isoformat(),
                "model_architecture": "LSTM Autoencoder",
                "input_shape": self.sequences.shape[1:],
                "dataset_info": self.dataset_config,
                "training_samples": len(self.x_train),
                "validation_samples": len(self.x_val)
            }
            
            with open(thresholds_path, 'w') as f:
                json.dump(thresholds_data, f, indent=2)
            logger.info(f"Thresholds saved: {thresholds_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving files: {e}")
            return False

    def generate_training_report(self):
        """Generate a training report"""
        logger.info("Generating training report...")
        
        try:
            # Performance graphs
            plt.figure(figsize=(15, 5))
            
            # Loss
            plt.subplot(1, 3, 1)
            plt.plot(self.training_history.history['loss'], label='Train Loss', linewidth=2)
            plt.plot(self.training_history.history['val_loss'], label='Val Loss', linewidth=2)
            plt.title('Evolution of the Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # MAE
            plt.subplot(1, 3, 2)
            plt.plot(self.training_history.history['mae'], label='Train MAE', linewidth=2)
            plt.plot(self.training_history.history['val_mae'], label='Val MAE', linewidth=2)
            plt.title('Evolution of the MAE')
            plt.xlabel('Epochs')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Error distribution
            plt.subplot(1, 3, 3)
            train_predictions = self.model.predict(self.x_train, verbose=0)
            train_errors = np.mean(np.square(self.x_train - train_predictions), axis=(1, 2))
            
            plt.hist(train_errors, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(self.anomaly_thresholds['normal_threshold'], color='orange', linestyle='--', label='Normal Threshold')
            plt.axvline(self.anomaly_thresholds['warning_threshold'], color='red', linestyle='--', label='Warning Threshold')
            plt.axvline(self.anomaly_thresholds['critical_threshold'], color='darkred', linestyle='--', label='Critical Threshold')
            plt.title('Error Distribution')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            report_path = os.path.join(OBC_LOGS_DIR, "training_report.png")
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training report generated: {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return False

    def run_training_pipeline(self):
        """Execute the complete training pipeline"""
        logger.info("STARTING OBC AI TRAINING PIPELINE")
        
        try:
            # Sequential pipeline with verification
            steps = [
                ("Loading data", self.load_training_data),
                ("Preparing data", self.prepare_data),
                ("Building model", self.build_lstm_autoencoder),
                ("Training", lambda: self.train_model(epochs=50)),  # Reduced for testing
                ("Calculating thresholds", self.calculate_anomaly_thresholds),
                ("Saving", self.save_model_and_thresholds),
                ("Generating report", self.generate_training_report)
            ]
            
            for step_name, step_func in steps:
                logger.info(f"STEP: {step_name}")
                if not step_func():
                    logger.error(f"Step failed: {step_name}")
                    return False
                logger.info(f"STEP COMPLETED: {step_name}")
            
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            return True
            
        except Exception as e:
            logger.error(f"ERROR DURING TRAINING: {e}")
            return False

def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("OBC AI TRAINING SYSTEM - EPS GUARDIAN")
    logger.info("=" * 60)
    
    trainer = OBCAITrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        logger.info("SUCCESSFUL TRAINING - The OBC model is ready!")
        return 0
    else:
        logger.error("FAILED TRAINING - Check the data and logs")
        return 1

if __name__ == "__main__":
    exit(main())

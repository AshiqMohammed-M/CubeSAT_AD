#!/usr/bin/env python3
"""
COMPLEX AI DATABASE GENERATION - CORRECTED VERSION
With paths adapted to your existing structure
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
import json
import joblib

PROJECT_ROOT = r"D:\final_year_project\Cubesat_AD"
DATA_INPUT = os.path.join(PROJECT_ROOT, "data", "dataset", "pro_eps_dataset.csv")

# Output directories 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "ai_training_base")
VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "analyse", "visualizations")
LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "logs")
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")
ANALYSE_DIR = os.path.join(PROJECT_ROOT, "data", "analyse")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

DATA_OUTPUT = os.path.join(DATASET_DIR, "pro_eps_extended.csv") 
SEQUENCE_DATA = os.path.join(OUTPUT_DIR, "ai_sequence_data.npy")
SEQUENCE_LABELS = os.path.join(OUTPUT_DIR, "ai_sequence_labels.npy")
FEATURES_DATA = os.path.join(OUTPUT_DIR, "ai_sequence_features.npy")
SCALER_FILE = os.path.join(OUTPUT_DIR, "ai_sequence_scaler.pkl")
FEATURES_SCALER_FILE = os.path.join(OUTPUT_DIR, "ai_features_scaler.pkl")
SUMMARY_STATS = os.path.join(ANALYSE_DIR, "extended_summary_stats.csv")  
CONFIG_FILE = os.path.join(ANALYSE_DIR, "dataset_config.json") 

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "generation_complex_data.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ComplexDataGenerator")

logger.info(f"Root directory: {PROJECT_ROOT}")
logger.info(f"Input: {DATA_INPUT}")
logger.info(f"Output AI Training: {OUTPUT_DIR}")
logger.info(f"Output Dataset: {DATASET_DIR}")
logger.info(f"Output Analysis: {ANALYSE_DIR}")

class ComplexDataGenerator:
    def __init__(self, window_size=30, stride=5):
        self.window_size = window_size
        self.stride = stride
        self.scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        
        self.thresholds = {
            "T_batt_critical": 60.0,
            "V_batt_critical_min": 3.0,
            "I_batt_warning": 2.5,
            "V_bus_oscillation": 0.5,
            "converter_ratio_min": 0.7,
            "T_batt_warning": 50.0
        }
        
        logger.info(f"Generator initialized - Window: {window_size}, Stride: {stride}")

    def load_base_data(self):
        """Loads and prepares base data"""
        logger.info(f"Loading data from: {DATA_INPUT}")
        
        if not os.path.exists(DATA_INPUT):
            logger.error(f"File not found: {DATA_INPUT}")
            logger.info("Creating simulated data...")
            return self._create_simulated_data()
        
        try:
            df = pd.read_csv(DATA_INPUT)
            logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Check for required columns
            expected_columns = ["V_batt", "I_batt", "T_batt", "V_bus", "I_bus", "V_solar", "I_solar"]
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                df = self._add_missing_columns(df, missing_columns)
            
            # Timestamp handling
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            else:
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1S')
            
            return df
            
        except Exception as e:
            logger.error(f"Error during loading: {e}")
            logger.info("Creating fallback simulated data...")
            return self._create_simulated_data()

    def _create_simulated_data(self, num_samples=1000):
        """Creates simulated data if file doesn't exist"""
        logger.warning("Creating simulated dataset")
        logger.info(f"Creating {num_samples} simulated samples")
        
        np.random.seed(42)
        
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=num_samples, freq='1S'),
            'V_batt': np.random.uniform(3.2, 8.4, num_samples),
            'I_batt': np.random.uniform(-2.0, 3.5, num_samples),
            'T_batt': np.random.uniform(20, 70, num_samples),
            'V_bus': np.random.uniform(7.0, 8.0, num_samples),
            'I_bus': np.random.uniform(0.5, 2.0, num_samples),
            'V_solar': np.random.uniform(0, 18, num_samples),
            'I_solar': np.random.uniform(0, 2.5, num_samples)
        }
        
        # Add simulated anomalies
        data['T_batt'][100:150] = 65  # Overheat
        data['V_batt'][300:320] = 2.8  # Critical voltage
        data['I_batt'][500:530] = 4.0  # High current
        
        df = pd.DataFrame(data)
        
        simulated_path = os.path.join(DATASET_DIR, "simulated_base_data.csv")
        df.to_csv(simulated_path, index=False)
        logger.info(f"Simulated data saved: {simulated_path}")
        
        return df

    def _add_missing_columns(self, df, missing_columns):
        """Adds missing columns with simulated data"""
        for col in missing_columns:
            if col.startswith('V_'):
                df[col] = np.random.uniform(5, 15, len(df))
            elif col.startswith('I_'):
                df[col] = np.random.uniform(0.5, 3.0, len(df))
            elif col == 'T_batt':
                df[col] = np.random.uniform(20, 45, len(df))
        
        logger.info(f"Columns added: {missing_columns}")
        return df

    def calculate_derived_features(self, window):
        """Calculates derived features for a temporal window"""
        features = {}
        
        # Basic statistics for each column
        for col in ["V_batt", "I_batt", "T_batt", "V_bus", "I_bus", "V_solar", "I_solar"]:
            if col in window.columns:
                features[f"{col}_mean"] = window[col].mean()
                features[f"{col}_std"] = window[col].std()
                features[f"{col}_max"] = window[col].max()
                features[f"{col}_min"] = window[col].min()
                features[f"{col}_trend"] = self._calculate_trend(window[col].values)
        
        # Power features
        if all(col in window.columns for col in ["V_batt", "I_batt"]):
            features["P_batt_mean"] = (window["V_batt"] * window["I_batt"]).mean()
            features["P_batt_var"] = (window["V_batt"] * window["I_batt"]).var()
        
        # Conversion features
        if all(col in window.columns for col in ["V_bus", "V_solar"]):
            ratios = window["V_bus"] / np.maximum(window["V_solar"], 0.1)
            features["converter_ratio_mean"] = ratios.mean()
            features["converter_ratio_std"] = ratios.std()
            features["converter_efficiency"] = ratios.mean() if ratios.mean() > 0 else 0
        
        # Correlations
        if all(col in window.columns for col in ["V_batt", "I_batt"]):
            corr = window["V_batt"].corr(window["I_batt"])
            features["corr_V_I"] = 0 if pd.isna(corr) else corr
        
        # Oscillation score
        features["oscillation_score"] = self._calculate_oscillation_score(window)
        
        return features

    def _calculate_trend(self, values):
        """Calculates the slope of the linear trend"""
        if len(values) < 2:
            return 0
        try:
            x = np.arange(len(values))
            return np.polyfit(x, values, 1)[0]
        except:
            return 0

    def _calculate_oscillation_score(self, window):
        """Calculates an oscillation score based on normalized variance"""
        oscillation_indicators = []
        
        for col in ["V_bus", "V_batt"]:
            if col in window.columns:
                values = window[col].values
                if len(values) > 1:
                    diffs = np.diff(values)
                    direction_changes = np.sum(diffs[1:] * diffs[:-1] < 0)
                    oscillation_indicators.append(direction_changes / len(diffs))
        
        return np.mean(oscillation_indicators) if oscillation_indicators else 0

    def assign_window_label(self, window):
        """Assigns a label to the temporal window"""
        # Conditions CRITICAL
        if (window["T_batt"].max() > self.thresholds["T_batt_critical"] or
            window["V_batt"].min() < self.thresholds["V_batt_critical_min"]):
            return "CRITICAL"
        
        # Conditions WARNING
        elif (window["I_batt"].max() > self.thresholds["I_batt_warning"] or
              window["V_bus"].std() > self.thresholds["V_bus_oscillation"] or
              window["T_batt"].max() > self.thresholds["T_batt_warning"] or
              self._calculate_oscillation_score(window) > 0.3):
            return "WARNING"
        
        # NORMAL
        else:
            return "NORMAL"

    def generate_sequences(self, df):
        """Generates temporal sequences and labels with consistent normalization"""
        logger.info("Generating temporal sequences...")
        
        sequences = []
        sequence_features = []
        labels = []
        timestamps_start = []
        timestamps_end = []
        
        total_windows = (len(df) - self.window_size) // self.stride + 1
        
        for i in range(0, len(df) - self.window_size + 1, self.stride):
            window = df.iloc[i:i + self.window_size].copy()
            
            # Extract sequence data
            sequence_cols = ["V_batt", "I_batt", "T_batt", "V_bus", "I_bus", "V_solar", "I_solar"]
            sequence_data = window[sequence_cols].values
            sequences.append(sequence_data)
            
            # Calculate derived features
            features = self.calculate_derived_features(window)
            sequence_features.append(list(features.values()))
            
            # Assign label
            label = self.assign_window_label(window)
            labels.append(label)
            
            timestamps_start.append(window.iloc[0]['timestamp'])
            timestamps_end.append(window.iloc[-1]['timestamp'])
            
            if len(sequences) % 100 == 0 and len(sequences) > 0:
                logger.info(f"Progress: {len(sequences)}/{total_windows} windows")
        
        logger.info(f"Raw generation completed: {len(sequences)} sequences created")
        
        # DATA CLEANING
        logger.info("Cleaning data (NaN/Inf)...")
        
        # Clean main sequences
        all_sequences = np.array(sequences)
        nan_count_before = np.sum(np.isnan(all_sequences)) + np.sum(np.isinf(all_sequences))
        
        if nan_count_before > 0:
            logger.warning(f"{nan_count_before} NaN/Inf values detected in sequences")
        
        all_sequences = np.nan_to_num(all_sequences, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clean derived features
        sequence_features_array = np.array(sequence_features)
        nan_count_features = np.sum(np.isnan(sequence_features_array)) + np.sum(np.isinf(sequence_features_array))
        
        if nan_count_features > 0:
            logger.warning(f"{nan_count_features} NaN/Inf values detected in derived features")
        
        sequence_features_array = np.nan_to_num(sequence_features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info("NaN/Inf cleaning completed")
        
        # CONSISTENT SEQUENCE NORMALIZATION
        logger.info("Applying consistent normalization...")
        
        flat_sequences = all_sequences.reshape(-1, all_sequences.shape[-1])
        scaled_flat = self.scaler.fit_transform(flat_sequences)
        scaled_sequences = scaled_flat.reshape(all_sequences.shape)
        
        logger.info(f"Sequence normalization applied - Mean: {self.scaler.mean_.round(3)}, Scale: {self.scaler.scale_.round(3)}")
        
        # DERIVED FEATURES NORMALIZATION
        scaled_features = self.feature_scaler.fit_transform(sequence_features_array)
        logger.info(f"Derived features normalized - {scaled_features.shape[1]} features")
        
        # Feature names for traceability
        feature_names = list(features.keys()) if features else []
        
        return {
            'sequences': scaled_sequences,
            'features': scaled_features,
            'labels': np.array(labels),
            'timestamps_start': timestamps_start,
            'timestamps_end': timestamps_end,
            'feature_names': feature_names
        }

    def save_datasets(self, data):
        """Saves all generated datasets with improved traceability"""
        logger.info("Saving datasets...")
        
        # Save numpy arrays
        np.save(SEQUENCE_DATA, data['sequences'])
        logger.info(f"Sequences saved: {data['sequences'].shape}")
        
        np.save(SEQUENCE_LABELS, data['labels'])
        logger.info(f"Labels saved: {len(data['labels'])}")
        
        np.save(FEATURES_DATA, data['features'])
        logger.info(f"Derived features saved: {data['features'].shape}")
        
        # Save scalers
        joblib.dump(self.scaler, SCALER_FILE)
        joblib.dump(self.feature_scaler, FEATURES_SCALER_FILE)
        logger.info("Scalers saved")
        
        # Save extended CSV
        extended_df = pd.DataFrame({
            'timestamp_start': data['timestamps_start'],
            'timestamp_end': data['timestamps_end'],
            'label': data['labels'],
            'sequence_id': range(len(data['labels']))
        })
        extended_df.to_csv(DATA_OUTPUT, index=False)
        logger.info(f"Extended CSV saved: {len(extended_df)} rows")
        
        # Configuration with traceability
        config = {
            'window_size': self.window_size,
            'stride': self.stride,
            'thresholds': self.thresholds,
            'dataset_shape': {
                'sequences': data['sequences'].shape,
                'features': data['features'].shape,
                'labels': len(data['labels'])
            },
            'feature_names': data['feature_names'],
            'generation_date': datetime.now().isoformat(),
            'base_dataset': os.path.basename(DATA_INPUT),
            'derived_from': "EPS MCU Simple Dataset",
            'dataset_type': "Complex Temporal Sequences for LSTM/Autoencoder",
            'normalization_applied': True,
            'scaler_parameters': {
                'sequence_scaler_mean': self.scaler.mean_.tolist(),
                'sequence_scaler_scale': self.scaler.scale_.tolist(),
                'feature_scaler_mean': self.feature_scaler.mean_.tolist(),
                'feature_scaler_scale': self.feature_scaler.scale_.tolist()
            },
            'input_features': ["V_batt", "I_batt", "T_batt", "V_bus", "I_bus", "V_solar", "I_solar"],
            'output_labels': ["NORMAL", "WARNING", "CRITICAL"],
            'purpose': "Training advanced AI models (LSTM, Autoencoder, GRU) for temporal anomaly detection"
        }
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info("Configuration with traceability saved")

    def generate_statistics(self, data):
        """Generates statistics and visualizations"""
        logger.info("Generating statistics...")
        
        label_counts = pd.Series(data['labels']).value_counts()
        total_sequences = len(data['labels'])
        
        stats = {
            'total_sequences': total_sequences,
            'window_size': self.window_size,
            'stride': self.stride,
            'label_distribution': {
                'NORMAL': label_counts.get('NORMAL', 0),
                'WARNING': label_counts.get('WARNING', 0), 
                'CRITICAL': label_counts.get('CRITICAL', 0)
            },
            'label_percentages': {
                'NORMAL': (label_counts.get('NORMAL', 0) / total_sequences) * 100,
                'WARNING': (label_counts.get('WARNING', 0) / total_sequences) * 100,
                'CRITICAL': (label_counts.get('CRITICAL', 0) / total_sequences) * 100
            },
            'normalization_info': {
                'applied': True,
                'sequences_shape': data['sequences'].shape,
                'features_shape': data['features'].shape
            }
        }
        
        # Save statistics
        stats_df = pd.DataFrame([
            {'Label': 'NORMAL', 'Count': stats['label_distribution']['NORMAL'], 'Percentage': stats['label_percentages']['NORMAL']},
            {'Label': 'WARNING', 'Count': stats['label_distribution']['WARNING'], 'Percentage': stats['label_percentages']['WARNING']},
            {'Label': 'CRITICAL', 'Count': stats['label_distribution']['CRITICAL'], 'Percentage': stats['label_percentages']['CRITICAL']},
            {'Label': 'TOTAL', 'Count': total_sequences, 'Percentage': 100.0}
        ])
        stats_df.to_csv(SUMMARY_STATS, index=False)
        
        self._generate_visualizations(data, stats)
        
        logger.info("=== FINAL STATISTICS ===")
        logger.info(f"Total sequences: {stats['total_sequences']}")
        for label in ['NORMAL', 'WARNING', 'CRITICAL']:
            logger.info(f"  {label}: {stats['label_distribution'][label]} ({stats['label_percentages'][label]:.1f}%)")
        logger.info(f"Sequence shape: {stats['normalization_info']['sequences_shape']}")
        logger.info(f"Feature shape: {stats['normalization_info']['features_shape']}")
        
        return stats

    def _generate_visualizations(self, data, stats):
        """Generates visualizations"""
        logger.info("Generating visualizations...")
        
        plt.figure(figsize=(15, 10))
        
        # Chart 1: Label distribution
        plt.subplot(2, 3, 1)
        labels, counts = zip(*stats['label_distribution'].items())
        colors = ['green', 'orange', 'red']
        plt.bar(labels, counts, color=colors)
        plt.title('Label Distribution')
        plt.ylabel('Number of sequences')
        
        # Chart 2: Correlation matrix
        plt.subplot(2, 3, 2)
        if len(data['sequences']) > 0:
            first_sequence = data['sequences'][0]
            corr_matrix = np.corrcoef(first_sequence.T)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                       xticklabels=["V_batt", "I_batt", "T_batt", "V_bus", "I_bus", "V_solar", "I_solar"],
                       yticklabels=["V_batt", "I_batt", "T_batt", "V_bus", "I_bus", "V_solar", "I_solar"])
            plt.title('Correlation - Normalized Sequence')
        
        # Chart 3: Example temporal sequence
        plt.subplot(2, 3, 3)
        if len(data['sequences']) > 0:
            example_sequence = data['sequences'][0]
            plt.plot(example_sequence[:, 0], label='V_batt', alpha=0.7)
            plt.plot(example_sequence[:, 1], label='I_batt', alpha=0.7)
            plt.plot(example_sequence[:, 2], label='T_batt', alpha=0.7)
            plt.legend()
            plt.title('Normalized Temporal Sequence')
            plt.xlabel('Time (samples)')
            plt.ylabel('Normalized values')
        
        # Chart 4: Pie chart
        plt.subplot(2, 3, 4)
        plt.pie(stats['label_distribution'].values(), labels=stats['label_distribution'].keys(), 
                colors=colors, autopct='%1.1f%%')
        plt.title('State Distribution')
        
        # Chart 5: Normalized values distribution
        plt.subplot(2, 3, 5)
        if len(data['sequences']) > 0:
            all_values = data['sequences'].flatten()
            plt.hist(all_values, bins=50, alpha=0.7, edgecolor='black')
            plt.title('Normalized Values Distribution')
            plt.xlabel('Normalized values')
            plt.ylabel('Frequency')
        
        # Chart 6: Top derived features
        plt.subplot(2, 3, 6)
        if data['features'].size > 0 and data['feature_names']:
            feature_means = np.mean(data['features'], axis=0)
            top_features_idx = np.argsort(np.abs(feature_means))[-10:]
            top_feature_names = [data['feature_names'][i] for i in top_features_idx]
            plt.barh(range(len(top_features_idx)), feature_means[top_features_idx])
            plt.yticks(range(len(top_features_idx)), top_feature_names, fontsize=8)
            plt.title('Top 10 Features (mean)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, "dataset_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved in: {VISUALIZATIONS_DIR}")

def main():
    logger.info("STARTING COMPLEX AI DATABASE GENERATION")
    
    try:
        generator = ComplexDataGenerator(window_size=30, stride=5)
        df = generator.load_base_data()
        data = generator.generate_sequences(df)
        generator.save_datasets(data)
        stats = generator.generate_statistics(data)
        
        logger.info("GENERATION COMPLETED SUCCESSFULLY")
        logger.info(f"RESULTS: {stats['total_sequences']} sequences generated")
        logger.info(f"AI Training Directory: {OUTPUT_DIR}")
        logger.info(f"Dataset Directory: {DATASET_DIR}")
        logger.info(f"Analysis Directory: {ANALYSE_DIR}")
        logger.info("READY FOR LSTM/AUTOENCODER MODEL TRAINING!")
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR: {e}")
        raise

if __name__ == "__main__":
    main()

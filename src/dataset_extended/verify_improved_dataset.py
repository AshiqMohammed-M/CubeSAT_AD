#!/usr/bin/env python3
"""
Verification of improved database with normalization
Corrected version with proper paths for your structure
"""

import os
import numpy as np
import pandas as pd
import json
import joblib

# Corrected paths for your structure
PROJECT_ROOT = r"D:\final_year_project\Cubesat_AD"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "ai_training_base")
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")
ANALYSE_DIR = os.path.join(PROJECT_ROOT, "data", "analyse")

def verify_improved_datasets():
    """Verifies integrity of generated datasets with improvements"""
    print("VERIFICATION OF IMPROVED DATASETS")
    print("=" * 60)
    
    # File definitions with their CORRECT locations
    files_to_check = {
        os.path.join(DATASET_DIR, 'pro_eps_extended.csv'): 'Extended CSV (dataset/)',
        os.path.join(OUTPUT_DIR, 'ai_sequence_data.npy'): 'Normalized temporal sequences (ai_training_base/)', 
        os.path.join(OUTPUT_DIR, 'ai_sequence_labels.npy'): 'Sequence labels (ai_training_base/)',
        os.path.join(OUTPUT_DIR, 'ai_sequence_features.npy'): 'Normalized derived features (ai_training_base/)',
        os.path.join(OUTPUT_DIR, 'ai_sequence_scaler.pkl'): 'Sequence scaler (ai_training_base/)',
        os.path.join(OUTPUT_DIR, 'ai_features_scaler.pkl'): 'Feature scaler (ai_training_base/)',
        os.path.join(ANALYSE_DIR, 'extended_summary_stats.csv'): 'Statistics (analyse/)',
        os.path.join(ANALYSE_DIR, 'dataset_config.json'): 'Configuration with traceability (analyse/)'
    }
    
    for filepath, description in files_to_check.items():
        if os.path.exists(filepath):
            print(f"{description}: {os.path.basename(filepath)}")
            
            # Additional information based on file type
            try:
                if filepath.endswith('.npy'):
                    data = np.load(filepath)
                    print(f"   Shape: {data.shape} | dtype: {data.dtype}")
                elif filepath.endswith('.csv'):
                    df = pd.read_csv(filepath)
                    print(f"   Rows: {len(df)} | Columns: {len(df.columns)}")
                elif filepath.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    print(f"   Normalization: {config.get('normalization_applied', 'N/A')}")
                    print(f"   Sequences: {config.get('dataset_shape', {}).get('sequences', 'N/A')}")
                elif filepath.endswith('.pkl'):
                    scaler = joblib.load(filepath)
                    print(f"   Type: {type(scaler).__name__} | Features: {len(scaler.mean_)}")
                    
            except Exception as e:
                print(f"   Read error: {e}")
        else:
            print(f"{description}: FILE MISSING")
    
    print("=" * 60)
    
    # Consistency and normalization verification
    try:
        sequences_path = os.path.join(OUTPUT_DIR, 'ai_sequence_data.npy')
        labels_path = os.path.join(OUTPUT_DIR, 'ai_sequence_labels.npy')
        features_path = os.path.join(OUTPUT_DIR, 'ai_sequence_features.npy')
        
        if all(os.path.exists(p) for p in [sequences_path, labels_path, features_path]):
            sequences = np.load(sequences_path)
            labels = np.load(labels_path)
            features = np.load(features_path)
            
            print("CONSISTENCY VERIFICATION:")
            print(f"  Sequences: {sequences.shape[0]}")
            print(f"  Labels: {labels.shape[0]}")
            print(f"  Features: {features.shape[0]}")
            
            if sequences.shape[0] == labels.shape[0] == features.shape[0]:
                print(" Dimensions consistent")
            else:
                print(" Dimensions inconsistent!")
            
            # Normalization verification
            sequence_mean = np.mean(sequences)
            sequence_std = np.std(sequences)
            print(f"\nNORMALIZATION VERIFICATION:")
            print(f"  Mean: {sequence_mean:.6f} (expected ~0.0)")
            print(f"  Std:  {sequence_std:.6f} (expected ~1.0)")
            
            if abs(sequence_mean) < 0.1 and abs(sequence_std - 1.0) < 0.2:
                print("  Normalization correct")
            else:
                print("  Normalization potentially problematic")
                
        else:
            print("Missing files for consistency verification")
            
    except Exception as e:
        print(f"Verification error: {e}")

if __name__ == "__main__":
    verify_improved_datasets()

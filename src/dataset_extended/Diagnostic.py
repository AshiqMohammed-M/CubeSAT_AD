#!/usr/bin/env python3
"""
NORMALIZATION DIAGNOSTIC - Problem investigation
"""

import os
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = r"D:\final_year_project\Cubesat_AD"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "ai_training_base")

def diagnose_normalization_issue():
    """Investigates the overly perfect normalization problem"""
    print(" NORMALIZATION PROBLEM DIAGNOSTIC")
    print("=" * 60)
    
    try:
        # Load data
        sequences = np.load(os.path.join(OUTPUT_DIR, 'ai_sequence_data.npy'))
        features = np.load(os.path.join(OUTPUT_DIR, 'ai_sequence_features.npy'))
        scaler_seq = joblib.load(os.path.join(OUTPUT_DIR, 'ai_sequence_scaler.pkl'))
        
        print("1. ORIGINAL SEQUENCE ANALYSIS (before normalization):")
        print("=" * 50)
        
        # Check if we have original data
        original_data_path = os.path.join(PROJECT_ROOT, "data", "dataset", "pro_eps_dataset.csv")
        if os.path.exists(original_data_path):
            df_original = pd.read_csv(original_data_path)
            print(f"Original data - Shape: {df_original.shape}")
            print("\nOriginal data statistics:")
            for col in ['V_batt', 'I_batt', 'T_batt', 'V_bus', 'I_bus', 'V_solar', 'I_solar']:
                if col in df_original.columns:
                    print(f"  {col}: mean={df_original[col].mean():.3f}, std={df_original[col].std():.3f}")
        
        print("\n2. NORMALIZED SEQUENCE ANALYSIS:")
        print("=" * 50)
        
        # Analyze normalized sequences
        print(f"Sequence shape: {sequences.shape}")
        
        # Check by channel
        print("\nBy channel (average over all sequences):")
        channels = ['V_batt', 'I_batt', 'T_batt', 'V_bus', 'I_bus', 'V_solar', 'I_solar']
        for i, channel in enumerate(channels):
            channel_data = sequences[:, :, i]
            print(f"  {channel}: mean={np.mean(channel_data):.6f}, std={np.std(channel_data):.6f}")
        
        print("\n3. EXTREME VALUES VERIFICATION:")
        print("=" * 50)
        
        # Check min/max
        print(f"Global min: {np.min(sequences):.6f}")
        print(f"Global max: {np.max(sequences):.6f}")
        
        # Check unique values
        unique_vals = np.unique(sequences)
        print(f"Number of unique values: {len(unique_vals)}")
        if len(unique_vals) < 20:
            print(f"Unique values: {unique_vals}")
        
        print("\n4. REALISM TEST:")
        print("=" * 50)
        
        # Take a random sample
        sample_seq = sequences[np.random.randint(0, len(sequences))]
        print("Sample sequence (first 10 values of first channel):")
        print(sample_seq[:10, 0])
        
        # Check temporal variance
        temporal_variance = np.var(sequences, axis=1)  # Variance along time
        avg_temporal_var = np.mean(temporal_variance)
        print(f"\nAverage temporal variance: {avg_temporal_var:.6f}")
        
        if avg_temporal_var < 0.01:
            print("  WARNING: Very low temporal variance!")
            print("   Sequences might be too constant.")
        
        print("\n5. SCALER INFORMATION:")
        print("=" * 50)
        print(f"Scaler mean: {scaler_seq.mean_}")
        print(f"Scaler scale: {scaler_seq.scale_}")
        
        # Check if scaler has very small scales (division problem)
        if np.any(scaler_seq.scale_ < 1e-6):
            print(" PROBLEM: Some scales are almost zero!")
            problematic_indices = np.where(scaler_seq.scale_ < 1e-6)[0]
            for idx in problematic_indices:
                print(f"  Channel {channels[idx]}: scale={scaler_seq.scale_[idx]}")
        
    except Exception as e:
        print(f"Diagnostic error: {e}")

def check_data_generation_process():
    """Verifies the data generation process"""
    print("\n DATA GENERATION PROCESS VERIFICATION")
    print("=" * 60)
    
    # Check if we're using simulated data
    simulated_path = os.path.join(PROJECT_ROOT, "data", "dataset", "simulated_base_data.csv")
    if os.path.exists(simulated_path):
        print("  SIMULATED DATA USAGE DETECTED")
        df_sim = pd.read_csv(simulated_path)
        print(f"File: {simulated_path}")
        print(f"Shape: {df_sim.shape}")
        
        # Analyze simulated data quality
        print("\nSimulated data quality:")
        for col in ['V_batt', 'I_batt', 'T_batt']:
            if col in df_sim.columns:
                unique_ratio = df_sim[col].nunique() / len(df_sim)
                print(f"  {col}: {df_sim[col].nunique()} unique values ({unique_ratio:.1%})")

if __name__ == "__main__":
    diagnose_normalization_issue()
    check_data_generation_process()
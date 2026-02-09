#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import sys

class AIPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.features = None
        self.base_dir = r"D:\final_year_project\Cubesat_AD"
        self.data_dir = os.path.join(self.base_dir, "data")
        self.dataset_dir = os.path.join(self.data_dir, "dataset")
        self.training_data_dir = os.path.join(self.data_dir, "training_data")
        self.analyse_dir = os.path.join(self.data_dir, "analyse")
        
    def load_dataset(self):
        data_path = os.path.join(self.dataset_dir, "pro_eps_dataset.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        print(f"Loading dataset: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Dataset loaded: {len(df)} samples, {len(df.columns)} columns")
        return df
    
    def select_features(self, df):
        base_features = ["V_batt", "I_batt", "T_batt", "V_bus", "I_bus", "V_solar", "I_solar", "SOC", "T_eps"]
        available_features = [f for f in base_features if f in df.columns]
        print(f"Selected features ({len(available_features)}):")
        for feature in available_features: 
            print(f"  - {feature}")
        return available_features
    
    def prepare_training_data(self, df, features):
        if "anomaly_type" in df.columns:
            normal_data = df[df["anomaly_type"] == "normal"].copy()
            print(f"Normal data for training: {len(normal_data)} samples")
        else:
            normal_data = df.copy()
            print(f"Column 'anomaly_type' not found, using entire dataset")
        
        normal_data_processed = normal_data.copy()
        
        # Base features
        final_features = [f for f in features if f in normal_data_processed.columns]
        
        # Existing calculated features
        calculated_features = ["P_batt", "P_solar", "P_bus", "converter_ratio", 
                              "delta_V_batt", "delta_I_batt", "delta_T_batt",
                              "rolling_std_V_batt", "rolling_mean_V_batt"]
        
        for feature in calculated_features:
            if feature in normal_data_processed.columns and feature not in final_features:
                final_features.append(feature)
        
        print(f"Final features ({len(final_features)}):")
        for feature in final_features: 
            print(f"  - {feature}")
        
        X = normal_data_processed[final_features]
    
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.interpolate(method='linear', limit_direction='forward')
                
        X_clean = X_clean.bfill().ffill()
        
        if X_clean.isna().any().any():
            X_clean = X_clean.fillna(0)
            
        print(f"Cleaned data: {X_clean.shape}")
        return X_clean, final_features
    
    def normalize_data(self, X):
        print("\nNormalizing data...")
        X_scaled = self.scaler.fit_transform(X)
        print(f"Normalized data range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
        return X_scaled
    
    def save_processed_data(self, X_scaled, feature_names):
        """Saves processed data in training_data/"""
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        # Save training data
        train_data_path = os.path.join(self.training_data_dir, "ai_train_data.npy")
        np.save(train_data_path, X_scaled)
        print(f" Training data saved: {train_data_path}")
        
        # Save feature names
        feature_path = os.path.join(self.training_data_dir, "ai_feature_names.npy")
        np.save(feature_path, np.array(feature_names))
        print(f" Feature names saved: {feature_path}")
        
        # Save scaler
        scaler_path = os.path.join(self.training_data_dir, "ai_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f" Scaler saved: {scaler_path}")
        
        # Save CSV summary in analyse/ for verification
        summary_path = os.path.join(self.analyse_dir, "ai_training_summary.csv")
        summary_df = pd.DataFrame(X_scaled, columns=feature_names)
        summary_df.to_csv(summary_path, index=False)
        print(f" CSV summary saved: {summary_path}")
        
        return train_data_path
    
    def run(self):
        print(" Starting AI preprocessing")
        print("=" * 60)
        try:
            # Display path structure
            print(f"Base directory: {self.base_dir}")
            print(f"Dataset source: {self.dataset_dir}")
            print(f"Training data output: {self.training_data_dir}")
            print(f"Analysis reports: {self.analyse_dir}")
            print("-" * 40)
            
            df = self.load_dataset()
            features = self.select_features(df)
            X, final_features = self.prepare_training_data(df, features)
            X_scaled = self.normalize_data(X)
            output_path = self.save_processed_data(X_scaled, final_features)
            
            print("\n AI preprocessing completed successfully!")
            print(f" Final data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
            print(f" TensorFlow files in: {self.training_data_dir}")
            print(f" Report in: {self.analyse_dir}")
            
            return True
            
        except Exception as e:
            print(f" Preprocessing error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    preprocessor = AIPreprocessor()
    success = preprocessor.run()
    if success:
        print("\n AI preprocessing is complete!")
        print(" You can now proceed to TensorFlow model training.")
        print(f" Training data: D:\\Challenge AESS&IES\\data\\training_data\\")
        print(f" Report: D:\\Challenge AESS&IES\\data\\analyse\\ai_training_summary.csv")
    else:
        print("\n AI preprocessing failed.")

if __name__ == "__main__":
    main()

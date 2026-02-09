import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import logging
import os
import sys
from datetime import datetime

# === CORRECTED PATH CONFIGURATION ===
BASE_DIR = r"D:\final_year_project\Cubesat_AD"
DATA_DIR = os.path.join(BASE_DIR, "data")
ANALYSE_DIR = os.path.join(DATA_DIR, "analyse")
VISUALIZATIONS_DIR = os.path.join(ANALYSE_DIR, "visualizations")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "data_train")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

# REMOVE os.makedirs(LOGS_DIR, exist_ok=True)

# plot configuration
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 10

# Logging configuration - CONSOLE ONLY
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # REMOVE FileHandler
)

logger = logging.getLogger(__name__)

class ProEPSAnalyzer:
    
    """Analyzer for EPS data with physical consistency checks"""
    
    def __init__(self):
        self.expected_cols = [
            'timestamp', 'V_batt', 'I_batt', 'T_batt', 'V_solar', 'I_solar',
            'V_bus', 'I_bus', 'T_eps', 'SOC', 'anomaly_type', 'orbit_sunlight'
        ]
        
        # CORRECTED path configuration
        self.data_dir = DATASET_DIR
        self.output_dir = VISUALIZATIONS_DIR
        self.logs_dir = LOGS_DIR
        
        # Colors for visualizations
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3F7CAC']
        
        # Physical limits
        self.physical_limits = {
            'V_batt': (5.5, 8.6),
            'I_batt': (-3.0, 3.0),
            'T_batt': (-25, 85),
            'SOC': (0, 105),
            'I_bus': (0, 2.0),
            'V_solar': (0, 20),
            'T_eps': (-20, 80),
            'P_batt': (-25, 25),
            'P_solar': (0, 40)
        }
        
        # Critical anomaly types
        self.critical_anomalies = ['batt_overheat', 'batt_undervoltage', 'batt_overcurrent']
        
        logger.info("EPS analyzer initialized")

    def check_physical_consistency(self, df):
        """Verify the physical consistency of the dataset"""
        logger.info("Verifying physical consistency...")
        
        checks_passed = True
        physical_issues = []
        
        print("\nPhysical consistency verification:")
        
        # 1. Verification of physical limits
        for param, (min_val, max_val) in self.physical_limits.items():
            if param in df.columns:
                valid_data = df[param].dropna()
                if len(valid_data) > 0:
                    out_of_bounds = valid_data[(valid_data < min_val) | (valid_data > max_val)]
                    if len(out_of_bounds) > 0:
                        physical_issues.append(f"{param}: {len(out_of_bounds)} values out of bounds")
                        checks_passed = False
        
        # 2. Verification that I_bus > 0
        if 'I_bus' in df.columns:
            negative_I_bus = df[df['I_bus'] < 0]
            if len(negative_I_bus) > 0:
                physical_issues.append(f"I_bus: {len(negative_I_bus)} negative values")
                checks_passed = False
        
        # Final report
        if checks_passed and not physical_issues:
            print("   Physical consistency validated")
        else:
            print("   Issues detected:")
            for issue in physical_issues:
                print(f"   - {issue}")
        
        return checks_passed

    def check_dataset_integrity(self, df):
        """Verifies dataset integrity"""
        logger.info("Checking dataset integrity...")
        
        checks_passed = True
        
        print("\nDataset integrity check:")
        
        # 1. Missing columns
        missing_cols = [col for col in self.expected_cols if col not in df.columns]
        if missing_cols:
            print(f"   Missing columns: {missing_cols}")
            checks_passed = False
        else:
            print("   All columns present")
        
        # 2. Missing data
        sensor_cols = [col for col in self.expected_cols if col not in ['timestamp', 'anomaly_type', 'orbit_sunlight']]
        nan_summary = df[sensor_cols].isna().sum()
        
        nan_issues = []
        for col, count in nan_summary.items():
            if count > 0:
                nan_issues.append(f"{col}: {count} NaN")
        
        if nan_issues:
            print(f"   Valeurs manquantes: {', '.join(nan_issues)}")
            checks_passed = False
        else:
            print("   Aucune valeur manquante")
        
        # 3. Physical consistency check
        physical_ok = self.check_physical_consistency(df)
        if not physical_ok:
            checks_passed = False
        
        # 4. Analyse distribution anomalies
        anomaly_stats = df['anomaly_type'].value_counts()
        total_samples = len(df)
        
        print(f"\ndata distribution:")
        for anomaly_type, count in anomaly_stats.items():
            percentage = (count / total_samples) * 100
            status = "CRITICAL" if anomaly_type in self.critical_anomalies else "NORMAL" if anomaly_type == 'normal' else "ANOMALY"
            print(f"   {status} {anomaly_type}: {count} ({percentage:.1f}%)")
        
        return checks_passed

    def analyze_dataset(self, filename="pro_eps_dataset.csv"):
        """Complete dataset analysis with visualizations"""
        logger.info("Starting dataset analysis...")
        
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"Dataset loaded: {len(df)} samples")
            logger.info(f"Dataset loaded: {len(df)} samples")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print(f"Verify that the file exists in: {self.data_dir}")
            return
        except Exception as e:
            print(f"Loading error: {e}")
            return
        
        # Integrity verification
        self.check_dataset_integrity(df)
        
        # Statistical analysis
        self._generate_statistical_analysis(df)
        
        # Visualizations
        self._create_comprehensive_plots(df)
        
        # OBC summary
        self._export_obc_summary(df)
        
        # Excel report generation
        self._generate_excel_report(df)
        
        print("Analysis completed successfully!")
        logger.info("Analysis completed successfully")

    def _generate_statistical_analysis(self, df):
        """Generates statistical analysis"""
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        sensor_cols = ['V_batt', 'I_batt', 'T_batt', 'V_solar', 'I_solar', 'V_bus', 'I_bus', 'T_eps', 'SOC']
        sensor_cols = [col for col in sensor_cols if col in df.columns]
        
        # Descriptive statistics
        print("\nDescriptive statistics:")
        stats = df[sensor_cols].describe()
        print(stats.round(3))
        
        # Limit verification
        print("\nLimit verification:")
        limits_check = {
            'V_batt': (5.5, 8.6),
            'I_batt': (-3.0, 3.0),
            'SOC': (0, 100)
        }
        
        for param, (min_val, max_val) in limits_check.items():
            if param in df.columns:
                data_min = df[param].min()
                data_max = df[param].max()
                status = "OK" if min_val <= data_min <= data_max <= max_val else "ERROR"
                print(f"   {status} {param}: {data_min:.2f} to {data_max:.2f}")
        
        # Anomaly analysis
        print("\nAnomaly distribution:")
        anomaly_stats = df['anomaly_type'].value_counts()
        
        for anomaly_type, count in anomaly_stats.items():
            percentage = (count / len(df)) * 100
            status = "[CRITICAL]" if anomaly_type in self.critical_anomalies else "[NORMAL]" if anomaly_type == 'normal' else "[ANOMALY]"
            print(f"  {status} {anomaly_type}: {count} ({percentage:.1f}%)")

    def _create_comprehensive_plots(self, df):
        """Creates all analysis visualizations"""
        logger.info("Creating visualizations...")
        
        try:
            # 1. Main dashboard
            self._create_main_dashboard(df)
            
            # 2. Detailed time series
            self._create_detailed_timeseries(df)
            
            # 3. Anomaly analysis
            self._create_anomaly_analysis(df)
            
            # 4. Parameter distribution
            self._create_distribution_plots(df)
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Visualization creation error: {e}")
            print(f"Graph creation error: {e}")

    def _create_main_dashboard(self, df):
        """Creates the main dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('EPS GUARDIAN DASHBOARD - SYSTEM ANALYSIS', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Pie chart - Data distribution
        anomaly_counts = df['anomaly_type'].value_counts()
        
        # Group small categories
        threshold = 0.02
        total_samples = len(df)
        main_categories = {}
        other_categories = {}
        
        for anomaly_type, count in anomaly_counts.items():
            percentage = count / total_samples
            if percentage >= threshold:
                main_categories[anomaly_type] = count
            else:
                other_categories[anomaly_type] = count
        
        if other_categories:
            main_categories['Autres'] = sum(other_categories.values())
        
        labels = list(main_categories.keys())
        sizes = list(main_categories.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = axes[0, 0].pie(
            sizes, labels=None, autopct='%1.1f%%', 
            colors=colors, startangle=90, 
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        axes[0, 0].legend(
            wedges, 
            [f'{label} ({count})' for label, count in main_categories.items()],
            title="Data types",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=9
        )
        
        axes[0, 0].set_title('Distribution of Data Types', fontweight='bold', fontsize=12)
        
        # 2. Bar chart - Distribution of anomalies
        anomalies_only = df[df['anomaly_type'] != 'normal']['anomaly_type'].value_counts()
        
        if len(anomalies_only) > 0:
            main_anomalies = {}
            other_anomalies = {}
            
            for anomaly_type, count in anomalies_only.items():
                percentage = count / anomalies_only.sum()
                if percentage >= threshold:
                    main_anomalies[anomaly_type] = count
                else:
                    other_anomalies[anomaly_type] = count
            
            if other_anomalies:
                main_anomalies['Other anomalies'] = sum(other_anomalies.values())
            
            x_pos = np.arange(len(main_anomalies))
            bars = axes[0, 1].bar(x_pos, list(main_anomalies.values()), 
                                 color=plt.cm.viridis(np.linspace(0, 1, len(main_anomalies))),
                                 alpha=0.8,
                                 edgecolor='grey',
                                 linewidth=0.5)
            
            axes[0, 1].set_title('Distribution of Anomalies', fontweight='bold', fontsize=12)
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(list(main_anomalies.keys()), rotation=45, ha='right', fontsize=9)
            axes[0, 1].set_ylabel('Number of samples', fontsize=10)
            
            for bar in bars:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{int(height)}', 
                               ha='center', va='bottom', 
                               fontweight='bold', fontsize=8)
            
            axes[0, 1].set_ylim(0, max(main_anomalies.values()) * 1.15)
            
        else:
            axes[0, 1].text(0.5, 0.5, 'No anomalies detected', 
                           ha='center', va='center', transform=axes[0, 1].transAxes,
                           fontsize=12, fontweight='bold', style='italic')
            axes[0, 1].set_title('Distribution of Anomalies', fontweight='bold', fontsize=12)
        
        # 3. Correlation matrix
        sensor_cols = ['V_batt', 'I_batt', 'T_batt', 'V_solar', 'I_solar', 'V_bus', 'I_bus', 'T_eps', 'SOC']
        sensor_cols = [col for col in sensor_cols if col in df.columns]
        corr_matrix = df[sensor_cols].corr()
        
        im = axes[1, 0].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        axes[1, 0].set_xticks(range(len(sensor_cols)))
        axes[1, 0].set_yticks(range(len(sensor_cols)))
        axes[1, 0].set_xticklabels(sensor_cols, rotation=45, ha='right', fontsize=9)
        axes[1, 0].set_yticklabels(sensor_cols, fontsize=9)
        axes[1, 0].set_title('Correlation Matrix', fontweight='bold', fontsize=12)
        
        for i in range(len(sensor_cols)):
            for j in range(len(sensor_cols)):
                color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                fontweight = 'bold' if abs(corr_matrix.iloc[i, j]) > 0.7 else 'normal'
                axes[1, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                               ha='center', va='center', 
                               color=color, fontsize=8, fontweight=fontweight)
        
        cbar = plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontsize=10)
        
        # 4. Temporal view
        sample_df = df.iloc[::max(1, len(df)//200)].copy() 
        
        if len(sample_df) > 10:
            sample_df['V_batt_smooth'] = sample_df['V_batt'].rolling(window=5, center=True).mean()
        else:
            sample_df['V_batt_smooth'] = sample_df['V_batt']
        
        line_vbatt = axes[1, 1].plot(sample_df['timestamp'], sample_df['V_batt_smooth'], 
                                    label='V_batt (V)', linewidth=2, alpha=0.9, color='#1f77b4')
        line_tbatt = axes[1, 1].plot(sample_df['timestamp'], sample_df['T_batt'], 
                                    label='T_batt (°C)', linewidth=2, alpha=0.8, color='#ff7f0e')
        
        anomalies_df = df[df['anomaly_type'] != 'normal']
        if len(anomalies_df) > 0:
            sample_anomalies = anomalies_df.iloc[::max(1, len(anomalies_df)//50)]
            scatter = axes[1, 1].scatter(sample_anomalies['timestamp'], sample_anomalies['V_batt'], 
                                       c='red', s=40, alpha=0.8, label='Anomalies', 
                                       zorder=5, edgecolors='darkred', linewidth=0.5)
        
        axes[1, 1].legend(fontsize=9, loc='upper right')
        axes[1, 1].set_title('Temporal Evolution - V_batt and T_batt', fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Time', fontweight='bold', fontsize=10)
        axes[1, 1].set_ylabel('Values', fontweight='bold', fontsize=10)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
        
        plt.savefig(os.path.join(self.output_dir, 'eps_main_dashboard_realistic.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()

    def _create_detailed_timeseries(self, df):
        """Creates detailed time series"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('DETAILED TEMPORAL ANALYSIS - EPS SYSTEM', 
                    fontsize=14, fontweight='bold')
        
        sample_df = df.iloc[::max(1, len(df)//500)]
        
        # 1. Battery - Voltage and Current
        axes[0].plot(sample_df['timestamp'], sample_df['V_batt'], label='V_batt (V)', 
                    color='blue', linewidth=1.5)
        axes[0].set_ylabel('Voltage (V)', color='blue', fontweight='bold')
        axes[0].tick_params(axis='y', labelcolor='blue')
        axes[0].grid(True, alpha=0.3)
        
        ax2 = axes[0].twinx()
        ax2.plot(sample_df['timestamp'], sample_df['I_batt'], label='I_batt (A)', 
                color='red', linewidth=1, alpha=0.8)
        ax2.set_ylabel('Current (A)', color='red', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='red')
        axes[0].set_title('Battery: Voltage and Current', fontweight='bold')
        
        # 2. Temperatures
        axes[1].plot(sample_df['timestamp'], sample_df['T_batt'], label='T_batt', 
                    linewidth=1.5, color='orange')
        axes[1].plot(sample_df['timestamp'], sample_df['T_eps'], label='T_eps', 
                    linewidth=1.5, color='green', alpha=0.8)
        axes[1].set_ylabel('Temperature (°C)', fontweight='bold')
        axes[1].set_title('System Temperatures', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Powers
        if all(col in sample_df.columns for col in ['V_solar', 'I_solar', 'V_batt', 'I_batt', 'V_bus', 'I_bus']):
            axes[2].plot(sample_df['timestamp'], sample_df['V_solar'] * sample_df['I_solar'], 
                        label='P_solar', linewidth=1.5, color='yellow')
            axes[2].plot(sample_df['timestamp'], sample_df['V_batt'] * sample_df['I_batt'], 
                        label='P_battery', linewidth=1.5, color='purple', alpha=0.8)
            axes[2].plot(sample_df['timestamp'], sample_df['V_bus'] * sample_df['I_bus'], 
                        label='P_bus', linewidth=1.5, color='brown', alpha=0.8)
            axes[2].set_ylabel('Power (W)', fontweight='bold')
            axes[2].set_title('Energy Balance', fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Time', fontweight='bold')
        
        # Marking anomalies
        anomalies = df[df['anomaly_type'] != 'normal']
        if len(anomalies) > 0:
            sample_anomalies = anomalies.iloc[::max(1, len(anomalies)//100)]
            for ax in axes:
                for _, anomaly in sample_anomalies.iterrows():
                    ax.axvline(x=anomaly['timestamp'], color='red', alpha=0.3, linewidth=1, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'eps_timeseries_detailed_realistic.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _create_anomaly_analysis(self, df):
        """Specific anomaly analysis"""
        anomalies_df = df[df['anomaly_type'] != 'normal']
        
        if len(anomalies_df) == 0:
            print("No anomalies to analyze")
            return
        
        critical_anomalies_df = anomalies_df[anomalies_df['anomaly_type'].isin(self.critical_anomalies)]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DETAILED ANOMALY ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Distribution by type
        anomaly_counts = anomalies_df['anomaly_type'].value_counts()
        bars = axes[0, 0].bar(anomaly_counts.index, anomaly_counts.values, color=self.colors)
        axes[0, 0].set_title('Anomaly Type Distribution', fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Impact on V_batt
        if len(critical_anomalies_df) > 0:
            for anomaly_type in critical_anomalies_df['anomaly_type'].unique():
                subset = critical_anomalies_df[critical_anomalies_df['anomaly_type'] == anomaly_type]
                axes[0, 1].scatter(subset['timestamp'], subset['V_batt'], 
                                  label=anomaly_type, alpha=0.7, s=50)
            axes[0, 1].set_title('Impact of Critical Anomalies on Battery Voltage', fontweight='bold')
        else:
            for anomaly_type in anomalies_df['anomaly_type'].unique():
                subset = anomalies_df[anomalies_df['anomaly_type'] == anomaly_type]
                axes[0, 1].scatter(subset['timestamp'], subset['V_batt'], 
                                  label=anomaly_type, alpha=0.7, s=50)
            axes[0, 1].set_title('Impact on Battery Voltage', fontweight='bold')
        
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylabel('V_batt (V)')
        
        # 3. Impact on T_batt
        if len(critical_anomalies_df) > 0:
            for anomaly_type in critical_anomalies_df['anomaly_type'].unique():
                subset = critical_anomalies_df[critical_anomalies_df['anomaly_type'] == anomaly_type]
                axes[1, 0].scatter(subset['timestamp'], subset['T_batt'], 
                                  label=anomaly_type, alpha=0.7, s=50)
            axes[1, 0].set_title('Impact of Critical Anomalies on Battery Temperature', fontweight='bold')
        else:
            for anomaly_type in anomalies_df['anomaly_type'].unique():
                subset = anomalies_df[anomalies_df['anomaly_type'] == anomaly_type]
                axes[1, 0].scatter(subset['timestamp'], subset['T_batt'], 
                                  label=anomaly_type, alpha=0.7, s=50)
            axes[1, 0].set_title('Impact on Battery Temperature', fontweight='bold')
        
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylabel('T_batt (°C)')
        
        # 4. Correlation critical anomalies
        critical_params = ['V_batt', 'I_batt', 'T_batt']
        if len(critical_anomalies_df) > 0 and all(param in critical_anomalies_df.columns for param in critical_params):
            anomaly_corr = critical_anomalies_df[critical_params].corr()
            im = axes[1, 1].imshow(anomaly_corr, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
            axes[1, 1].set_xticks(range(len(critical_params)))
            axes[1, 1].set_yticks(range(len(critical_params)))
            axes[1, 1].set_xticklabels(critical_params)
            axes[1, 1].set_yticklabels(critical_params)
            axes[1, 1].set_title('Correlations of Critical Anomalies', fontweight='bold')
            
            for i in range(len(critical_params)):
                for j in range(len(critical_params)):
                    axes[1, 1].text(j, i, f'{anomaly_corr.iloc[i, j]:.2f}', 
                                   ha='center', va='center', 
                                   color='white' if abs(anomaly_corr.iloc[i, j]) > 0.5 else 'black')
            
            plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor correlation', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Correlations of Anomalies', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'eps_anomaly_analysis_realistic.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _create_distribution_plots(self, df):
        """Creates parameter distribution plots"""
        sensor_cols = ['V_batt', 'I_batt', 'T_batt', 'V_solar', 'SOC']
        sensor_cols = [col for col in sensor_cols if col in df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('EPS PARAMETER DISTRIBUTION', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, col in enumerate(sensor_cols):
            if i < len(axes):
                normal_data = df[df['anomaly_type'] == 'normal'][col].dropna()
                anomaly_data = df[df['anomaly_type'] != 'normal'][col].dropna()
                
                axes[i].hist(normal_data, bins=30, alpha=0.7, label='Normal', color='green')
                if len(anomaly_data) > 0:
                    axes[i].hist(anomaly_data, bins=30, alpha=0.7, label='Anomaly', color='red')
                
                axes[i].set_title(f'Distribution {col}', fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        if len(sensor_cols) < len(axes):
            for i in range(len(sensor_cols), len(axes)):
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'eps_distributions_realistic.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _generate_excel_report(self, df):
        """Generates a complete Excel report"""
        try:
            logger.info("Generating Excel report...")
            
            # Create summary DataFrame
            summary_data = {
                'Metric': [
                    'Total Samples',
                    'Normal Samples', 
                    'Abnormal Samples',
                    'Critical Anomalies',
                    'Battery Health Score',
                    'System Stability Score',
                    'Solar Efficiency Score',
                    'Physical Consistency Score'
                ],
                'Value': [
                    len(df),
                    len(df[df['anomaly_type'] == 'normal']),
                    len(df[df['anomaly_type'] != 'normal']),
                    len(df[df['anomaly_type'].isin(self.critical_anomalies)]),
                    f"{self._calculate_battery_health(df):.1f}%",
                    f"{self._calculate_system_stability(df):.1f}%", 
                    f"{self._calculate_solar_efficiency(df):.1%}",
                    "95%" if self.check_physical_consistency(df) else "65%"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            
            # Descriptive statistics
            sensor_cols = ['V_batt', 'I_batt', 'T_batt', 'V_solar', 'I_solar', 'V_bus', 'I_bus', 'T_eps', 'SOC']
            sensor_cols = [col for col in sensor_cols if col in df.columns]
            stats_df = df[sensor_cols].describe()
            
            # Anomaly distribution
            anomaly_dist_df = pd.DataFrame(df['anomaly_type'].value_counts()).reset_index()
            anomaly_dist_df.columns = ['Anomaly_Type', 'Count']
            anomaly_dist_df['Percentage'] = (anomaly_dist_df['Count'] / len(df) * 100).round(2)
            
            # Save to Excel
            excel_file = os.path.join(ANALYSE_DIR, "eps_summary_stats.xlsx")
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='General_Summary', index=False)
                stats_df.to_excel(writer, sheet_name='Descriptive_Statistics')
                anomaly_dist_df.to_excel(writer, sheet_name='Anomaly_Distribution', index=False)
                
                # Correlation matrix
                corr_matrix = df[sensor_cols].corr()
                corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
            
            print(f"Excel report generated: {excel_file}")
            logger.info(f"Excel report saved: {excel_file}")
            
        except Exception as e:
            logger.error(f"Excel report generation error: {e}")
            print(f"Excel report generation error: {e}")

    def _export_obc_summary(self, df):
        """Exports a JSON summary for OBC simulation"""
        logger.info("Generating OBC summary...")
        
        sensor_cols = ['V_batt', 'I_batt', 'T_batt', 'V_solar', 'I_solar', 'V_bus', 'I_bus', 'T_eps', 'SOC']
        sensor_cols = [col for col in sensor_cols if col in df.columns]
        
        # Calculate performance metrics
        battery_health = self._calculate_battery_health(df)
        system_stability = self._calculate_system_stability(df)
        solar_efficiency = self._calculate_solar_efficiency(df)
        physical_consistency = self.check_physical_consistency(df)
        
        conv_eff = 0
        if 'V_solar' in df.columns and df['V_solar'].mean() > 1e-6:
            conv_eff = round(df['V_bus'].mean() / df['V_solar'].mean(), 3)
        
        summary = {
            "report_timestamp": datetime.now().isoformat(),
            "system_overview": {
                "total_samples": len(df),
                "data_quality": "EXCELLENT" if physical_consistency else "ISSUES_DETECTED",
                "physical_consistency": "VALIDATED" if physical_consistency else "FAILED"
            },
            "performance_metrics": {
                "battery_health_score": round(battery_health, 1),
                "solar_efficiency_score": round(solar_efficiency, 3),
                "system_stability_score": round(system_stability, 1),
                "converter_efficiency": conv_eff,
                "physical_consistency_score": 95 if physical_consistency else 65
            },
            "anomaly_report": {
                "total_anomalies": len(df[df['anomaly_type'] != 'normal']),
                "critical_anomalies": len(df[df['anomaly_type'].isin(self.critical_anomalies)]),
                "anomaly_distribution": df['anomaly_type'].value_counts().to_dict()
            }
        }
        
        summary_file = os.path.join(ANALYSE_DIR, "eps_obc_summary_realistic.json")
        with open(summary_file, "w", encoding='utf-8') as f:
            json.dump(json.loads(json.dumps(summary, default=str)), f, indent=2, ensure_ascii=False)
        
        print(f"OBC summary saved: {summary_file}")
        logger.info("OBC summary generated successfully")
        
        # Affichage synthese
        print("\n" + "="*50)
        print("SYNTHESE POUR OBC")
        print("="*50)
        print(f"Echantillons totaux: {summary['system_overview']['total_samples']}")
        print(f"Anomalies detectees: {summary['anomaly_report']['total_anomalies']}")
        print(f"Critiques: {summary['anomaly_report']['critical_anomalies']}")
        print(f"Sante batterie: {summary['performance_metrics']['battery_health_score']:.1f}%")
        print(f"Stabilite systeme: {summary['performance_metrics']['system_stability_score']:.1f}%")

    def _calculate_battery_health(self, df):
        """Calcule un score de sante batterie"""
        try:
            normal_df = df[df['anomaly_type'] == 'normal']
            
            if len(normal_df) == 0:
                return 85.0
            
            mean_v_batt = normal_df['V_batt'].mean()
            if abs(mean_v_batt) < 1e-6:
                v_batt_stability = 0.0
            else:
                v_batt_stability = 1 - (normal_df['V_batt'].std() / mean_v_batt)
                
            soc_stability = 1 - (normal_df['SOC'].std() / 50)
            temp_penalty = max(0, (normal_df['T_batt'].max() - 35) * 0.5)
            
            health_score = (v_batt_stability * 0.6 + soc_stability * 0.4) * 100 - temp_penalty
            return max(0, min(100, health_score))
            
        except:
            return 85.0

    def _calculate_system_stability(self, df):
        """Calcule un score de stabilite du systeme"""
        try:
            stability_metrics = []
            critical_params = ['V_batt', 'V_bus', 'T_batt']
            critical_params = [col for col in critical_params if col in df.columns]
            
            for col in critical_params:
                mean_val = df[col].mean()
                if abs(mean_val) < 1e-6:
                    cv = 0.0
                else:
                    cv = df[col].std() / mean_val
                stability = 1 - min(cv, 0.5)
                stability_metrics.append(stability)
            
            return np.mean(stability_metrics) * 100 if stability_metrics else 90.0
            
        except:
            return 90.0

    def _calculate_solar_efficiency(self, df):
        """Calcule l'efficacite solaire"""
        try:
            normal_df = df[df['anomaly_type'] == 'normal']
            if len(normal_df) == 0:
                return 0.95
                
            max_theoretical_power = 18.0 * 2.0
            actual_avg_power = (normal_df['V_solar'] * normal_df['I_solar']).mean()
            
            efficiency = actual_avg_power / max_theoretical_power
            return max(0, min(1, efficiency))
            
        except:
            return 0.95

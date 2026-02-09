#!/usr/bin/env python3
import os
import sys
import logging
import time
import argparse
from datetime import datetime
import pandas as pd

# === CORRECTED PATH CONFIGURATION ===
BASE_DIR = r"D:\final_year_project\Cubesat_AD"
DATA_DIR = os.path.join(BASE_DIR, "data")
ANALYSE_DIR = os.path.join(DATA_DIR, "analyse")
VISUALIZATIONS_DIR = os.path.join(ANALYSE_DIR, "visualizations")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "data_train")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

# REMOVE ALL os.makedirs() LINES - YOU ALREADY HAVE THE DIRECTORIES

# Logging configuration - CONSOLE ONLY
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # DELETE FileHandler
)

logger = logging.getLogger(__name__)

# Add the current path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EPS Guardian Simulation - CubeSat Energy Monitoring')
    parser.add_argument('--fast', action='store_true', help='Fast mode with fewer samples')
    parser.add_argument('--debug', action='store_true', help='Debug mode with detailed logging')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed for reproducibility')
    return parser.parse_args()

def print_banner():
    """Displays a simplified banner"""
    banner = """
=================================================================
        EPS GUARDIAN - ENERGY MONITORING SYSTEM
                   FOR CUBESAT AESS/IES
=================================================================
    """
    print(banner)

def check_environment():
    """Verifies environment and dependencies"""
    logger.info("Verifying environment...")
    
    # VERIFICATION ONLY - CRITICAL DIRECTORIES ONLY
    required_dirs = {
        "Data": DATA_DIR,
        "Analyse": ANALYSE_DIR,
        "Visualizations": VISUALIZATIONS_DIR,
        "Dataset": DATASET_DIR,
    }
    
    missing_dirs = []
    for name, dir_path in required_dirs.items():
        if not os.path.exists(dir_path):
            missing_dirs.append(f"{name}: {dir_path}")
    
    if missing_dirs:
        logger.error(f"Missing directories: {missing_dirs}")
        return False
    
    # Module verification
    required_modules = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'openpyxl']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing modules: {missing_modules}")
        return False
    
    logger.info("Environment verified")
    return True


def phase_generation(args):
    """Phase 1: EPS data generation"""
    print("\n" + "="*70)
    print("PHASE 1: EPS DATASET GENERATION")
    print("="*70)
    
    try:
        # Direct import to avoid path issues
        from pro_eps_data_generator import ProEPSSensorDataGenerator
        
        logger.info("Initializing generator...")
        generator = ProEPSSensorDataGenerator(random_seed=args.seed)
        
        # Generation configuration
        if args.fast:
            config = {
                'num_normal': 500,
                'num_anomalies': 100,
                'duration_hours': 24,
                'low_res': True
            }
            print("Fast mode active")
        else:
            config = {
                'num_normal': 5000,
                'num_anomalies': 1000,
                'duration_hours': 24,
                'low_res': False
            }
        
        print(f"Generation configuration:")
        print(f"   Normal samples: {config['num_normal']}")
        print(f"   Anomalies: {config['num_anomalies']}")
        print(f"   RNG Seed: {args.seed}")
        
        logger.info(f"Generation start: {config}")
        
        # Dataset generation
        start_time = time.time()
        df = generator.generate_dataset(**config)
        generation_time = time.time() - start_time
    
        print(f"Generation completed in {generation_time:.1f}s")
        print(f"Samples generated: {len(df)}")
        
        # Calculate derived features
        df = generator.calculate_derived_features(df)
        
        # Save dataset
        metadata = generator.save_dataset(df, "pro_eps_dataset.csv")
        
        if metadata:
            print(f"Dataset saved: {os.path.join(DATASET_DIR, 'pro_eps_dataset.csv')}")
            
            # Generation statistics
            print(f"\nGeneration statistics:")
            print(f"   Total: {metadata['dataset_info']['total_samples']} samples")
            print(f"   Normal: {metadata['dataset_info']['normal_samples']}")
            print(f"   Anomalies: {metadata['dataset_info']['anomaly_samples']}")
            print(f"   Anomaly types: {len(metadata['anomaly_distribution'])}")
        
        return df
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        print(f"Generation error: {e}")
        return None

def phase_analysis(df, args):
    """Phase 2: Complete data analysis"""
    print("\n" + "="*70)
    print("PHASE 2: COMPLETE ANALYSIS")
    print("="*70)
    
    try:
        from pro_eps_analyzer import ProEPSAnalyzer
        
        logger.info("Initializing analyzer...")
        analyzer = ProEPSAnalyzer()
        
        print("Starting analysis...")
        start_time = time.time()
        
        # Complete analysis
        analyzer.analyze_dataset("pro_eps_dataset.csv")
        
        analysis_time = time.time() - start_time
        print(f"Analysis completed in {analysis_time:.1f}s")
        
        # Verify generated files
        output_files = [
            'eps_main_dashboard_realistic.png',
            'eps_timeseries_detailed_realistic.png', 
            'eps_anomaly_analysis_realistic.png',
            'eps_distributions_realistic.png'
        ]
        
        json_files = [
            'eps_obc_summary_realistic.json'
        ]
        
        excel_files = [
            'eps_summary_stats.xlsx'
        ]
        
        print(f"\nFiles generated in {VISUALIZATIONS_DIR}:")
        for file in output_files:
            file_path = os.path.join(VISUALIZATIONS_DIR, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024
                print(f"   {file} ({file_size:.1f} KB)")
        
        print(f"\nFiles generated in {ANALYSE_DIR}:")
        for file in json_files + excel_files:
            file_path = os.path.join(ANALYSE_DIR, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024
                print(f"   {file} ({file_size:.1f} KB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        print(f"Analysis error: {e}")
        return False

def phase_simulation_mcu_obc():
    """Phase 3: MCU + OBC architecture simulation"""
    print("\n" + "="*70)
    print("PHASE 3: MCU + OBC ARCHITECTURE SIMULATION")
    print("="*70)
    
    try:
        print("MCU Simulation (Simple AI):")
        print("   Real-time monitoring")
        print("   Deterministic anomaly detection")
        print("   Immediate protection actions")
        
        print("\nOBC Simulation (Complex AI):")
        print("   Deep pattern analysis")
        print("   Strategic optimization")
        print("   Long-term decision making")
        
        print("\nHybrid workflow:")
        print("   1. MCU detection -> immediate alert")
        print("   2. MCU coherence verification")
        print("   3. MCU alerts OBC -> cause analysis")
        print("   4. OBC optimization -> new parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"MCU/OBC simulation error: {e}")
        return False

def generate_summary_table(df, simulation_time, args):
    """Generates a CSV summary table"""
    try:
        from pro_eps_analyzer import ProEPSAnalyzer
        
        # Calculate metrics
        analyzer = ProEPSAnalyzer()
        battery_health = analyzer._calculate_battery_health(df)
        system_stability = analyzer._calculate_system_stability(df)
        solar_efficiency = analyzer._calculate_solar_efficiency(df)
        physical_consistency = analyzer.check_physical_consistency(df)
        
        # Create summary table
        summary_data = {
            'simulation_date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'simulation_mode': ['FAST' if args.fast else 'FULL'],
            'total_samples': [len(df)],
            'normal_samples': [len(df[df['anomaly_type'] == 'normal'])],
            'anomaly_samples': [len(df[df['anomaly_type'] != 'normal'])],
            'critical_anomalies': [len(df[df['anomaly_type'].isin(['batt_overheat', 'batt_undervoltage', 'batt_overcurrent'])])],
            'battery_health_score': [round(battery_health, 1)],
            'system_stability_score': [round(system_stability, 1)],
            'solar_efficiency_score': [round(solar_efficiency, 3)],
            'physical_consistency': ['VALIDATED' if physical_consistency else 'FAILED'],
            'simulation_time_seconds': [round(simulation_time, 1)],
            'data_quality': ['HIGH' if len(df) > 1000 else 'MEDIUM'],
            'random_seed': [args.seed]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_file = os.path.join(ANALYSE_DIR, "simulation_summary_table.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Summary table generated: {summary_file}")
        return summary_df
        
    except Exception as e:
        logger.error(f"Summary table generation error: {e}")
        return None

def generate_final_report(df, simulation_time, args):
    """Generates final simulation report"""
    print("\n" + "="*70)
    print("FINAL SIMULATION REPORT")
    print("="*70)
    
    import json
    
    try:
        # Load OBC summary
        summary_path = os.path.join(ANALYSE_DIR, 'eps_obc_summary_realistic.json')
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            print(f"\nPerformance summary:")
            print(f"   Battery health: {summary['performance_metrics']['battery_health_score']}%")
            print(f"   System stability: {summary['performance_metrics']['system_stability_score']}%")
            print(f"   Solar efficiency: {summary['performance_metrics']['solar_efficiency_score']:.1%}")
            print(f"   Physical consistency: {summary['system_overview']['physical_consistency']}")
            
            print(f"\nAnomaly report:")
            print(f"   Total anomalies: {summary['anomaly_report']['total_anomalies']}")
            print(f"   Critical anomalies: {summary['anomaly_report']['critical_anomalies']}")
        
        # Generate summary table
        summary_table = generate_summary_table(df, simulation_time, args)
        
        # Verify files
        print(f"\nGenerated files:")
        files_to_check = {
            'Main Dataset': os.path.join(DATASET_DIR, 'pro_eps_dataset.csv'),
            'Metadata': os.path.join(DATASET_DIR, 'pro_eps_dataset_metadata.json'),
            'Dashboard': os.path.join(VISUALIZATIONS_DIR, 'eps_main_dashboard_realistic.png'),
            'Time Series': os.path.join(VISUALIZATIONS_DIR, 'eps_timeseries_detailed_realistic.png'),
            'Anomaly Analysis': os.path.join(VISUALIZATIONS_DIR, 'eps_anomaly_analysis_realistic.png'),
            'Distributions': os.path.join(VISUALIZATIONS_DIR, 'eps_distributions_realistic.png'),
            'OBC Report': os.path.join(ANALYSE_DIR, 'eps_obc_summary_realistic.json'),
            'Excel Report': os.path.join(ANALYSE_DIR, 'eps_summary_stats.xlsx'),
            'Summary Table': os.path.join(ANALYSE_DIR, 'simulation_summary_table.csv')
        }
        
        for description, file_path in files_to_check.items():
            if os.path.exists(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                filename = os.path.basename(file_path)
                print(f"   {description}: {filename} ({size_kb:.1f} KB)")
                
    except Exception as e:
        print(f"Report generation error: {e}")

def main():
    """Main function orchestrating the simulation"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Debug logging configuration if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("DEBUG mode active")
    
    # Display banner
    print_banner()
    
    # Start timer
    start_time = time.time()
    simulation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\nSimulation start: {simulation_date}")
    print(f"Data directory: {DATA_DIR}")
    print(f"RNG Seed: {args.seed}")
    
    if args.fast:
        print("Fast mode active")
    
    # Environment verification
    if not check_environment():
        print("Simulation stopped - environment non-compliant")
        return 1
    
    try:
        # Phase 1: Data generation
        df = phase_generation(args)
        if df is None:
            return 1
        
        # Phase 2: Complete analysis
        if not phase_analysis(df, args):
            return 1
        
        # Phase 3: Architecture simulation
        if not phase_simulation_mcu_obc():
            return 1
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Final report
        generate_final_report(df, total_time, args)
        
        print(f"\nSIMULATION COMPLETED SUCCESSFULLY!")
        print(f"Total time: {total_time:.1f}s")
        print(f"Results in: {DATA_DIR}")
        print(f"  - Dataset: {DATASET_DIR}")
        print(f"  - Visualizations: {VISUALIZATIONS_DIR}")
        print(f"  - Analysis: {ANALYSE_DIR}")
        print(f"  - Logs: {LOGS_DIR}")
        
        logger.info(f"Simulation completed successfully in {total_time:.1f}s")
        
        # Force log flush for Windows
        for handler in logging.getLogger().handlers:
            handler.flush()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        logger.info("Simulation interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\nCritical error: {e}")
        logger.error(f"Critical error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

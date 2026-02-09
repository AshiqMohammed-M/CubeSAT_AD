# EPS Guardian - Translation, Flow, and Function Documentation


## Task 2: Flow of the src Folder

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    EPS GUARDIAN SYSTEM                      │
│                  (CubeSat Power Management)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
        ┌───────▼───────┐         ┌────────▼────────┐
        │  MCU Module   │◄────────┤  OBC Module     │
        │  (Embedded)   │  Messages│  (On-Board)    │
        └───────┬───────┘         └─────────────────┘
                │
        ┌───────▼──────────────────────────────────┐
        │      Dataset Generation & Training        │
        └──────────────────────────────────────────┘
```

### Module Flow

#### 1. **Dataset Physical** (`dataset_physical/`)
**Purpose**: Generate and analyze physical EPS sensor data

**Flow**:
```
main_simulation.py (Orchestrator)
    │
    ├─► pro_eps_data_generator.py
    │   └─► Generates realistic EPS sensor data (V_batt, I_batt, T_batt, etc.)
    │       └─► Output: pro_eps_dataset.csv
    │
    ├─► pro_eps_analyzer.py
    │   └─► Analyzes dataset, creates visualizations, generates reports
    │       └─► Output: Analysis reports, visualizations, JSON summaries
    │
    └─► ai_preprocessor.py
        └─► Prepares data for AI training (normalization, feature extraction)
            └─► Output: Training-ready data in training_data/
```

**Key Files**:
- `main_simulation.py`: Main orchestrator for data generation pipeline
- `pro_eps_data_generator.py`: Generates realistic EPS sensor data with anomalies
- `pro_eps_analyzer.py`: Performs statistical analysis and visualization
- `ai_preprocessor.py`: Preprocesses data for machine learning

#### 2. **Dataset Extended** (`dataset_extended/`)
**Purpose**: Create extended temporal sequences for complex AI models (LSTM/Autoencoder)

**Flow**:
```
main_extended_simulation.py
    │
    └─► ComplexDataGenerator
        ├─► Loads base dataset from dataset_physical
        ├─► Creates temporal windows (30 timesteps)
        ├─► Calculates derived features
        ├─► Normalizes sequences
        └─► Output: 
            - ai_sequence_data.npy (sequences)
            - ai_sequence_labels.npy (NORMAL/WARNING/CRITICAL)
            - ai_sequence_features.npy (derived features)
            - Scalers for normalization
```

**Key Files**:
- `main_extended_simulation.py`: Generates complex temporal sequences
- `Diagnostic.py`: Diagnostic tool for normalization issues
- `verify_improved_dataset.py`: Verifies dataset integrity

#### 3. **MCU Module** (`mcu/`)
**Purpose**: Embedded system simulation - real-time monitoring and rule-based detection

**Flow**:
```
mcu_main_loop.py (Main Loop)
    │
    ├─► mcu_data_interface.py
    │   └─► Reads sensor data (CSV or simulated)
    │
    ├─► mcu_rule_engine.py
    │   ├─► Applies safety rules (R1-R7)
    │   ├─► Detects anomalies (overheat, overcurrent, etc.)
    │   └─► Sends alerts to OBC via obc_interface
    │
    ├─► mcu_resource_monitor.py
    │   └─► Monitors CPU/memory usage
    │
    ├─► mcu_logger.py
    │   └─► Logging system
    │
    └─► mcu_ai/ (AI Subsystem)
        ├─► ai_model_trainer.py
        │   └─► Trains lightweight autoencoder for MCU
        │
        ├─► ai_model_inference.py
        │   └─► Runs AI inference on MCU
        │
        ├─► ai_model_evaluator.py
        │   └─► Evaluates model performance
        │
        ├─► ai_model_converter.py
        │   └─► Converts model to TensorFlow Lite for ESP32
        │
        └─► mcu_ai_main_loop.py
            └─► Hybrid system: Rules + AI fusion
```

**Key Files**:
- `mcu_main_loop.py`: Main simulation loop for MCU
- `mcu_rule_engine.py`: Rule-based anomaly detection (R1-R7)
- `mcu_data_interface.py`: Data interface for reading sensor data
- `mcu_resource_monitor.py`: Resource monitoring
- `mcu_logger.py`: Logging system
- `mcu_ai/`: AI subsystem for advanced detection

#### 4. **OBC Module** (`obc/`)
**Purpose**: On-Board Computer - complex AI analysis and strategic decision-making

**Flow**:
```
obc_main.py (Main Entry Point)
    │
    ├─► interface/
    │   ├─► obc_message_handler.py
    │   │   └─► Receives and processes MCU messages
    │   │       └─► Extracts temporal window data
    │   │
    │   └─► obc_response_generator.py
    │       └─► Generates structured responses for MCU
    │
    ├─► ai/
    │   ├─► ai_complex_trainer.py
    │   │   └─► Trains LSTM Autoencoder for temporal anomaly detection
    │   │
    │   ├─► ai_complex_inference.py
    │   │   └─► Runs complex AI analysis on temporal sequences
    │   │
    │   └─► ai_model_converter.py
    │       └─► Converts complex model to TFLite for deployment
    │
    └─► simulation/
        ├─► obc_simulate_incoming_data.py
        │   └─► Simulates incoming MCU messages for testing
        │
        └─► obc_realtime_fusion_test.py
            └─► Tests real-time AI fusion
```

**Key Files**:
- `obc_main.py`: Main OBC system orchestrator
- `interface/obc_message_handler.py`: Handles MCU messages
- `interface/obc_response_generator.py`: Generates OBC responses
- `ai/ai_complex_inference.py`: Complex AI inference (LSTM Autoencoder)
- `ai/ai_complex_trainer.py`: Trains complex AI models

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA GENERATION PHASE                     │
└──────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                             │
┌───────▼────────┐                          ┌───────▼────────┐
│ dataset_       │                          │ dataset_       │
│ physical/      │                          │ extended/      │
│                │                          │                │
│ Generates      │                          │ Creates        │
│ base dataset   │                          │ temporal       │
│ (CSV)          │                          │ sequences      │
└───────┬────────┘                          └───────┬────────┘
        │                                             │
        └─────────────────────┬─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  pro_eps_dataset  │
                    │      .csv         │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                             │
┌───────▼────────┐                          ┌───────▼────────┐
│   MCU Module   │                          │   OBC Module   │
│                │                          │                │
│ Reads dataset  │                          │ Uses extended │
│ Applies rules  │                          │ sequences     │
│ Sends alerts   │─────────────────────────►│ AI analysis   │
│                │      JSON Messages       │ Returns       │
│                │◄─────────────────────────│ decisions     │
└────────────────┘                          └────────────────┘
```

### Interconnection Summary

1. **Dataset Physical → Dataset Extended**
   - `pro_eps_dataset.csv` is used as input for extended sequence generation

2. **Dataset Extended → MCU/OBC AI Training**
   - Extended sequences feed into AI training pipelines
   - MCU uses simple autoencoder
   - OBC uses complex LSTM Autoencoder

3. **MCU → OBC Communication**
   - MCU sends JSON messages via `mcu_rule_engine.py` → `obc_interface`
   - OBC receives via `obc_message_handler.py`
   - OBC responds via `obc_response_generator.py`

4. **AI Models Flow**
   - Training: `ai_model_trainer.py` (MCU) / `ai_complex_trainer.py` (OBC)
   - Inference: `ai_model_inference.py` (MCU) / `ai_complex_inference.py` (OBC)
   - Conversion: `ai_model_converter.py` (both) → TensorFlow Lite for ESP32

---

## Task 3: Overall View Functions of Each File

### Dataset Physical Module

#### `main_simulation.py`
**Main Functions**:
- `main()`: Orchestrates the entire simulation pipeline
- `phase_generation()`: Phase 1 - Generates EPS dataset
- `phase_analysis()`: Phase 2 - Analyzes generated data
- `phase_simulation_mcu_obc()`: Phase 3 - Simulates MCU+OBC architecture
- `generate_final_report()`: Generates comprehensive final report

**Purpose**: Main orchestrator that runs the complete data generation and analysis pipeline

#### `pro_eps_data_generator.py`
**Main Functions**:
- `generate_normal_data()`: Generates normal EPS sensor readings
- `generate_anomaly()`: Generates various anomaly types (overheat, undervoltage, etc.)
- `generate_dataset()`: Creates complete dataset with normal + anomaly samples
- `calculate_derived_features()`: Calculates power, ratios, deltas, rolling stats
- `save_dataset()`: Saves dataset with metadata

**Purpose**: Generates realistic EPS sensor data with physical consistency

#### `pro_eps_analyzer.py`
**Main Functions**:
- `analyze_dataset()`: Main analysis function
- `check_dataset_integrity()`: Validates data quality
- `check_physical_consistency()`: Verifies physical limits
- `_create_comprehensive_plots()`: Generates visualizations
- `_export_obc_summary()`: Creates JSON summary for OBC
- `_generate_excel_report()`: Creates Excel report

**Purpose**: Analyzes dataset, creates visualizations, and generates reports

#### `ai_preprocessor.py`
**Main Functions**:
- `load_dataset()`: Loads CSV dataset
- `select_features()`: Selects relevant features
- `prepare_training_data()`: Prepares data for ML (normal samples only)
- `normalize_data()`: Normalizes features
- `save_processed_data()`: Saves preprocessed data

**Purpose**: Prepares data for AI training (normalization, feature selection)

### Dataset Extended Module

#### `main_extended_simulation.py`
**Main Functions**:
- `main()`: Entry point for extended dataset generation
- `ComplexDataGenerator` class:
  - `load_base_data()`: Loads base dataset
  - `generate_sequences()`: Creates temporal sequences (30 timesteps)
  - `calculate_derived_features()`: Calculates window-based features
  - `assign_window_label()`: Labels sequences (NORMAL/WARNING/CRITICAL)
  - `save_datasets()`: Saves all generated datasets
  - `generate_statistics()`: Generates statistics and visualizations

**Purpose**: Creates temporal sequences for complex AI models (LSTM/Autoencoder)

### MCU Module

#### `mcu_main_loop.py`
**Main Functions**:
- `MCU_MainLoop` class:
  - `__init__()`: Initializes rule engine, data interface, resource monitor
  - `process_sample()`: Processes single sensor sample
  - `run_simulation()`: Runs complete simulation loop
  - `save_results()`: Saves simulation results
  - `generate_report()`: Generates performance report

**Purpose**: Main simulation loop for MCU rule-based system

#### `mcu_rule_engine.py`
**Main Functions**:
- `MCU_RuleEngine` class:
  - `apply_rules()`: Applies safety rules R1-R7
  - `_check_overheat()`: R1 - Battery overheat detection
  - `_check_overcurrent()`: R2 - Overcurrent detection
  - `_check_deep_discharge()`: R3 - Deep discharge detection
  - `_check_dcdc_ratio()`: R4 - DC/DC converter ratio check
  - `_check_oscillation()`: R5 - Bus oscillation detection
  - `_check_sensor_fault()`: R6 - Sensor fault detection
  - `trigger_action()`: Triggers physical actions (LEDs, alarms, etc.)
  - `send_to_obc()`: Sends messages to OBC via interface

**Purpose**: Rule-based anomaly detection and alert generation

#### `mcu_data_interface.py`
**Main Functions**:
- `DataInterface` class:
  - `__init__()`: Initializes data source (CSV or simulated)
  - `get_next()`: Gets next sensor sample
  - `get_sample()`: Gets specific sample by index
  - `get_total_samples()`: Returns total sample count
  - `reset()`: Resets reading index

**Purpose**: Interface for reading sensor data

#### `mcu_resource_monitor.py`
**Main Functions**:
- `measure_resource_usage()`: Decorator for measuring execution time and memory
- `ResourceMonitor` class:
  - `get_current_usage()`: Gets current CPU/memory usage
  - `save_usage_data()`: Saves usage data to CSV/JSON

**Purpose**: Monitors system resource usage

#### `mcu_ai/ai_model_trainer.py`
**Main Functions**:
- `AIModelTrainer` class:
  - `load_training_data()`: Loads preprocessed training data
  - `create_autoencoder()`: Creates lightweight autoencoder architecture
  - `train_model()`: Trains the autoencoder
  - `calculate_anomaly_threshold()`: Calculates anomaly detection thresholds
  - `save_model()`: Saves model and artifacts
  - `evaluate_model_size()`: Checks if model fits ESP32 constraints

**Purpose**: Trains lightweight autoencoder for MCU deployment

#### `mcu_ai/ai_model_inference.py`
**Main Functions**:
- `AIModelInference` class:
  - `load_artifacts()`: Loads model, scaler, thresholds
  - `predict_from_normalized()`: Runs inference on normalized features
  - `test_real_scenarios()`: Tests with real training data
  - `test_synthetic_normal()`: Tests with synthetic data
  - `save_results()`: Saves inference results

**Purpose**: Runs AI inference for anomaly detection

#### `mcu_ai/mcu_ai_main_loop.py`
**Main Functions**:
- `MCUAI_MainLoop` class:
  - `load_ai_model()`: Loads AI model and artifacts
  - `prepare_features()`: Prepares sensor features for AI
  - `ai_anomaly_detection()`: Runs AI-based anomaly detection
  - `hybrid_decision_fusion()`: Fuses AI + rule-based decisions
  - `hybrid_decision_pipeline()`: Complete hybrid decision pipeline
  - `run_hybrid_simulation()`: Runs hybrid simulation

**Purpose**: Hybrid system combining rule-based and AI-based detection

### OBC Module

#### `obc_main.py`
**Main Functions**:
- `OBCSystem` class:
  - `__init__()`: Initializes OBC system components
  - `start_system()`: Starts the OBC system
  - `process_single_message()`: Processes single MCU message
  - `run_continuous_mode()`: Runs continuous listening mode
  - `stop_system()`: Stops the system gracefully
  - `get_system_status()`: Returns system status
  - `_save_response()`: Saves OBC responses
  - `_save_system_status()`: Saves system status

**Purpose**: Main OBC system orchestrator

#### `interface/obc_message_handler.py`
**Main Functions**:
- `OBCMessageHandler` class:
  - `process_mcu_message()`: Main message processing function
  - `_extract_temporal_data()`: Extracts temporal window from message
  - `_simulate_ai_analysis()`: Simulates AI analysis if model unavailable
  - `_generate_obc_response()`: Generates OBC response based on AI analysis

**Purpose**: Handles incoming MCU messages and triggers AI analysis

#### `interface/obc_response_generator.py`
**Main Functions**:
- `OBCResponseGenerator` class:
  - `generate_response()`: Generates structured response for MCU
  - `generate_heartbeat()`: Generates heartbeat message
  - `generate_error_response()`: Generates error response
  - `_determine_priority()`: Determines message priority

**Purpose**: Generates structured responses for MCU

#### `ai/ai_complex_inference.py`
**Main Functions**:
- `OBC_AI` class:
  - `__init__()`: Initializes OBC AI system
  - `load_model_and_thresholds()`: Loads LSTM Autoencoder model
  - `analyze_sequence()`: Analyzes temporal sequence (30 timesteps)
  - `_simulate_analysis()`: Simulates analysis if model unavailable
  - `_classify_anomaly()`: Classifies anomaly level (NORMAL/WARNING/CRITICAL)
  - `get_model_info()`: Returns model information

**Purpose**: Complex AI inference using LSTM Autoencoder for temporal anomaly detection

#### `ai/ai_complex_trainer.py`
**Main Functions**:
- `OBCAITrainer` class:
  - `load_training_data()`: Loads extended sequences
  - `prepare_data()`: Prepares data (uses only NORMAL sequences)
  - `build_lstm_autoencoder()`: Builds LSTM Autoencoder architecture
  - `train_model()`: Trains the model
  - `calculate_anomaly_thresholds()`: Calculates thresholds from reconstruction errors
  - `save_model_and_thresholds()`: Saves trained model and thresholds
  - `generate_training_report()`: Generates training visualization

**Purpose**: Trains complex LSTM Autoencoder for OBC

#### `simulation/obc_simulate_incoming_data.py`
**Main Functions**:
- `create_sample_mcu_message()`: Creates sample MCU message for testing
- `simulate_critical_anomaly()`: Creates message with critical anomaly
- `test_obc_system()`: Tests OBC system with various scenarios
- `save_test_results()`: Saves test results

**Purpose**: Simulates incoming MCU messages for OBC testing

---

## Summary

### System Flow Summary
1. **Data Generation**: `dataset_physical` → generates base dataset
2. **Extended Sequences**: `dataset_extended` → creates temporal sequences
3. **MCU Processing**: `mcu` → rule-based detection + simple AI
4. **OBC Processing**: `obc` → complex AI analysis on temporal patterns
5. **Communication**: MCU ↔ OBC via JSON messages

### Key Design Patterns
- **Hybrid Detection**: Rule-based (MCU) + AI-based (OBC)
- **Temporal Analysis**: OBC uses 30-timestep sequences for pattern detection
- **Modular Architecture**: Clear separation between MCU (embedded) and OBC (complex)
- **Data Pipeline**: Physical data → Extended sequences → AI training → Inference

### File Dependencies
- MCU depends on: `dataset_physical` (for data), `mcu_ai` (for AI)
- OBC depends on: `dataset_extended` (for sequences), `ai` (for complex models)
- Both use: Common data structures, JSON message format


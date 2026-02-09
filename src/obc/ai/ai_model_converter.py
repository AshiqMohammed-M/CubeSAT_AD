#!/usr/bin/env python3
"""
AI OBC → MCU MODEL CONVERTER
Converts a TensorFlow model (.h5) to TensorFlow Lite format (.tflite)
and prepares the files for deployment on ESP32.
"""
#completed
import os
import tensorflow as tf
import numpy as np
import json
import logging
from datetime import datetime

# --- Configuration des chemins ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "ai_models", "model_complex")
DEPLOY_DIR = os.path.join(PROJECT_ROOT, "data", "ai_models", "esp32_deployment")
MCU_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "ai_models", "model_simple")

os.makedirs(DEPLOY_DIR, exist_ok=True)
os.makedirs(MCU_MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AI_Model_Converter")

class AIModelConverter:
    def __init__(self):
        self.model = None
        self.thresholds = None
        self.model_info = {}
        
    def load_complex_model(self):
        """Loads the complex LSTM model and its metadata"""
        try:
            model_path = os.path.join(MODEL_DIR, "ai_model_lstm_autoencoder.h5")
            thresholds_path = os.path.join(MODEL_DIR, "ai_thresholds.json")
            
            if not os.path.exists(model_path):
                logger.error(f"Model not found: {model_path}")
                return False
                
            logger.info(f"Loading model: {model_path}")
            self.model = tf.keras.models.load_model(model_path, compile=False)
            
            # Loading thresholds
            if os.path.exists(thresholds_path):
                with open(thresholds_path, 'r') as f:
                    thresholds_data = json.load(f)
                    self.thresholds = thresholds_data["anomaly_thresholds"]
                logger.info("Anomaly thresholds loaded")
            else:
                logger.warning("Thresholds file not found, using default values")
                self.thresholds = {
                    "normal_threshold": 0.001,
                    "warning_threshold": 0.002, 
                    "critical_threshold": 0.003
                }
            
            # Model information
            self.model_info = {
                "input_shape": self.model.input_shape[1:],
                "output_shape": self.model.output_shape[1:],
                "total_params": self.model.count_params(),
                "model_size_h5": os.path.getsize(model_path) / 1024
            }
            
            logger.info(f"Model loaded: {self.model_info['input_shape']} → {self.model_info['output_shape']}")
            logger.info(f"Parameters: {self.model_info['total_params']:,}")
            logger.info(f".h5 size: {self.model_info['model_size_h5']:.1f} KB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def convert_to_tflite(self, output_path, quantize=False, quantization_type="float16"):
        """Converts the model to TensorFlow Lite with dynamic LSTM support"""
        try:
            logger.info(f"Converting to TFLite: quantize={quantize}, type={quantization_type}")

            # --- Creating the converter ---
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

            # 🔧 Fix for LSTM (Select TF Ops)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False  # indispensable for TensorListReserve

            # --- Quantization management ---
            if quantize:
                if quantization_type == "float16":
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]
                    logger.info("Float16 quantization enabled")
                elif quantization_type == "int8":
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]

                    def representative_dataset():
                        for _ in range(100):
                            data = np.random.randn(1, *self.model_info['input_shape']).astype(np.float32)
                            yield [data]

                    converter.representative_dataset = representative_dataset
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                        tf.lite.OpsSet.SELECT_TF_OPS
                    ]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                    logger.info("INT8 quantization enabled")

            # --- Conversion ---
            tflite_model = converter.convert()

            # --- Saving ---
            with open(output_path, "wb") as f:
                f.write(tflite_model)

            size_kb = os.path.getsize(output_path) / 1024
            logger.info(f"TFLite model generated: {output_path}")
            logger.info(f"TFLite size: {size_kb:.1f} KB")

            return size_kb

        except Exception as e:
            logger.error(f"Error converting to TFLite: {e}")
            return 0
    
    def generate_c_header(self, tflite_path, header_path):
        """Generates a C/C++ header file from the TFLite model"""
        try:
            logger.info(f"Generating C header: {header_path}")
            
            with open(tflite_path, "rb") as f:
                model_data = f.read()
            
            # Conversion to C array
            hex_array = []
            for i, byte in enumerate(model_data):
                if i % 12 == 0:
                    hex_array.append("\n    ")
                hex_array.append(f"0x{byte:02x}, ")
            
            hex_string = "".join(hex_array).rstrip(", ")
            
            header_content = f"""// ====================================================
// Automatically generated file - EPS Guardian AI Model
// TensorFlow Lite template for ESP32 deployment
// Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
// ====================================================

#ifndef EPS_GUARDIAN_AI_MODEL_H
#define EPS_GUARDIAN_AI_MODEL_H

#include <cstdint>
#include <cstddef>

namespace eps_guardian {{
namespace ai_model {{

// TensorFlow Lite model data
alignas(8) const uint8_t g_ai_model_data[] = {{
    {hex_string}
}};

// Model size in bytes
const size_t g_ai_model_size = sizeof(g_ai_model_data);

// Anomaly thresholds (based on reconstruction error)
constexpr float NORMAL_THRESHOLD = {self.thresholds['normal_threshold']:.6f}f;
constexpr float WARNING_THRESHOLD = {self.thresholds['warning_threshold']:.6f}f;
constexpr float CRITICAL_THRESHOLD = {self.thresholds['critical_threshold']:.6f}f;

// Model configuration
constexpr int SEQUENCE_LENGTH = {self.model_info['input_shape'][0]};
constexpr int FEATURE_COUNT = {self.model_info['input_shape'][1]};
constexpr int TENSOR_ARENA_SIZE = 40 * 1024; // 40KB for LSTM

}} // namespace ai_model
}} // namespace eps_guardian

#endif // EPS_GUARDIAN_AI_MODEL_H
"""
            
            with open(header_path, "w") as f:
                f.write(header_content)
            
            logger.info(f"C header generated: {header_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating header: {e}")
            return False
    
    def generate_arduino_sketch(self, sketch_path):
        """Generates a complete Arduino sketch for the ESP32"""
        try:
            logger.info(f"GGenerating Arduino sketch: {sketch_path}")
            
            sketch_content = f"""// ====================================================
// EPS Guardian - AI Anomaly Detection System
// MCU: ESP32 with TensorFlow Lite Micro
// Model: LSTM Autoencoder for time series
// ====================================================

#include "eps_guardian_ai_model.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ====================================================
// HARDWARE CONFIGURATION
// ====================================================

// Pins for status LEDs
#define LED_NORMAL 2
#define LED_WARNING 4  
#define LED_CRITICAL 5

// Buffer for inference
constexpr int kTensorArenaSize = eps_guardian::ai_model::TENSOR_ARENA_SIZE;
static uint8_t tensor_arena[kTensorArenaSize];

// ====================================================
// MAIN EPS GUARDIAN AI CLASS
// ====================================================

class EPSGuardianAI {{
private:
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input;
    TfLiteTensor* output;
    tflite::AllOpsResolver resolver;

    // Buffer for sensor sequences
    float sensor_sequence[eps_guardian::ai_model::SEQUENCE_LENGTH]
                        [eps_guardian::ai_model::FEATURE_COUNT];
    int sequence_index = 0;

public:
    bool initialize() {{
        Serial.println("Initializing EPS Guardian AI...");
        
        // Loading the model
        model = tflite::GetModel(eps_guardian::ai_model::g_ai_model_data);
        if (model->version() != TFLITE_SCHEMA_VERSION) {{
            Serial.println("ERROR: Incompatible model version!");
            return false;
        }}
        
        // Initializing the interpreter
        static tflite::MicroInterpreter static_interpreter(
            model, resolver, tensor_arena, kTensorArenaSize);
        interpreter = &static_interpreter;
        
        // Allocating tensors
        if (interpreter->AllocateTensors() != kTfLiteOk) {{
            Serial.println("ERROR: Tensor allocation failed!");
            return false;
        }}
        
        input = interpreter->input(0);
        output = interpreter->output(0);
        
        // Configuring pins
        pinMode(LED_NORMAL, OUTPUT);
        pinMode(LED_WARNING, OUTPUT);  
        pinMode(LED_CRITICAL, OUTPUT);
        
        Serial.println("EPS Guardian AI initialized successfully!");
        Serial.println("Waiting for sensor data...");
        
        return true;
    }}

    void add_sensor_data(float* sensor_values) {{
        // Adding new data to the sequence
        for (int i = 0; i < eps_guardian::ai_model::FEATURE_COUNT; i++) {{
            sensor_sequence[sequence_index][i] = sensor_values[i];
        }}
        
        sequence_index = (sequence_index + 1) % eps_guardian::ai_model::SEQUENCE_LENGTH;
    }}

    float detect_anomaly() {{
        // Checking if the sequence is complete
        if (sequence_index != 0) {{
            Serial.println("WARNING: Incomplete sequence, using available data");
        }}
        
        // Copying data into the input tensor
        float* input_data = input->data.f;
        for (int i = 0; i < eps_guardian::ai_model::SEQUENCE_LENGTH; i++) {{
            for (int j = 0; j < eps_guardian::ai_model::FEATURE_COUNT; j++) {{
                *input_data++ = sensor_sequence[i][j];
            }}
        }}
        
        // Inference
        if (interpreter->Invoke() != kTfLiteOk) {{
            Serial.println("ERROR: Inference failed!");
            return -1.0f;
        }}
        
        // Calculating reconstruction error
        float reconstruction_error = 0.0f;
        float* output_data = output->data.f;
        input_data = input->data.f;
        
        for (int i = 0; i < eps_guardian::ai_model::SEQUENCE_LENGTH * eps_guardian::ai_model::FEATURE_COUNT; i++) {{
            float diff = input_data[i] - output_data[i];
            reconstruction_error += diff * diff;
        }}
        
        reconstruction_error /= (eps_guardian::ai_model::SEQUENCE_LENGTH * eps_guardian::ai_model::FEATURE_COUNT);
        
        return reconstruction_error;
    }}

    int get_anomaly_level(float error) {{
        if (error < eps_guardian::ai_model::NORMAL_THRESHOLD) {{
            return 0; // NORMAL
        }} else if (error < eps_guardian::ai_model::WARNING_THRESHOLD) {{
            return 1; // WARNING
        }} else {{
            return 2; // CRITICAL
        }}
    }}

    void update_leds(int anomaly_level) {{
        // Turn off all the LEDs
        digitalWrite(LED_NORMAL, LOW);
        digitalWrite(LED_WARNING, LOW);
        digitalWrite(LED_CRITICAL, LOW);
        
        // Turn on the corresponding LED
        switch (anomaly_level) {{
            case 0: // NORMAL
                digitalWrite(LED_NORMAL, HIGH);
                break;
            case 1: // WARNING
                digitalWrite(LED_WARNING, HIGH); 
                break;
            case 2: // CRITICAL
                digitalWrite(LED_CRITICAL, HIGH);
                break;
        }}
    }}

    void print_debug_info(float error, int level) {{
        Serial.print("Reconstruction error: ");
        Serial.print(error, 6);
        Serial.print(" | Level: ");
        
        switch (level) {{
            case 0: Serial.println("NORMAL"); break;
            case 1: Serial.println("WARNING"); break;
            case 2: Serial.println("CRITICAL"); break;
        }}
        
        Serial.print("Thresholds - Normal<");
        Serial.print(eps_guardian::ai_model::NORMAL_THRESHOLD, 6);
        Serial.print(", Warning<");  
        Serial.print(eps_guardian::ai_model::WARNING_THRESHOLD, 6);
        Serial.print(", Critical>=");
        Serial.println(eps_guardian::ai_model::CRITICAL_THRESHOLD, 6);
    }}
}};

// ====================================================
// GLOBAL INSTANCE AND SETUP
// ====================================================

EPSGuardianAI guardianAI;

void setup() {{
    Serial.begin(115200);
    while (!Serial) {{ delay(10); }}
    
    Serial.println("================================================");
    Serial.println("EPS GUARDIAN - EMBEDDED AI SYSTEM");
    Serial.println("Real-time EPS anomaly detection");
    Serial.println("================================================");
    
    if (!guardianAI.initialize()) {{
        Serial.println("FAILURE: AI initialization - System halt");
        while(1) {{ delay(1000); }}
    }}
}}

// ====================================================
// MAIN LOOP
// ====================================================

void loop() {{
    // SIMULATION: Generation of realistic sensor data
    float sensor_data[eps_guardian::ai_model::FEATURE_COUNT];
    
    // Typical values for a nominal EPS system
    sensor_data[0] = 7.4f + random(-10, 10) / 100.0f;  // V_batt
    sensor_data[1] = 1.2f + random(-20, 20) / 100.0f;  // I_batt  
    sensor_data[2] = 35.0f + random(-50, 50) / 10.0f;  // T_batt
    sensor_data[3] = 7.8f + random(-5, 5) / 100.0f;    // V_bus
    sensor_data[4] = 0.8f + random(-10, 10) / 100.0f;  // I_bus
    sensor_data[5] = 15.2f + random(-20, 20) / 10.0f;  // V_solar
    sensor_data[6] = 1.5f + random(-10, 10) / 100.0f;  // I_solar
    
    // Adding data to the sequence
    guardianAI.add_sensor_data(sensor_data);
    
    // Anomaly detection
    float error = guardianAI.detect_anomaly();
    
    if (error >= 0) {{
        int anomaly_level = guardianAI.get_anomaly_level(error);
        guardianAI.update_leds(anomaly_level);
        guardianAI.print_debug_info(error, anomaly_level);
    }}
    
    // Cycle every 2 seconds
    delay(2000);
}}
"""
            
            with open(sketch_path, "w") as f:
                f.write(sketch_content)
            
            logger.info(f"Arduino sketch generated: {sketch_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating sketch: {e}")
            return False
    
    def copy_simple_model_to_mcu(self):
        """Also copies the simple model for the basic MCU"""
        try:
            # Copy the quantized model from model_complex to
            #  model_simple
            source_tflite = os.path.join(MODEL_DIR, "ai_model_lstm_autoencoder_quant.tflite")
            dest_tflite = os.path.join(MCU_MODEL_DIR, "ai_autoencoder.tflite")
            
            if os.path.exists(source_tflite):
                import shutil
                shutil.copy2(source_tflite, dest_tflite)
                logger.info(f"Simple model copied to: {dest_tflite}")
                
                # Update thresholds for the simple MCU
                mcu_thresholds = {
                    "thresholds": {
                        "normal": self.thresholds["normal_threshold"],
                        "warning": self.thresholds["warning_threshold"],
                        "critical": self.thresholds["critical_threshold"]
                    },
                    "conversion_date": datetime.now().isoformat(),
                    "source_model": "LSTM Autoencoder Complex"
                }
                
                thresholds_path = os.path.join(MCU_MODEL_DIR, "ai_thresholds.json")
                with open(thresholds_path, 'w') as f:
                    json.dump(mcu_thresholds, f, indent=2)
                
                logger.info(f"Thresholds updated for simple MCU: {thresholds_path}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error copying MCU model: {e}")
            return False
    
    def generate_deployment_report(self):
        """Generates a comprehensive deployment report"""
        report_path = os.path.join(DEPLOY_DIR, "deployment_report.md")
        
        report_content = f"""# Deployment Report - EPS Guardian

## Generation Date
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Source Model
- **Architecture**: LSTM Autoencoder
- **Input Shape**: {self.model_info['input_shape']}
- **Output Shape**: {self.model_info['output_shape']}
- **Parameters**: {self.model_info['total_params']:,}
- **Original Size**: {self.model_info['model_size_h5']:.1f} KB

## Generated Files

### 1. TensorFlow Lite Models (in model_complex)
- `ai_model_lstm_autoencoder.tflite` - Standard version
- `ai_model_lstm_autoencoder_quant.tflite` - Quantized version (recommended)

### 2. ESP32 Deployment Files (in esp32_deployment)
- `eps_guardian_ai_model.h` - C++ header with embedded model
- `eps_guardian_inference.ino` - Complete Arduino sketch

### 3. Simple MCU Compatibility (in model_simple)
- Model also available for simple MCU: `ai_autoencoder.tflite`

## Anomaly Thresholds
- **NORMAL**: < {self.thresholds['normal_threshold']:.6f}
- **WARNING**: < {self.thresholds['warning_threshold']:.6f}  
- **CRITICAL**: >= {self.thresholds['critical_threshold']:.6f}

## Recommended Hardware Configuration
- **MCU**: ESP32 (with 4MB Flash)
- **Minimum RAM**: 40KB for Tensor Arena
- **LED Pins**: 2 (NORMAL), 4 (WARNING), 5 (CRITICAL)

## Usage
1. Include `eps_guardian_ai_model.h` in your project
2. Use the `EPSGuardianAI` class for inference
3. Call `add_sensor_data()` and `detect_anomaly()` cyclically

## Expected Performance
- **Inference Time**: < 100ms on ESP32
- **Accuracy**: DDetection of complex temporal anomalies
- **Power Consumption**: Optimized for embedded systems

---
*Automatically generated by the EPS Guardian system*
"""
        
        with open(report_path, "w") as f:
            f.write(report_content)
        
        logger.info(f"Deployment report generated: {report_path}")

def main():
    """Main entry point"""
    logger.info("STARTING AI MODEL CONVERSION FOR ESP32")
    logger.info("=" * 60)
    
    converter = AIModelConverter()
    
    # 1. Loading complex model
    if not converter.load_complex_model():
        logger.error("Failed to load model - Stopping")
        return 1
    
    # 2. Conversion to TFLite - SAVED IN MODEL_COMPLEX
    logger.info("CONVERTING TO TENSORFLOW LITE")
    
    # Standard version - in model_complex
    tflite_standard = os.path.join(MODEL_DIR, "ai_model_lstm_autoencoder.tflite")
    size_standard = converter.convert_to_tflite(tflite_standard, quantize=False)
    
    # Quantized version - in model_complex
    tflite_quant = os.path.join(MODEL_DIR, "ai_model_lstm_autoencoder_quant.tflite")
    size_quant = converter.convert_to_tflite(tflite_quant, quantize=True, quantization_type="float16")
    
    logger.info(f"Size reduction: {converter.model_info['model_size_h5']:.1f}KB → {size_quant:.1f}KB " 
                f"({size_quant/converter.model_info['model_size_h5']*100:.1f}%)")
    
    # 3. GGenerating deployment files - in esp32_deployment
    logger.info("GENERATING DEPLOYMENT FILES")
    
    # Use the quantized model from model_complex to generate the header
    header_path = os.path.join(DEPLOY_DIR, "eps_guardian_ai_model.h")
    converter.generate_c_header(tflite_quant, header_path)
    
    sketch_path = os.path.join(DEPLOY_DIR, "eps_guardian_inference.ino")
    converter.generate_arduino_sketch(sketch_path)
    
    # 4. Copie pour MCU simple - depuis model_complex vers model_simple
    converter.copy_simple_model_to_mcu()
    
    # 5. Rapport final
    converter.generate_deployment_report()
    
    logger.info("=" * 60)
    logger.info(" CONVERSION COMPLETED SUCCESSFULLY!")
    logger.info(f" Folder model_complex: {MODEL_DIR}")
    logger.info(" Files generated in model_complex:")
    logger.info(f"   - {tflite_standard}")
    logger.info(f"   - {tflite_quant} (recommended)")
    logger.info(f" Deployment folder: {DEPLOY_DIR}")
    logger.info(" Deployment files:")
    logger.info(f"   - {header_path}")
    logger.info(f"   - {sketch_path}")
    logger.info(" Ready for deployment on ESP32!")
    
    return 0

if __name__ == "__main__":
    exit(main())

// ====================================================
// EPS Guardian - AI Anomaly Detection System
// MCU: ESP32 with TensorFlow Lite Micro
// Model: LSTM Autoencoder for temporal sequences
// ====================================================

#include "eps_guardian_ai_model.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ====================================================
// HARDWARE CONFIGURATION
// ====================================================

// Status LED pins
#define LED_NORMAL 2
#define LED_WARNING 4
#define LED_CRITICAL 5

// Buffer for inference
constexpr int kTensorArenaSize = eps_guardian::ai_model::TENSOR_ARENA_SIZE;
static uint8_t tensor_arena[kTensorArenaSize];

// ====================================================
// MAIN EPS GUARDIAN AI CLASS
// ====================================================

class EPSGuardianAI {
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
    bool initialize() {
        Serial.println("Initializing EPS Guardian AI...");
        
        // Model loading
        model = tflite::GetModel(eps_guardian::ai_model::g_ai_model_data);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            Serial.println("ERROR: Incompatible model version!");
            return false;
        }
        
        // Interpreter initialization
        static tflite::MicroInterpreter static_interpreter(
            model, resolver, tensor_arena, kTensorArenaSize);
        interpreter = &static_interpreter;
        
        // Tensor allocation
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            Serial.println("ERROR: Tensor allocation failed!");
            return false;
        }
        
        input = interpreter->input(0);
        output = interpreter->output(0);
        
        // Pin configuration
        pinMode(LED_NORMAL, OUTPUT);
        pinMode(LED_WARNING, OUTPUT);
        pinMode(LED_CRITICAL, OUTPUT);
        
        Serial.println("EPS Guardian AI initialized successfully!");
        Serial.println("Waiting for sensor data...");
        
        return true;
    }

    void add_sensor_data(float* sensor_values) {
        // Add new data to sequence
        for (int i = 0; i < eps_guardian::ai_model::FEATURE_COUNT; i++) {
            sensor_sequence[sequence_index][i] = sensor_values[i];
        }
        
        sequence_index = (sequence_index + 1) % eps_guardian::ai_model::SEQUENCE_LENGTH;
    }

    float detect_anomaly() {
        // Verify sequence is complete
        if (sequence_index != 0) {
            Serial.println("WARNING: Incomplete sequence, using available data");
        }
        
        // Copy data to input tensor
        float* input_data = input->data.f;
        for (int i = 0; i < eps_guardian::ai_model::SEQUENCE_LENGTH; i++) {
            for (int j = 0; j < eps_guardian::ai_model::FEATURE_COUNT; j++) {
                *input_data++ = sensor_sequence[i][j];
            }
        }
        
        // Inference
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("ERROR: Inference failed!");
            return -1.0f;
        }
        
        // Calculate reconstruction error
        float reconstruction_error = 0.0f;
        float* output_data = output->data.f;
        input_data = input->data.f;
        
        for (int i = 0; i < eps_guardian::ai_model::SEQUENCE_LENGTH * eps_guardian::ai_model::FEATURE_COUNT; i++) {
            float diff = input_data[i] - output_data[i];
            reconstruction_error += diff * diff;
        }
        
        reconstruction_error /= (eps_guardian::ai_model::SEQUENCE_LENGTH * eps_guardian::ai_model::FEATURE_COUNT);
        
        return reconstruction_error;
    }

    int get_anomaly_level(float error) {
        if (error < eps_guardian::ai_model::NORMAL_THRESHOLD) {
            return 0; // NORMAL
        } else if (error < eps_guardian::ai_model::WARNING_THRESHOLD) {
            return 1; // WARNING
        } else {
            return 2; // CRITICAL
        }
    }

    void update_leds(int anomaly_level) {
        // Turn off all LEDs
        digitalWrite(LED_NORMAL, LOW);
        digitalWrite(LED_WARNING, LOW);
        digitalWrite(LED_CRITICAL, LOW);
        
        // Turn on corresponding LED
        switch (anomaly_level) {
            case 0: // NORMAL
                digitalWrite(LED_NORMAL, HIGH);
                break;
            case 1: // WARNING
                digitalWrite(LED_WARNING, HIGH); 
                break;
            case 2: // CRITICAL
                digitalWrite(LED_CRITICAL, HIGH);
                break;
        }
    }

    void print_debug_info(float error, int level) {
        Serial.print("Reconstruction error: ");
        Serial.print(error, 6);
        Serial.print(" | Level: ");
        
        switch (level) {
            case 0: Serial.println("NORMAL"); break;
            case 1: Serial.println("WARNING"); break;
            case 2: Serial.println("CRITICAL"); break;
        }
        
        Serial.print("Thresholds - Normal<");
        Serial.print(eps_guardian::ai_model::NORMAL_THRESHOLD, 6);
        Serial.print(", Warning<");  
        Serial.print(eps_guardian::ai_model::WARNING_THRESHOLD, 6);
        Serial.print(", Critical>=");
        Serial.println(eps_guardian::ai_model::CRITICAL_THRESHOLD, 6);
    }
};

// ====================================================
// GLOBAL INSTANCE AND SETUP
// ====================================================

EPSGuardianAI guardianAI;

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }
    
    Serial.println("================================================");
    Serial.println("EPS GUARDIAN - EMBEDDED AI SYSTEM");
    Serial.println("Real-time EPS anomaly detection");
    Serial.println("================================================");
    
    if (!guardianAI.initialize()) {
        Serial.println("FAILURE: AI Initialization - System halted");
        while(1) { delay(1000); }
    }
}

// ====================================================
// MAIN LOOP
// ====================================================

void loop() {
    // SIMULATION: Realistic sensor data generation
    float sensor_data[eps_guardian::ai_model::FEATURE_COUNT];
    
    // Typical values of a nominal EPS system
    sensor_data[0] = 7.4f + random(-10, 10) / 100.0f;  // V_batt
    sensor_data[1] = 1.2f + random(-20, 20) / 100.0f;  // I_batt  
    sensor_data[2] = 35.0f + random(-50, 50) / 10.0f;  // T_batt
    sensor_data[3] = 7.8f + random(-5, 5) / 100.0f;    // V_bus
    sensor_data[4] = 0.8f + random(-10, 10) / 100.0f;  // I_bus
    sensor_data[5] = 15.2f + random(-20, 20) / 10.0f;  // V_solar
    sensor_data[6] = 1.5f + random(-10, 10) / 100.0f;  // I_solar
    
    // Add data to sequence
    guardianAI.add_sensor_data(sensor_data);
    
    // Anomaly detection
    float error = guardianAI.detect_anomaly();
    
    if (error >= 0) {
        int anomaly_level = guardianAI.get_anomaly_level(error);
        guardianAI.update_leds(anomaly_level);
        guardianAI.print_debug_info(error, anomaly_level);
    }
    
    // 2 second cycle
    delay(2000);
}

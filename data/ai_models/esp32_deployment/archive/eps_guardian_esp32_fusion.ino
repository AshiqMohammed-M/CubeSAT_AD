// ======================================
// EPS GUARDIAN ESP32
// Deterministic Rules + Autoencoder AI
// Hybrid System
// ======================================

#include "eps_guardian_ai_model.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ======================
// MEMORY CONFIGURATION
// ======================
constexpr int kTensorArenaSize = 12 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// ======================
// DETERMINISTIC RULES THRESHOLDS (R1-R7)
// ======================
constexpr float RULE_TEMP_CRITICAL = 60.0f;      // R1 - Battery overheating
constexpr float RULE_CURRENT_CRITICAL = 3.0f;    // R2 - Current overload
constexpr float RULE_VOLTAGE_CRITICAL = 3.2f;    // R3 - Deep discharge
constexpr float RULE_RATIO_CRITICAL = 0.7f;      // R4 - DC/DC ratio
constexpr float RULE_SENSOR_FAULT = 120.0f;      // R6 - Sensor fault
constexpr float RULE_OSCILLATION_V = 0.5f;       // R5 - Voltage oscillation
constexpr float RULE_OSCILLATION_T = 5.0f;       // R5 - Temperature oscillation

// ======================
// ESP32 PINS
// ======================
const int LED_NORMAL = 2;
const int LED_WARNING = 4;
const int LED_CRITICAL = 5;
const int MOSFET_PIN = 12;
const int BUZZER_PIN = 13;

// ======================
// MAIN CLASS
// ======================
class EPSGuardianFusion {
private:
    // AI Components
    const tflite::Model* ai_model;
    tflite::MicroInterpreter* ai_interpreter;
    TfLiteTensor* ai_input;
    TfLiteTensor* ai_output;
    tflite::AllOpsResolver resolver;

    // System state and history
    struct SystemState {
        float v_batt = 7.4f;
        float i_batt = 1.2f;
        float t_batt = 35.0f;
        float v_bus = 7.3f;
        float i_bus = 0.8f;
        float v_solar = 14.5f;
        float i_solar = 1.1f;
        bool mosfet_enabled = true;
        bool charge_enabled = true;
        int anomaly_level = 0; // 0=Normal, 1=Warning, 2=Critical
        unsigned long last_cycle_time = 0;
    } state;

    // History for delta calculations
    float prev_v_batt = 7.4f;
    float prev_t_batt = 35.0f;

public:
    // ======================
    // AI INITIALIZATION
    // ======================
    bool initializeAI() {
        Serial.println("Initializing AI model...");
        
        // Load the model
        ai_model = tflite::GetModel(eps_guardian::ai_model::g_ai_model_data);
        if (ai_model->version() != TFLITE_SCHEMA_VERSION) {
            Serial.println("Incompatible model version");
            return false;
        }

        // Create the interpreter
        static tflite::MicroInterpreter static_interpreter(
            ai_model, resolver, tensor_arena, kTensorArenaSize);
        ai_interpreter = &static_interpreter;

        // Allocate memory
        if (ai_interpreter->AllocateTensors() != kTfLiteOk) {
            Serial.println("AI memory allocation error");
            return false;
        }

        ai_input = ai_interpreter->input(0);
        ai_output = ai_interpreter->output(0);
        
        Serial.println("AI model initialized");
        Serial.print("Model size: ");
        Serial.print(eps_guardian::ai_model::g_ai_model_size);
        Serial.println(" bytes");
        
        return true;
    }

    // ======================
    // DETERMINISTIC RULES (R1-R7)
    // ======================
    int executeSafetyRules() {
        // R1: Battery overheating -> MOSFET cutoff
        if (state.t_batt > RULE_TEMP_CRITICAL) {
            Serial.println("R1: Battery overheating detected");
            state.mosfet_enabled = false;
            return 2; // CRITICAL
        }
        
        // R2: Current overload -> Charge limitation
        if (abs(state.i_batt) > RULE_CURRENT_CRITICAL) {
            Serial.println("R2: Current overload detected");
            state.charge_enabled = false;
            return 2; // CRITICAL
        }
        
        // R3: Deep discharge -> Battery isolation
        if (state.v_batt < RULE_VOLTAGE_CRITICAL && state.i_batt < 0) {
            Serial.println("R3: Deep discharge detected");
            state.mosfet_enabled = false;
            return 2; // CRITICAL
        }
        
        // R4: Abnormal DC/DC ratio -> Charge reduction
        float ratio = (state.v_solar > 0.1f) ? state.v_bus / state.v_solar : 0.0f;
        if (ratio < RULE_RATIO_CRITICAL && ratio > 0.01f) {
            Serial.println("R4: Abnormal DC/DC ratio");
            state.charge_enabled = false;
            return 1; // WARNING
        }
        
        // R5: Bus oscillation -> Increased logging
        float delta_v = abs(state.v_batt - prev_v_batt);
        float delta_t = abs(state.t_batt - prev_t_batt);
        
        if (delta_v > RULE_OSCILLATION_V || delta_t > RULE_OSCILLATION_T) {
            Serial.println("R5: Oscillation detected");
            return 1; // WARNING
        }
        
        // R6: Sensor fault -> Safe mode
        if (state.t_batt > RULE_SENSOR_FAULT || state.v_batt > 20.0f) {
            Serial.println("R6: Suspected sensor fault");
            return 1; // WARNING
        }
        
        // R7: Normal state -> Green LED
        return 0; // NORMAL
    }

    // ======================
    // AUTOENCODER AI DETECTION
    // ======================
    float detectAIAnomaly() {
        // Calculate deltas for temporal features
        float delta_v_batt = state.v_batt - prev_v_batt;
        float delta_t_batt = state.t_batt - prev_t_batt;
        
        // Prepare 11 features exactly like training
        float features[11] = {
            state.v_batt, state.i_batt, state.t_batt,
            state.v_bus, state.i_bus, 
            state.v_solar, state.i_solar,
            state.v_batt * state.i_batt,                    // P_batt
            (state.v_solar > 0.1f) ? state.v_bus / state.v_solar : 0.0f, // converter_ratio
            delta_v_batt,                                   // delta_V_batt
            delta_t_batt                                    // delta_T_batt
        };
        
        // Copy into input tensor
        for (int i = 0; i < 11; i++) {
            ai_input->data.f[i] = features[i];
        }
        
        // Execute inference
        if (ai_interpreter->Invoke() != kTfLiteOk) {
            Serial.println("AI inference error");
            return -1.0f;
        }
        
        // Calculate reconstruction error (MSE)
        float reconstruction_error = 0.0f;
        for (int i = 0; i < 11; i++) {
            float diff = ai_input->data.f[i] - ai_output->data.f[i];
            reconstruction_error += diff * diff;
        }
        reconstruction_error /= 11.0f;
        
        return reconstruction_error;
    }

    int getAIAnomalyLevel(float error) {
        if (error < 0) {
            return -1; // ERROR
        } else if (error < eps_guardian::ai_model::NORMAL_THRESHOLD) {
            return 0;  // NORMAL
        } else if (error < eps_guardian::ai_model::WARNING_THRESHOLD) {
            return 1;  // WARNING
        } else {
            return 2;  // CRITICAL
        }
    }

    // ======================
    // HYBRID DECISION
    // ======================
    int hybridDecision() {
        // 1. EXECUTE DETERMINISTIC RULES (High priority)
        int rule_level = executeSafetyRules();
        if (rule_level == 2) { // CRITICAL by rules -> Immediate action
            Serial.println("Decision: Critical rules -> Immediate action");
            takeEmergencyAction();
            return 2;
        }
        
        // 2. AI DETECTION (Complex patterns)
        float ai_error = detectAIAnomaly();
        int ai_level = getAIAnomalyLevel(ai_error);
        
        if (ai_level == -1) {
            Serial.println("AI in error, reduced confidence");
            ai_level = 0; // Fallback to normal
        }
        
        // 3. FINAL DECISION - Worst of the two
        int final_level = max(rule_level, ai_level);
        state.anomaly_level = final_level;
        
        // Update history
        prev_v_batt = state.v_batt;
        prev_t_batt = state.t_batt;
        
        return final_level;
    }

    // ======================
    // PHYSICAL ACTIONS
    // ======================
    void takeEmergencyAction() {
        // Immediate critical actions
        digitalWrite(LED_CRITICAL, HIGH);
        digitalWrite(LED_WARNING, LOW);
        digitalWrite(LED_NORMAL, LOW);
        
        // Safety cutoff
        state.mosfet_enabled = false;
        state.charge_enabled = false;
        digitalWrite(MOSFET_PIN, LOW);
        
        // Sound alarm
        tone(BUZZER_PIN, 1000, 1000);
        
        Serial.println("URGENCY ACTION: Safety cutoff activated");
    }

    void takePreventiveAction(int level) {
        // Reset all LEDs
        digitalWrite(LED_CRITICAL, LOW);
        digitalWrite(LED_WARNING, LOW);
        digitalWrite(LED_NORMAL, LOW);
        
        switch(level) {
            case 0: // NORMAL
                digitalWrite(LED_NORMAL, HIGH);
                state.mosfet_enabled = true;
                state.charge_enabled = true;
                digitalWrite(MOSFET_PIN, HIGH);
                noTone(BUZZER_PIN);
                break;
                
            case 1: // WARNING
                digitalWrite(LED_WARNING, HIGH);
                state.charge_enabled = false; // Charge reduction
                tone(BUZZER_PIN, 500, 200); // Short beep
                Serial.println("ACTION: Charge reduction active");
                break;
                
            case 2: // CRITICAL  
                takeEmergencyAction();
                break;
        }
    }

    // ======================
    // SENSOR UPDATE (Simulation/Real)
    // ======================
    void updateSensorReadings() {
        // REPLACE WITH REAL SENSOR READINGS
        // Simulation: slight random variations + test scenarios
        
        static int cycle_count = 0;
        cycle_count++;
        
        // Test scenario: every 20 cycles, simulate an anomaly
        if (cycle_count % 20 == 0) {
            // Simulate overheating
            state.t_batt = 65.0f;
            state.i_batt = 3.5f;
        } else if (cycle_count % 15 == 0) {
            // Simulate subtle anomaly (detectable by AI)
            state.v_batt = 5.8f;
            state.v_solar = 25.0f;
        } else {
            // Normal behavior with noise
            state.v_batt += random(-10, 10) * 0.001f;
            state.i_batt += random(-5, 5) * 0.01f;
            state.t_batt += random(-3, 3) * 0.05f;
            state.v_bus += random(-10, 10) * 0.001f;
            state.i_bus += random(-2, 2) * 0.01f;
            state.v_solar += random(-20, 20) * 0.002f;
            state.i_solar += random(-10, 10) * 0.01f;
        }
        
        // Realistic physical limits
        state.v_batt = constrain(state.v_batt, 2.5f, 8.5f);
        state.i_batt = constrain(state.i_batt, -4.0f, 4.0f);
        state.t_batt = constrain(state.t_batt, -10.0f, 80.0f);
        state.v_bus = constrain(state.v_bus, 5.0f, 9.0f);
        state.v_solar = constrain(state.v_solar, 0.0f, 30.0f);
    }

    // ======================
    // STATUS DISPLAY
    // ======================
    void printSystemStatus() {
        Serial.print("Battery: ");
        Serial.print(state.v_batt, 2);
        Serial.print("V, ");
        Serial.print(state.i_batt, 2);
        Serial.print("A, ");
        Serial.print(state.t_batt, 1);
        Serial.println("C");
        
        Serial.print("Bus: ");
        Serial.print(state.v_bus, 2);
        Serial.print("V, ");
        Serial.print(state.i_bus, 2);
        Serial.println("A");
        
        Serial.print("Solar: ");
        Serial.print(state.v_solar, 2);
        Serial.print("V, ");
        Serial.print(state.i_solar, 2);
        Serial.println("A");
        
        Serial.print("MOSFET: ");
        Serial.print(state.mosfet_enabled ? "ON" : "OFF");
        Serial.print(" | Charge: ");
        Serial.println(state.charge_enabled ? "ON" : "OFF");
    }
};

// ======================
// GLOBAL INSTANCE
// ======================
EPSGuardianFusion guardian;

// ======================
// SETUP
// ======================
void setup() {
    Serial.begin(115200);
    delay(1000); // Wait for serial connection
    
    Serial.println("\n\n EPS GUARDIAN FUSION - Startup");
    Serial.println("==========================================");
    
    // Pin configuration
    pinMode(LED_NORMAL, OUTPUT);
    pinMode(LED_WARNING, OUTPUT);
    pinMode(LED_CRITICAL, OUTPUT);
    pinMode(MOSFET_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    
    // Startup sequence
    digitalWrite(LED_NORMAL, HIGH);
    delay(300);
    digitalWrite(LED_WARNING, HIGH);
    delay(300);
    digitalWrite(LED_CRITICAL, HIGH);
    delay(300);
    digitalWrite(LED_NORMAL, LOW);
    digitalWrite(LED_WARNING, LOW);
    digitalWrite(LED_CRITICAL, LOW);
    
    // AI Initialization
    if (guardian.initializeAI()) {
        Serial.println("Hybrid system initialized");
    } else {
        Serial.println("Initialization failed - Safe mode");
        // Continue with rules only
    }
    
    Serial.println("Ready for real-time monitoring");
    Serial.println("==========================================");
}

// ======================
// MAIN LOOP
// ======================
void loop() {
    Serial.println("\n--- Monitoring cycle ---");
    
    // 1. Update sensor readings
    guardian.updateSensorReadings();
    
    // 2. Hybrid decision (Rules + AI)
    int anomaly_level = guardian.hybridDecision();
    
    // 3. Actions based on level
    guardian.takePreventiveAction(anomaly_level);
    
    // 4. Status display
    Serial.print("Final decision: ");
    switch(anomaly_level) {
        case 0: Serial.println("NORMAL"); break;
        case 1: Serial.println("WARNING"); break;
        case 2: Serial.println("CRITICAL"); break;
        default: Serial.println("UNKNOWN");
    }
    
    guardian.printSystemStatus();
    
    delay(2000); // 2 second cycle
}

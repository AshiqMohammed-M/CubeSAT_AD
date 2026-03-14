/*
 * Phase 8a firmware preparation (software-only) for ESP32-S3 DevKit-C.
 *
 * Optimization audit summary:
 * - Model size (profiler): 186.73 KB
 * - Estimated tensor memory (profiler): 81.40 KB
 * - Optimization applied: NO (both metrics are within limits)
 *
 * Threshold source (do not hardcode in source logic):
 * - src/config/eps_config.yaml -> ai_thresholds.simple
 *   normal   = 0.6091226281233376
 *   warning  = 0.8377888951216337
 *   critical = 1.0664551621199299
 */

#include <Arduino.h>
#include <Wire.h>
#include <MPU6050.h>
#include <TensorFlowLite_ESP32.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"

#if defined(__has_include)
#if __has_include(<esp32-hal-rgb-led.h>)
#include <esp32-hal-rgb-led.h>
#define HAS_RGB_LED 1
#else
#define HAS_RGB_LED 0
#endif
#else
#define HAS_RGB_LED 0
#endif

// -------------------- Includes and defines --------------------
constexpr uint32_t UART_BAUD_RATE = 115200;
constexpr uint8_t I2C_SDA_PIN = 21;
constexpr uint8_t I2C_SCL_PIN = 22;

constexpr size_t WINDOW_SIZE = 30;
constexpr size_t FEATURE_COUNT = 14;
constexpr size_t UART_FLOAT_COUNT = 15;
constexpr size_t MAX_FRAME_BYTES = 512;
constexpr uint32_t UART_TIMEOUT_MS = 5000;

// Phase 8c guard: keep false in Phase 8a software prep.
#define LIVE_IMU false

constexpr float THRESH_NORMAL = 0.6091226281f;
constexpr float THRESH_WARNING = 0.8377888951f;
constexpr float THRESH_CRITICAL = 1.0664551621f;

constexpr uint8_t FALLBACK_LED_PIN = 2;
// Input buffer footprint for [30,14] float32 is 30*14*4 = 1680 bytes.
// Arena is set below 350KB target while leaving room for model intermediates.
constexpr size_t TENSOR_ARENA_SIZE = 300 * 1024;

// -------------------- Global state --------------------
const tflite::Model *g_model = nullptr;
tflite::MicroInterpreter *g_interpreter = nullptr;
TfLiteTensor *g_input_tensor = nullptr;
TfLiteTensor *g_output_tensor = nullptr;
tflite::AllOpsResolver g_resolver;

alignas(16) static uint8_t g_tensor_arena[TENSOR_ARENA_SIZE];
static bool g_tflm_ready = false;

static float g_window[WINDOW_SIZE][FEATURE_COUNT];
static size_t g_window_index = 0;
static size_t g_window_count = 0;
static bool g_window_ready = false;

static uint32_t g_sample_index = 0;

MPU6050 g_imu;
bool g_imu_available = false;

enum Severity : uint8_t {
  SEVERITY_NORMAL = 0,
  SEVERITY_WARNING = 1,
  SEVERITY_CRITICAL = 2,
};

struct ClassificationResult {
  Severity severity;
  float reconstruction_error;
  uint32_t sample_index;
};

static ClassificationResult g_last_result = {
    SEVERITY_NORMAL,
    0.0f,
    0,
};

static bool g_led_on = false;
static unsigned long g_last_led_toggle_ms = 0;

// Packet fields (12 EPS + 3 attitude):
// [0] V_batt, [1] I_batt, [2] T_batt, [3] V_bus, [4] I_bus,
// [5] V_solar, [6] I_solar, [7] SOC, [8] T_eps,
// [9] P_batt, [10] P_solar, [11] P_bus,
// [12] attitude_error_deg, [13] gyro_magnitude, [14] accel_magnitude

// -------------------- COBS encode/decode --------------------
// Keep framing compatible with Python router framing in src/output/output_router.py
size_t encode_cobs(const uint8_t *input, size_t length, uint8_t *output, size_t output_capacity) {
  if (output_capacity == 0) {
    return 0;
  }

  size_t read_index = 0;
  size_t write_index = 1;
  size_t code_index = 0;
  uint8_t code = 1;

  while (read_index < length) {
    if (input[read_index] == 0) {
      if (code_index >= output_capacity) {
        return 0;
      }
      output[code_index] = code;
      code = 1;
      code_index = write_index;
      if (write_index >= output_capacity) {
        return 0;
      }
      write_index++;
      read_index++;
      continue;
    }

    if (write_index >= output_capacity) {
      return 0;
    }
    output[write_index++] = input[read_index++];
    code++;

    if (code == 0xFF) {
      if (code_index >= output_capacity) {
        return 0;
      }
      output[code_index] = code;
      code = 1;
      code_index = write_index;
      if (write_index >= output_capacity) {
        return 0;
      }
      write_index++;
    }
  }

  if (code_index >= output_capacity) {
    return 0;
  }
  output[code_index] = code;
  return write_index;
}

size_t decode_cobs(const uint8_t *input, size_t length, uint8_t *output, size_t output_capacity) {
  size_t read_index = 0;
  size_t write_index = 0;

  while (read_index < length) {
    uint8_t code = input[read_index];
    if (code == 0) {
      return 0;
    }

    if ((read_index + code) > (length + 1)) {
      return 0;
    }

    read_index++;

    for (uint8_t i = 1; i < code; ++i) {
      if (read_index >= length || write_index >= output_capacity) {
        return 0;
      }
      output[write_index++] = input[read_index++];
    }

    if (code != 0xFF && read_index < length) {
      if (write_index >= output_capacity) {
        return 0;
      }
      output[write_index++] = 0;
    }
  }

  return write_index;
}

// COBS symmetry check example:
// uint8_t raw[3] = {0x11, 0x00, 0x22};
// uint8_t enc[8];
// uint8_t dec[8];
// size_t enc_len = encode_cobs(raw, 3, enc, sizeof(enc));
// size_t dec_len = decode_cobs(enc, enc_len, dec, sizeof(dec));
// Assert: dec_len == 3 && memcmp(raw, dec, 3) == 0

// -------------------- Utility helpers --------------------
void set_led_rgb(uint8_t r, uint8_t g, uint8_t b) {
#if HAS_RGB_LED
  neopixelWrite(RGB_BUILTIN, r, g, b);
#else
  (void)r;
  (void)g;
  (void)b;
#endif
}

void apply_led_output(bool enabled, Severity severity) {
#if HAS_RGB_LED
  if (!enabled || severity == SEVERITY_NORMAL) {
    set_led_rgb(0, 0, 0);
    return;
  }

  if (severity == SEVERITY_WARNING) {
    // Amber
    set_led_rgb(255, 120, 0);
  } else {
    // Critical red
    set_led_rgb(255, 0, 0);
  }
#else
  digitalWrite(FALLBACK_LED_PIN, enabled ? HIGH : LOW);
#endif
}

void update_led_pattern(Severity severity) {
  if (severity == SEVERITY_NORMAL) {
    g_led_on = false;
    apply_led_output(false, severity);
    return;
  }

  const uint32_t interval = (severity == SEVERITY_WARNING) ? 500 : 100;
  const unsigned long now = millis();
  if ((now - g_last_led_toggle_ms) >= interval) {
    g_last_led_toggle_ms = now;
    g_led_on = !g_led_on;
    apply_led_output(g_led_on, severity);
  }
}

const char *severity_to_text(Severity severity) {
  switch (severity) {
    case SEVERITY_NORMAL:
      return "NORMAL";
    case SEVERITY_WARNING:
      return "WARNING";
    case SEVERITY_CRITICAL:
      return "CRITICAL";
    default:
      return "WARNING";
  }
}

Severity classify_error(float error_value) {
  if (error_value < THRESH_NORMAL) {
    return SEVERITY_NORMAL;
  }
  if (error_value < THRESH_WARNING) {
    return SEVERITY_WARNING;
  }
  if (error_value >= THRESH_CRITICAL) {
    return SEVERITY_CRITICAL;
  }
  // Keep warning for the gap [warning, critical).
  return SEVERITY_WARNING;
}

bool read_uart_frame(uint8_t *encoded, size_t *encoded_len, uint32_t timeout_ms) {
  size_t index = 0;
  const unsigned long start_ms = millis();

  while ((millis() - start_ms) < timeout_ms) {
    while (Serial.available() > 0) {
      int value = Serial.read();
      if (value < 0) {
        break;
      }

      const uint8_t byte_value = static_cast<uint8_t>(value);
      if (byte_value == 0x00) {
        if (index == 0) {
          continue;
        }
        *encoded_len = index;
        return true;
      }

      if (index < MAX_FRAME_BYTES) {
        encoded[index++] = byte_value;
      } else {
        // Overflow protection: drop this frame.
        index = 0;
      }
    }
    delay(1);
  }

  return false;
}

bool json_extract_float(const char *json_payload, const char *key, float *out_value) {
  char pattern[48];
  const int pattern_len = snprintf(pattern, sizeof(pattern), "\"%s\":", key);
  if (pattern_len <= 0 || pattern_len >= static_cast<int>(sizeof(pattern))) {
    return false;
  }

  const char *position = strstr(json_payload, pattern);
  if (position == nullptr) {
    return false;
  }

  position += pattern_len;
  while (*position == ' ' || *position == '\t') {
    position++;
  }

  char *end_ptr = nullptr;
  const float value = strtof(position, &end_ptr);
  if (end_ptr == position) {
    return false;
  }

  *out_value = value;
  return true;
}

bool parse_json_payload(const uint8_t *payload, size_t payload_len, float fields[UART_FLOAT_COUNT]) {
  if (payload_len == 0 || payload_len >= MAX_FRAME_BYTES) {
    return false;
  }

  char json_buffer[MAX_FRAME_BYTES];
  memcpy(json_buffer, payload, payload_len);
  json_buffer[payload_len] = '\0';

  bool ok = true;
  ok &= json_extract_float(json_buffer, "V_batt", &fields[0]);
  ok &= json_extract_float(json_buffer, "I_batt", &fields[1]);
  ok &= json_extract_float(json_buffer, "T_batt", &fields[2]);
  ok &= json_extract_float(json_buffer, "V_bus", &fields[3]);
  ok &= json_extract_float(json_buffer, "I_bus", &fields[4]);
  ok &= json_extract_float(json_buffer, "V_solar", &fields[5]);
  ok &= json_extract_float(json_buffer, "I_solar", &fields[6]);
  ok &= json_extract_float(json_buffer, "SOC", &fields[7]);
  ok &= json_extract_float(json_buffer, "T_eps", &fields[8]);
  ok &= json_extract_float(json_buffer, "P_batt", &fields[9]);
  ok &= json_extract_float(json_buffer, "P_solar", &fields[10]);
  ok &= json_extract_float(json_buffer, "P_bus", &fields[11]);
  ok &= json_extract_float(json_buffer, "attitude_error_deg", &fields[12]);
  ok &= json_extract_float(json_buffer, "gyro_magnitude", &fields[13]);
  ok &= json_extract_float(json_buffer, "accel_magnitude", &fields[14]);

  // If power values are not present in JSON, derive them.
  if (!json_extract_float(json_buffer, "P_batt", &fields[9])) {
    fields[9] = fields[0] * fields[1];
  }
  if (!json_extract_float(json_buffer, "P_solar", &fields[10])) {
    fields[10] = fields[5] * fields[6];
  }
  if (!json_extract_float(json_buffer, "P_bus", &fields[11])) {
    fields[11] = fields[3] * fields[4];
  }

  return ok;
}

bool parse_payload_to_fields(const uint8_t *payload, size_t payload_len, float fields[UART_FLOAT_COUNT]) {
  const size_t expected_binary_size = UART_FLOAT_COUNT * sizeof(float);

  if (payload_len == expected_binary_size) {
    memcpy(fields, payload, expected_binary_size);
    return true;
  }

  if (payload_len > 0 && payload[0] == '{') {
    return parse_json_payload(payload, payload_len, fields);
  }

  return false;
}

void build_feature_vector(const float uart_fields[UART_FLOAT_COUNT], float features[FEATURE_COUNT]) {
  // Keep 14-model features by dropping P_bus while preserving 3 attitude values.
  features[0] = uart_fields[0];   // V_batt
  features[1] = uart_fields[1];   // I_batt
  features[2] = uart_fields[2];   // T_batt
  features[3] = uart_fields[3];   // V_bus
  features[4] = uart_fields[4];   // I_bus
  features[5] = uart_fields[5];   // V_solar
  features[6] = uart_fields[6];   // I_solar
  features[7] = uart_fields[7];   // SOC
  features[8] = uart_fields[8];   // T_eps
  features[9] = uart_fields[9];   // P_batt
  features[10] = uart_fields[10]; // P_solar
  features[11] = uart_fields[12]; // attitude_error_deg
  features[12] = uart_fields[13]; // gyro_magnitude
  features[13] = uart_fields[14]; // accel_magnitude
}

bool read_live_imu(float *attitude_error_deg, float *gyro_magnitude, float *accel_magnitude) {
  if (!g_imu_available) {
    return false;
  }

  int16_t ax = 0, ay = 0, az = 0;
  int16_t gx = 0, gy = 0, gz = 0;
  g_imu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  if (ax == 0 && ay == 0 && az == 0 && gx == 0 && gy == 0 && gz == 0) {
    return false;
  }

  const float ax_g = static_cast<float>(ax) / 16384.0f;
  const float ay_g = static_cast<float>(ay) / 16384.0f;
  const float az_g = static_cast<float>(az) / 16384.0f;

  const float gx_rad_s = (static_cast<float>(gx) / 131.0f) * (PI / 180.0f);
  const float gy_rad_s = (static_cast<float>(gy) / 131.0f) * (PI / 180.0f);
  const float gz_rad_s = (static_cast<float>(gz) / 131.0f) * (PI / 180.0f);

  const float roll_deg = atan2f(ay_g, az_g) * 57.2957795f;
  const float pitch_deg = atan2f(-ax_g, sqrtf(ay_g * ay_g + az_g * az_g)) * 57.2957795f;

  *attitude_error_deg = sqrtf(roll_deg * roll_deg + pitch_deg * pitch_deg);
  *gyro_magnitude = sqrtf(gx_rad_s * gx_rad_s + gy_rad_s * gy_rad_s + gz_rad_s * gz_rad_s);
  *accel_magnitude = sqrtf(ax_g * ax_g + ay_g * ay_g + az_g * az_g) * 9.80665f;

  return true;
}

void push_features_to_window(const float features[FEATURE_COUNT]) {
  for (size_t i = 0; i < FEATURE_COUNT; ++i) {
    g_window[g_window_index][i] = features[i];
  }

  g_window_index = (g_window_index + 1U) % WINDOW_SIZE;
  if (g_window_count < WINDOW_SIZE) {
    g_window_count++;
  }
  g_window_ready = (g_window_count >= WINDOW_SIZE);
}

bool flatten_window_to_input(float *flat_input) {
  if (!g_window_ready) {
    return false;
  }

  // Oldest sample is at g_window_index when the ring is full.
  for (size_t t = 0; t < WINDOW_SIZE; ++t) {
    const size_t src_idx = (g_window_index + t) % WINDOW_SIZE;
    for (size_t f = 0; f < FEATURE_COUNT; ++f) {
      *flat_input++ = g_window[src_idx][f];
    }
  }

  return true;
}

size_t tensor_element_size(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return sizeof(float);
    case kTfLiteInt8:
      return sizeof(int8_t);
    case kTfLiteUInt8:
      return sizeof(uint8_t);
    default:
      return 0;
  }
}

bool run_inference(float *reconstruction_error) {
  if (!g_tflm_ready || g_input_tensor == nullptr || g_output_tensor == nullptr) {
    return false;
  }

  const size_t expected_elements = WINDOW_SIZE * FEATURE_COUNT;
  float ordered_input[WINDOW_SIZE * FEATURE_COUNT];
  if (!flatten_window_to_input(ordered_input)) {
    return false;
  }

  int model_steps = static_cast<int>(WINDOW_SIZE);
  int model_features = static_cast<int>(FEATURE_COUNT);
  if (g_input_tensor->dims != nullptr && g_input_tensor->dims->size >= 2) {
    model_steps = g_input_tensor->dims->data[g_input_tensor->dims->size - 2];
    model_features = g_input_tensor->dims->data[g_input_tensor->dims->size - 1];
  }
  if (model_steps <= 0 || model_features <= 0) {
    return false;
  }

  const size_t input_elem_size = tensor_element_size(g_input_tensor->type);
  if (input_elem_size == 0) {
    return false;
  }

  const size_t model_input_elements = static_cast<size_t>(model_steps) * static_cast<size_t>(model_features);
  const size_t input_capacity = g_input_tensor->bytes / input_elem_size;
  const size_t write_elements = (model_input_elements < input_capacity) ? model_input_elements : input_capacity;

  if (write_elements == 0) {
    return false;
  }

  const size_t available_steps = static_cast<size_t>(model_steps) < WINDOW_SIZE ? static_cast<size_t>(model_steps) : WINDOW_SIZE;
  const size_t step_offset = WINDOW_SIZE - available_steps;

  size_t write_idx = 0;
  for (size_t t = 0; t < static_cast<size_t>(model_steps) && write_idx < write_elements; ++t) {
    const size_t src_step = (t < step_offset) ? 0 : (t - step_offset);
    const size_t base = src_step * FEATURE_COUNT;
    for (size_t f = 0; f < static_cast<size_t>(model_features) && write_idx < write_elements; ++f) {
      const float value = (f < FEATURE_COUNT) ? ordered_input[base + f] : 0.0f;

      if (g_input_tensor->type == kTfLiteFloat32) {
        g_input_tensor->data.f[write_idx] = value;
      } else if (g_input_tensor->type == kTfLiteInt8) {
        const float scale = g_input_tensor->params.scale;
        const int32_t zero_point = g_input_tensor->params.zero_point;
        if (scale == 0.0f) {
          return false;
        }
        const int32_t q = static_cast<int32_t>(roundf(value / scale)) + zero_point;
        const int8_t q8 = static_cast<int8_t>(q < -128 ? -128 : (q > 127 ? 127 : q));
        g_input_tensor->data.int8[write_idx] = q8;
      } else if (g_input_tensor->type == kTfLiteUInt8) {
        const float scale = g_input_tensor->params.scale;
        const int32_t zero_point = g_input_tensor->params.zero_point;
        if (scale == 0.0f) {
          return false;
        }
        const int32_t q = static_cast<int32_t>(roundf(value / scale)) + zero_point;
        const uint8_t q8 = static_cast<uint8_t>(q < 0 ? 0 : (q > 255 ? 255 : q));
        g_input_tensor->data.uint8[write_idx] = q8;
      } else {
        return false;
      }

      write_idx++;
    }
  }

  const TfLiteStatus invoke_status = g_interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.printf("TFLITE INVOKE FAIL: %d\n", static_cast<int>(invoke_status));
    return false;
  }

  const size_t output_elem_size = tensor_element_size(g_output_tensor->type);
  if (output_elem_size == 0) {
    return false;
  }

  const size_t output_elements = g_output_tensor->bytes / output_elem_size;
  size_t compare_count = write_elements;
  if (output_elements < compare_count) {
    compare_count = output_elements;
  }

  if (compare_count == 0) {
    return false;
  }

  float mse = 0.0f;
  size_t ref_idx = 0;
  for (size_t t = 0; t < static_cast<size_t>(model_steps) && ref_idx < compare_count; ++t) {
    const size_t src_step = (t < step_offset) ? 0 : (t - step_offset);
    const size_t base = src_step * FEATURE_COUNT;

    for (size_t f = 0; f < static_cast<size_t>(model_features) && ref_idx < compare_count; ++f) {
      const float reference_value = (f < FEATURE_COUNT) ? ordered_input[base + f] : 0.0f;
      float output_value = 0.0f;

      if (g_output_tensor->type == kTfLiteFloat32) {
        output_value = g_output_tensor->data.f[ref_idx];
      } else if (g_output_tensor->type == kTfLiteInt8 || g_output_tensor->type == kTfLiteUInt8) {
        const float scale = g_output_tensor->params.scale;
        const int32_t zero_point = g_output_tensor->params.zero_point;
        if (scale == 0.0f) {
          return false;
        }

        if (g_output_tensor->type == kTfLiteInt8) {
          output_value = (static_cast<int32_t>(g_output_tensor->data.int8[ref_idx]) - zero_point) * scale;
        } else {
          output_value = (static_cast<int32_t>(g_output_tensor->data.uint8[ref_idx]) - zero_point) * scale;
        }
      } else {
        return false;
      }

      const float diff = reference_value - output_value;
      mse += diff * diff;
      ref_idx++;
    }
  }

  *reconstruction_error = mse / static_cast<float>(compare_count);
  return true;
}

void send_classification(Severity severity, float error_value, uint32_t sample_index) {
  char json_payload[160];
  const int payload_len = snprintf(
      json_payload,
      sizeof(json_payload),
      "{\"severity\":\"%s\",\"error\":%.6f,\"sample_index\":%lu}",
      severity_to_text(severity),
      static_cast<double>(error_value),
      static_cast<unsigned long>(sample_index));

  if (payload_len <= 0 || payload_len >= static_cast<int>(sizeof(json_payload))) {
    return;
  }

  uint8_t encoded[MAX_FRAME_BYTES];
  const size_t encoded_len = encode_cobs(
      reinterpret_cast<const uint8_t *>(json_payload),
      static_cast<size_t>(payload_len),
      encoded,
      sizeof(encoded));

  if (encoded_len == 0) {
    return;
  }

  Serial.write(encoded, encoded_len);
  Serial.write(static_cast<uint8_t>(0x00));
}

void setup_tflm() {
  g_model = tflite::GetModel(g_model_data);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLITE SCHEMA MISMATCH");
    g_tflm_ready = false;
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      g_model,
      g_resolver,
      g_tensor_arena,
      TENSOR_ARENA_SIZE);

  g_interpreter = &static_interpreter;

  const TfLiteStatus alloc_status = g_interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.printf("TFLITE ALLOC FAIL: %d\n", static_cast<int>(alloc_status));
    g_tflm_ready = false;
    return;
  }

  g_input_tensor = g_interpreter->input(0);
  g_output_tensor = g_interpreter->output(0);
  g_tflm_ready = (g_input_tensor != nullptr && g_output_tensor != nullptr);

  if (!g_tflm_ready) {
    Serial.println("TFLITE TENSOR HANDLE FAIL");
  }
}

// -------------------- Arduino lifecycle --------------------
void setup() {
  Serial.begin(UART_BAUD_RATE);
  delay(200);

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  pinMode(FALLBACK_LED_PIN, OUTPUT);
  digitalWrite(FALLBACK_LED_PIN, LOW);

  g_imu.initialize();
  g_imu_available = g_imu.testConnection();
  if (!g_imu_available) {
    Serial.println("IMU NOT FOUND - CONTINUING");
  }

  setup_tflm();

  if (g_tflm_ready) {
    Serial.println("ESP32 READY");
  } else {
    Serial.println("ESP32 READY (TFLITE INACTIVE)");
  }
}

void loop() {
  update_led_pattern(g_last_result.severity);

  uint8_t encoded[MAX_FRAME_BYTES];
  size_t encoded_len = 0;
  if (!read_uart_frame(encoded, &encoded_len, UART_TIMEOUT_MS)) {
    Serial.println("UART TIMEOUT");
    return;
  }

  uint8_t decoded[MAX_FRAME_BYTES];
  const size_t decoded_len = decode_cobs(encoded, encoded_len, decoded, sizeof(decoded));
  if (decoded_len == 0) {
    Serial.println("COBS DECODE FAIL");
    return;
  }

  float uart_fields[UART_FLOAT_COUNT] = {0.0f};
  if (!parse_payload_to_fields(decoded, decoded_len, uart_fields)) {
    Serial.println("UART PAYLOAD PARSE FAIL");
    return;
  }

#if LIVE_IMU
  {
    float imu_attitude_error = 0.0f;
    float imu_gyro_magnitude = 0.0f;
    float imu_accel_magnitude = 0.0f;

    if (read_live_imu(&imu_attitude_error, &imu_gyro_magnitude, &imu_accel_magnitude)) {
      uart_fields[12] = imu_attitude_error;
      uart_fields[13] = imu_gyro_magnitude;
      uart_fields[14] = imu_accel_magnitude;
    } else {
      Serial.println("IMU FALLBACK");
    }
  }
#endif

  float features[FEATURE_COUNT] = {0.0f};
  build_feature_vector(uart_fields, features);
  push_features_to_window(features);
  g_sample_index++;

  if (!g_window_ready) {
    return;
  }

  float reconstruction_error = 0.0f;
  if (!run_inference(&reconstruction_error)) {
    // Skip classification on inference failure, continue operation.
    Serial.println("INFERENCE SKIPPED");
    return;
  }

  const Severity severity = classify_error(reconstruction_error);
  g_last_result = {
      severity,
      reconstruction_error,
      g_sample_index,
  };

  send_classification(severity, reconstruction_error, g_sample_index);
}

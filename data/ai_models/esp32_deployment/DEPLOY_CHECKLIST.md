# ESP32-S3 Deployment Checklist

## Pre-flash
- [ ] Arduino IDE version confirmed (>=2.0)
- [ ] ESP32-S3 board package installed in Arduino IDE
- [ ] TensorFlowLite_ESP32 library installed
- [ ] MPU6050 library installed
- [ ] model.h present in same directory as esp32_fusion.ino
- [ ] model size confirmed < 1500KB
- [ ] tensor arena size set correctly in firmware

## Flashing
- [ ] ESP32-S3 connected via USB data cable (not charge-only)
- [ ] Correct COM port selected in Arduino IDE
- [ ] Board set to: ESP32S3 Dev Module
- [ ] Flash mode: QIO 80MHz
- [ ] Partition scheme: Huge APP (3MB No OTA)
- [ ] Upload speed: 921600

## Post-flash verification
- [ ] Open Serial Monitor at 115200 baud
- [ ] Confirm "ESP32 READY" message appears
- [ ] Start Python pipeline: python src/main.py --mode realtime
- [ ] Confirm UART packets arriving on ESP32 (sample_index incrementing)
- [ ] Confirm classification responses returning to Python
- [ ] Inject test anomaly via /api/inject and confirm CRITICAL response
- [ ] Confirm LED behaviour matches severity

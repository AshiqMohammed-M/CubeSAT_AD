import logging
import math
import json
import os
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import yaml
from mcu_logger import MCULogger
#completed

# ---------------------------------------------------------------------------
# Load all thresholds from the central config file
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "eps_config.yaml"

def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        logging.getLogger(__name__).warning(
            "eps_config.yaml not found at %s – using hardcoded defaults", _CONFIG_PATH
        )
        return {}

_CFG = _load_config()
_RULES_CFG = _CFG.get("rules", {})
# ======================================================
# MCU → OBC COMMUNICATION INTERFACE WITH TEMPORAL BUFFER
# ======================================================
class OBCInterface:
    def __init__(self, window_size_seconds=30, max_data_points=100):
        self.logger = MCULogger("mcu_obc_interface.log")
        self.window_size = timedelta(seconds=window_size_seconds)
        self.data_buffer = deque(maxlen=max_data_points)  # Circular memory
        self.message_counter = 0
        
        # Create output directory for OBC messages
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.output_dir = os.path.join(BASE_DIR, "data", "mcu", "outputs", "obc_messages")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"OBC interface initialized (buffer: {window_size_seconds}s)")
    
    def add_sensor_data(self, sensor_data):
        """Adds sensor data to temporal buffer with timestamp"""
        data_point = {
            "timestamp": datetime.now(),
            "data": sensor_data.copy()
        }
        self.data_buffer.append(data_point)
        
        # Clean old data
        self._clean_old_data()
    
    def _clean_old_data(self):
        """Removes data older than the temporal window"""
        cutoff_time = datetime.now() - self.window_size
        while self.data_buffer and self.data_buffer[0]["timestamp"] < cutoff_time:
            self.data_buffer.popleft()
    
    def get_window_data(self):
        """Returns temporal window data (last 30 seconds)"""
        self._clean_old_data()
        return [point["data"] for point in self.data_buffer]
    
    def send_to_obc(self, message_type, payload, include_window_data=True):
        """Sends a message to OBC with optional window data"""
        try:
            self.message_counter += 1
            
            # Prepare complete message
            obc_message = {
                "header": {
                    "message_id": self.message_counter,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "source": "EPS_MCU",
                    "message_type": message_type,
                    "priority": "HIGH" if "CRITICAL" in message_type else "MEDIUM",
                    "version": "1.0"
                },
                "payload": payload
            }
            
            # Add window data for SUMMARY and ALERT
            # OBC requires minimum 30-point window
            if include_window_data and message_type in ["SUMMARY", "ALERT_CRITICAL"]:
                window_data = self.get_window_data()
                if len(window_data) >= 30:
                    obc_message["payload"]["temporal_window"] = {
                        "window_size_seconds": self.window_size.total_seconds(),
                        "data_points_count": len(window_data),
                        "sensor_data": window_data[-30:]  # OBC LSTM needs >=30 points
                    }
                else:
                    self.logger.info(f"OBC window deferred: only {len(window_data)}/30 points buffered")
                    obc_message["payload"]["temporal_window"] = None
            
            # Save message to file
            message_filename = f"obc_message_{self.message_counter:06d}_{message_type}.json"
            message_path = os.path.join(self.output_dir, message_filename)
            
            with open(message_path, 'w') as f:
                json.dump(obc_message, f, indent=2, default=str)
            
            # Console simulation
            print(f"\n{'='*60}")
            print(f"📡 [OBC-TX] Message #{self.message_counter}")
            print(f"   Type: {message_type} | Priority: {obc_message['header']['priority']}")
            print(f"   Timestamp: {obc_message['header']['timestamp']}")
            tw = obc_message['payload'].get('temporal_window') or {}
            print(f"   Window: {len(tw.get('sensor_data', []))} points")
            print(f"   Saved: {message_filename}")
            print(f"{'='*60}")
            
            self.logger.info(f"OBC_TX [{message_type}] → {len(window_data) if include_window_data else 0} data points")
            return True
            
        except Exception as e:
            self.logger.error(f"OBC send error: {e}")
            return False

# Global instance
obc_interface = OBCInterface(window_size_seconds=30)

def send_to_obc(message_type, payload):
    """Legacy function for compatibility"""
    return obc_interface.send_to_obc(message_type, payload)

class MCU_RuleEngine:
    def __init__(self, window_size=30):
        self.logger = MCULogger("mcu_rule_engine.log")
        self.window_size = window_size
        self.obc_interface = obc_interface  # Use global interface

        # Buffers for rule persistence
        self.overcurrent_buffer = deque(maxlen=3)
        self.ratio_buffer = deque(maxlen=10)
        self.oscillation_buffer = deque(maxlen=10)
        self.attitude_error_buffer = deque(maxlen=2)     # R8
        self.tumble_buffer = deque(maxlen=3)              # R11
        self.heartbeat_counter = 0

        # System state
        self.last_alert_time = {}
        self.alert_throttle_interval = 60
        self.system_state = "NORMAL"
        self.actions_taken = []

        # Thresholds – loaded from src/config/eps_config.yaml (fallback to defaults)
        self.thresholds = {
            "T_batt_critical":    _RULES_CFG.get("T_batt_critical",   60.0),
            "I_batt_overcurrent": _RULES_CFG.get("I_batt_overcurrent", 3.0),
            "V_batt_undervoltage":_RULES_CFG.get("V_batt_undervoltage",3.2),
            "ratio_min":          _RULES_CFG.get("ratio_min",           0.7),
            "T_batt_sensor_fault":_RULES_CFG.get("T_batt_sensor_fault",120.0),
            "V_batt_sensor_min":  _RULES_CFG.get("V_batt_sensor_min",   0.0),
            "V_batt_sensor_max":  _RULES_CFG.get("V_batt_sensor_max",  20.0),
            "I_batt_sensor_max":  _RULES_CFG.get("I_batt_sensor_max",  10.0),
            # R8-R11: attitude / IMU thresholds
            "attitude_error_critical_deg": _RULES_CFG.get("attitude_error_critical_deg", 30.0),
            "gyro_spike_threshold_rad_s":  _RULES_CFG.get("gyro_spike_threshold_rad_s",  1.5),
            "accel_dead_threshold_m_s2":   _RULES_CFG.get("accel_dead_threshold_m_s2",   0.1),
            "angular_velocity_tumble_rad_s": _RULES_CFG.get("angular_velocity_tumble_rad_s", 0.4),
        }

        self.logger.info("MCU_RuleEngine initialized with 30s temporal buffer")

    def trigger_action(self, action_type, details=None):
        """Triggers a physical action and logs it"""
        mapping = {
            "CUT_POWER": "Emergency power cut",
            "ACTIVATE_COOLING": "Cooling activation", 
            "REDUCE_LOAD": "Load reduction",
            "ISOLATE_BATTERY": "Battery isolation",
            "LED_RED": "Red LED activated",
            "LED_YELLOW": "Yellow LED activated", 
            "LED_GREEN": "Green LED activated",
            "BUZZER_ALARM": "Sound alarm activated",
            "INCREASE_LOGGING": "Intensive logging activated"
        }
        
        msg = mapping.get(action_type, "Unknown action")
        log_msg = f"ACTION: {msg}"
        if details:
            log_msg += f" - Details: {details}"
            
        self.logger.info(log_msg)
        self.actions_taken.append({
            "action": action_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        })
        return True

    # ======================================================
    # RULES R1-R7 (unchanged but optimized)
    # ======================================================
    def _check_sensor_fault(self, s):
        faults = []
        for key, value in s.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                faults.append(f"{key}_NaN")
        if s["T_batt"] > self.thresholds["T_batt_sensor_fault"]:
            faults.append("T_batt_out_of_range")
        if not (self.thresholds["V_batt_sensor_min"] <= s["V_batt"] <= self.thresholds["V_batt_sensor_max"]):
            faults.append("V_batt_out_of_range") 
        if abs(s["I_batt"]) > self.thresholds["I_batt_sensor_max"]:
            faults.append("I_batt_out_of_range")
        return faults

    def _check_overheat(self, s): 
        return s["T_batt"] > self.thresholds["T_batt_critical"]

    def _check_overcurrent(self, s):
        self.overcurrent_buffer.append(abs(s["I_batt"]) > self.thresholds["I_batt_overcurrent"])
        return len(self.overcurrent_buffer) == 3 and all(self.overcurrent_buffer)

    def _check_deep_discharge(self, s):
        return (s["V_batt"] < self.thresholds["V_batt_undervoltage"] and s["I_batt"] < 0)

    def _check_dcdc_ratio(self, s):
        ratio = s["V_bus"] / s["V_solar"] if s["V_solar"] > 0.1 else 0
        self.ratio_buffer.append(ratio)
        if len(self.ratio_buffer) >= 5:
            avg_ratio = sum(self.ratio_buffer) / len(self.ratio_buffer)
            return avg_ratio < self.thresholds["ratio_min"]
        return False

    def _check_oscillation(self, s):
        self.oscillation_buffer.append(s["V_bus"])
        if len(self.oscillation_buffer) >= 5:
            mean = sum(self.oscillation_buffer) / len(self.oscillation_buffer)
            variance = sum((x - mean) ** 2 for x in self.oscillation_buffer) / len(self.oscillation_buffer)
            return variance > 0.1
        return False

    # ── R8-R11: attitude / IMU checks ──────────────────────────
    def _check_attitude_error(self, s):
        """R8: 2 consecutive samples with attitude_error > critical → CRITICAL"""
        err = s.get("attitude_error_deg", 0.0)
        self.attitude_error_buffer.append(err)
        if len(self.attitude_error_buffer) >= 2:
            return all(
                v > self.thresholds["attitude_error_critical_deg"]
                for v in self.attitude_error_buffer
            )
        return False

    def _check_gyro_spike(self, s):
        """R9: any single gyro axis exceeds spike threshold → WARNING"""
        thresh = self.thresholds["gyro_spike_threshold_rad_s"]
        for axis in ("gyro_x", "gyro_y", "gyro_z"):
            if abs(s.get(axis, 0.0)) > thresh:
                return True
        return False

    def _check_accel_failure(self, s):
        """R10: accelerometer magnitude near zero → CRITICAL (sensor dead)"""
        mag = s.get("accel_magnitude", 9.81)
        return mag < self.thresholds["accel_dead_threshold_m_s2"]

    def _check_tumble(self, s):
        """R11: 3 consecutive angular-velocity magnitudes > tumble threshold → CRITICAL"""
        mag = s.get("angular_velocity_magnitude", 0.0)
        self.tumble_buffer.append(mag)
        if len(self.tumble_buffer) >= 3:
            return all(
                v > self.thresholds["angular_velocity_tumble_rad_s"]
                for v in self.tumble_buffer
            )
        return False

    def _throttle_alert(self, alert_type):
        """Anti-spam for alerts"""
        current_time = datetime.now().timestamp()
        last_time = self.last_alert_time.get(alert_type, 0)
        
        if current_time - last_time < self.alert_throttle_interval:
            return False
            
        self.last_alert_time[alert_type] = current_time
        return True

    # ======================================================
    # RULE APPLICATION WITH COMPLETE OBC TRANSMISSION
    # ======================================================
    def apply_rules(self, s):
        """Applies rules R1-R11 with OBC transmission and temporal data"""
        # Add data to temporal buffer
        self.obc_interface.add_sensor_data(s)
        
        # Reset actions for this cycle
        self.actions_taken = []
        actions = []
        message_type = "STATUS_OK"
        details = {"rule_triggered": None, "values": {}}

        # R6: Sensor fault check first
        sensor_faults = self._check_sensor_fault(s)
        if sensor_faults:
            self.trigger_action("LED_YELLOW", "Sensor fault detected")
            self.trigger_action("INCREASE_LOGGING", f"Faults: {sensor_faults}")
            message_type = "DIAGNOSTIC_SENSOR"
            details.update({
                "sensor_faults": sensor_faults,
                "actions_taken": [action["action"] for action in self.actions_taken]
            })
            
            # Send to OBC WITHOUT window data (instantaneous fault)
            self.obc_interface.send_to_obc(message_type, details, include_window_data=False)
            self.logger.warning(f"Sensor fault: {sensor_faults}")
            return actions, message_type, details

        # R1: Battery overheat (immediate critical)
        if self._check_overheat(s):
            if self._throttle_alert("R1"):
                self.trigger_action("CUT_POWER", f"Critical temperature: {s['T_batt']}°C")
                self.trigger_action("LED_RED")
                self.trigger_action("BUZZER_ALARM")
                message_type = "ALERT_CRITICAL"
                details.update({
                    "rule_triggered": "R1", 
                    "T_batt": s["T_batt"],
                    "actions_taken": [action["action"] for action in self.actions_taken],
                    "emergency_level": "HIGH"
                })
                
                # Send to OBC WITH window data
                self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
                self.logger.critical(f"R1 - Battery overheat: {s['T_batt']}°C")
                self.system_state = "CRITICAL"
                return actions, message_type, details

        # R3: Deep discharge (immediate critical)
        if self._check_deep_discharge(s):
            if self._throttle_alert("R3"):
                self.trigger_action("ISOLATE_BATTERY", f"Low voltage: {s['V_batt']}V")
                self.trigger_action("LED_RED")
                self.trigger_action("BUZZER_ALARM")
                message_type = "ALERT_CRITICAL"
                details.update({
                    "rule_triggered": "R3",
                    "V_batt": s["V_batt"],
                    "I_batt": s["I_batt"],
                    "actions_taken": [action["action"] for action in self.actions_taken],
                    "emergency_level": "HIGH"
                })
                
                self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
                self.logger.critical(f"R3 - Deep discharge: V={s['V_batt']}V")
                self.system_state = "CRITICAL"
                return actions, message_type, details

        # R2: Overcurrent (3s persistence)
        if self._check_overcurrent(s):
            if self._throttle_alert("R2"):
                self.trigger_action("REDUCE_LOAD", f"High current: {s['I_batt']}A")
                self.trigger_action("LED_RED")
                message_type = "ALERT_CRITICAL"
                details.update({
                    "rule_triggered": "R2",
                    "I_batt": s["I_batt"],
                    "actions_taken": [action["action"] for action in self.actions_taken],
                    "emergency_level": "HIGH"
                })
                
                self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
                self.logger.critical(f"R2 - Overcurrent: {s['I_batt']}A")
                self.system_state = "CRITICAL"
                return actions, message_type, details

        # R4: Abnormal DC/DC ratio
        if self._check_dcdc_ratio(s):
            ratio_value = s["V_bus"] / s["V_solar"] if s["V_solar"] > 0.1 else 0
            self.trigger_action("REDUCE_LOAD", f"Abnormal ratio: {ratio_value:.2f}")
            self.trigger_action("LED_YELLOW")
            message_type = "SUMMARY"
            details.update({
                "rule_triggered": "R4",
                "ratio": ratio_value,
                "actions_taken": [action["action"] for action in self.actions_taken],
                "emergency_level": "MEDIUM"
            })
            
            self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
            self.logger.warning(f"R4 - Abnormal DC/DC ratio: {ratio_value:.2f}")

        # R5: Bus oscillation
        elif self._check_oscillation(s):
            self.trigger_action("INCREASE_LOGGING", "Oscillation detected")
            self.trigger_action("LED_YELLOW")
            message_type = "SUMMARY"
            details.update({
                "rule_triggered": "R5",
                "actions_taken": [action["action"] for action in self.actions_taken],
                "emergency_level": "MEDIUM"
            })
            
            self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
            self.logger.warning("R5 - Bus oscillation detected")

        # R8: Attitude error critical (2-sample persistence)
        elif self._check_attitude_error(s):
            if self._throttle_alert("R8"):
                self.trigger_action("SAFE_MODE", f"Attitude error: {s.get('attitude_error_deg', 0):.1f}°")
                self.trigger_action("LED_RED")
                message_type = "ALERT_CRITICAL"
                details.update({
                    "rule_triggered": "R8",
                    "attitude_error_deg": s.get("attitude_error_deg", 0),
                    "actions_taken": [a["action"] for a in self.actions_taken],
                    "emergency_level": "HIGH"
                })
                self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
                self.logger.critical(f"R8 - Attitude error critical: {s.get('attitude_error_deg', 0):.1f}°")
                self.system_state = "CRITICAL"
                return actions, message_type, details

        # R9: Gyro spike (single-sample, WARNING)
        elif self._check_gyro_spike(s):
            self.trigger_action("INCREASE_LOGGING", "Gyro spike detected")
            self.trigger_action("LED_YELLOW")
            message_type = "SUMMARY"
            details.update({
                "rule_triggered": "R9",
                "gyro_x": s.get("gyro_x", 0),
                "gyro_y": s.get("gyro_y", 0),
                "gyro_z": s.get("gyro_z", 0),
                "actions_taken": [a["action"] for a in self.actions_taken],
                "emergency_level": "MEDIUM"
            })
            self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
            self.logger.warning("R9 - Gyro spike detected")

        # R10: Accelerometer failure (sensor dead, CRITICAL)
        elif self._check_accel_failure(s):
            if self._throttle_alert("R10"):
                self.trigger_action("SAFE_MODE", "Accelerometer dead")
                self.trigger_action("LED_RED")
                message_type = "ALERT_CRITICAL"
                details.update({
                    "rule_triggered": "R10",
                    "accel_magnitude": s.get("accel_magnitude", 0),
                    "actions_taken": [a["action"] for a in self.actions_taken],
                    "emergency_level": "HIGH"
                })
                self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
                self.logger.critical(f"R10 - Accelerometer failure: mag={s.get('accel_magnitude', 0):.3f}")
                self.system_state = "CRITICAL"
                return actions, message_type, details

        # R11: Tumble detection (3-sample persistence, CRITICAL)
        elif self._check_tumble(s):
            if self._throttle_alert("R11"):
                self.trigger_action("SAFE_MODE", "Tumble detected")
                self.trigger_action("LED_RED")
                self.trigger_action("BUZZER_ALARM")
                message_type = "ALERT_CRITICAL"
                details.update({
                    "rule_triggered": "R11",
                    "angular_velocity_magnitude": s.get("angular_velocity_magnitude", 0),
                    "actions_taken": [a["action"] for a in self.actions_taken],
                    "emergency_level": "HIGH"
                })
                self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
                self.logger.critical("R11 - Tumble detected")
                self.system_state = "CRITICAL"
                return actions, message_type, details

        # R7: Normal state + periodic heartbeat
        else:
            self.trigger_action("LED_GREEN", "Nominal system")
            message_type = "STATUS_HEARTBEAT"
            details.update({"rule_triggered": "R7"})
            
            # Heartbeat every 10 cycles (without window data)
            self.heartbeat_counter += 1
            if self.heartbeat_counter % 10 == 0:
                self.obc_interface.send_to_obc("STATUS_HEARTBEAT", {
                    "system_status": "NORMAL",
                    "uptime_cycles": self.heartbeat_counter,
                    "rule_engine_status": "OPERATIONAL"
                }, include_window_data=False)
                
            self.system_state = "NORMAL"

        details["actions_triggered"] = self.actions_taken.copy()
        return actions, message_type, details

    def get_system_state(self):
        return self.system_state

    def get_recent_actions(self):
        return self.actions_taken
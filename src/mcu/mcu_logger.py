import logging
import sys
import os
from datetime import datetime
#completed
class MCULogger:
    """
    Universal log manager for MCU + AI systems.
    Compatible with setup_logger() from older versions.
    Supports UTF-8 and avoids duplicate log handlers.
    """
    def __init__(self, log_file="mcu_rule_engine.log", name="MCU_Logger"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Prevent adding the same handlers multiple times
        if not self.logger.handlers:
            # Log format with date and milliseconds
            formatter = logging.Formatter(
                "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            # --- Log directory ---
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_dir = os.path.join(BASE_DIR, "data", "mcu", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_file)

            # --- File handler (UTF-8) ---
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)

            # --- Console handler (UTF-8 if supported) ---
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            try:
                # Python 3.7+ compatible
                console_handler.stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

            print(f"Logging initialized → {log_path}")
    # === Level methods ===
    def info(self, message):
        try:
            self.logger.info(message)
        except UnicodeEncodeError:
            self.logger.info(message.encode("utf-8", "ignore").decode("utf-8"))

    def warning(self, message):
        try:
            self.logger.warning(message)
        except UnicodeEncodeError:
            self.logger.warning(message.encode("utf-8", "ignore").decode("utf-8"))

    def error(self, message):
        try:
            self.logger.error(message)
        except UnicodeEncodeError:
            self.logger.error(message.encode("utf-8", "ignore").decode("utf-8"))

    def critical(self, message):
        try:
            self.logger.critical(message)
        except UnicodeEncodeError:
            self.logger.critical(message.encode("utf-8", "ignore").decode("utf-8"))


# === Retro compatibility function ===
def setup_logger(log_file="mcu_rule_engine.log"):
    """
    Compatibility with older code:
    MCU_MainLoop uses setup_logger(), so we return an instance of MCULogger.
    """
    return MCULogger(log_file)
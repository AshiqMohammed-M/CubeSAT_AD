#!/usr/bin/env python3
"""Recalibrate AI anomaly thresholds from normal-only reconstruction errors."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_PATH = Path("data/ai_models/model_complex/lstm_autoencoder_tflite_native.h5")
DATA_PATH = Path("data/ai_training_base/ai_sequence_data.npy")
CONFIG_PATH = Path("src/config/eps_config.yaml")


def _load_windows(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    windows = np.load(path, allow_pickle=False)
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D array [N, T, F], got shape: {windows.shape}")

    return windows.astype(np.float32)


def _compute_reconstruction_errors(model: tf.keras.Model, windows: np.ndarray) -> np.ndarray:
    recon = model.predict(windows, batch_size=32, verbose=0)
    if recon.shape != windows.shape:
        raise ValueError(f"Model output shape mismatch: expected {windows.shape}, got {recon.shape}")

    errors = np.mean(np.square(windows - recon), axis=(1, 2))
    return errors.astype(np.float64)


def _replace_threshold_line(line: str, key: str, value: float) -> str:
    prefix = line[: len(line) - len(line.lstrip(" "))]
    return f"{prefix}{key}: {value:.10f}\n"


def _update_simple_thresholds_in_yaml(config_path: Path, normal: float, warning: float, critical: float) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)

    ai_idx = None
    simple_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*ai_thresholds:\s*$", line):
            ai_idx = i
            continue
        if ai_idx is not None and i > ai_idx:
            if re.match(r"^\s{2}simple:\s*$", line):
                simple_idx = i
                break
            # Left the ai_thresholds block before finding simple
            if re.match(r"^\S", line):
                break

    if simple_idx is None:
        raise ValueError("Could not find 'ai_thresholds.simple' block in config")

    end_idx = len(lines)
    for i in range(simple_idx + 1, len(lines)):
        line = lines[i]
        if line.strip() == "":
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent <= 2 and not line.lstrip(" ").startswith("#"):
            end_idx = i
            break

    key_to_value = {
        "normal": normal,
        "warning": warning,
        "critical": critical,
    }

    replaced = {"normal": False, "warning": False, "critical": False}

    for i in range(simple_idx + 1, end_idx):
        stripped = lines[i].strip()
        for key, value in key_to_value.items():
            if re.match(rf"^{key}:\s*[-+]?\d", stripped):
                lines[i] = _replace_threshold_line(lines[i], key, value)
                replaced[key] = True

    missing = [k for k, ok in replaced.items() if not ok]
    if missing:
        raise ValueError(f"Missing threshold keys under ai_thresholds.simple: {', '.join(missing)}")

    config_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return 1

    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print(f"Loading data: {DATA_PATH}")
    windows = _load_windows(DATA_PATH)

    errors = _compute_reconstruction_errors(model, windows)

    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))

    normal = mean_error + (1.0 * std_error)
    warning = mean_error + (2.0 * std_error)
    critical = mean_error + (3.0 * std_error)

    print(f"mean reconstruction error: {mean_error:.10f}")
    print(f"std: {std_error:.10f}")
    print(f"new normal threshold:   {normal:.10f}")
    print(f"new warning threshold:  {warning:.10f}")
    print(f"new critical threshold: {critical:.10f}")

    _update_simple_thresholds_in_yaml(CONFIG_PATH, normal, warning, critical)
    print(f"Updated config: {CONFIG_PATH}")

    print(f"constexpr float THRESH_NORMAL   = {normal:.10f}f;")
    print(f"constexpr float THRESH_WARNING  = {warning:.10f}f;")
    print(f"constexpr float THRESH_CRITICAL = {critical:.10f}f;")

    return 0


if __name__ == "__main__":
    sys.exit(main())

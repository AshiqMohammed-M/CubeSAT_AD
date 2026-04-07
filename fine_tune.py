#!/usr/bin/env python3
"""Fine-tune the restructured TFLite-native autoencoder."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf

MODEL_PATH = Path("data/ai_models/model_complex/lstm_autoencoder_tflite_native.h5")
TRAINING_DIR = Path("data/ai_training_base")
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 5e-5


def _build_training_candidates(base_dir: Path) -> List[Path]:
    return [
        base_dir / "ai_sequence_data.npy",
        base_dir / "ai_sequence_features.npy",
        base_dir / "ai_train_data.npy",
        base_dir / "train_windows.npy",
        base_dir / "telemetry_windows.npy",
        base_dir / "training_data.npy",
        base_dir / "ai_sequence_data.csv",
        base_dir / "training_data.csv",
        base_dir / "telemetry_data.csv",
    ]


def _print_missing_training_paths(paths_checked: Sequence[Path]) -> None:
    print("Training data not found. Paths checked:")
    for path in paths_checked:
        print(f"- {path}")
    print(f"- {TRAINING_DIR / '*.npy'}")
    print(f"- {TRAINING_DIR / '*.csv'}")


def _sliding_windows(matrix: np.ndarray, window: int) -> np.ndarray:
    num_rows = matrix.shape[0]
    if num_rows < window:
        raise ValueError(f"Not enough rows for window={window}: got {num_rows}")

    windows = [matrix[i : i + window] for i in range(num_rows - window + 1)]
    return np.stack(windows, axis=0)


def _load_array_from_npy(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Unsupported object from npy file: {type(data)}")
    return data


def _load_array_from_csv(path: Path) -> np.ndarray:
    # Use plain numpy parsing to keep this script dependency-light.
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.size == 0:
        raise ValueError("CSV contains no usable rows")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _prepare_training_data(data: np.ndarray, timesteps: int, features: int) -> np.ndarray:
    if data.ndim == 3:
        if data.shape[1] != timesteps:
            raise ValueError(f"Expected timesteps={timesteps}, got {data.shape[1]}")
        if data.shape[2] < features:
            raise ValueError(f"Expected at least {features} features, got {data.shape[2]}")
        windows = data[:, :, :features]
    elif data.ndim == 2:
        if data.shape[1] < features:
            raise ValueError(f"Expected at least {features} feature columns, got {data.shape[1]}")
        trimmed = data[:, :features]
        trimmed = trimmed[np.all(np.isfinite(trimmed), axis=1)]
        windows = _sliding_windows(trimmed, timesteps)
    else:
        raise ValueError(f"Unsupported data rank: {data.ndim}")

    windows = windows.astype(np.float32)
    windows = windows[np.all(np.isfinite(windows), axis=(1, 2))]
    if windows.shape[0] == 0:
        raise ValueError("No finite windows remained after preprocessing")
    return windows


def _load_training_windows(timesteps: int, features: int) -> Tuple[np.ndarray, Path]:
    explicit_candidates = _build_training_candidates(TRAINING_DIR)
    existing_explicit = [path for path in explicit_candidates if path.exists()]

    glob_candidates = sorted(TRAINING_DIR.glob("*.npy")) + sorted(TRAINING_DIR.glob("*.csv"))

    seen = set()
    ordered_candidates: List[Path] = []
    for path in existing_explicit + glob_candidates:
        if path not in seen:
            ordered_candidates.append(path)
            seen.add(path)

    if not ordered_candidates:
        _print_missing_training_paths(explicit_candidates)
        raise FileNotFoundError("Training data files were not found")

    last_error: Exception | None = None
    for candidate in ordered_candidates:
        try:
            if candidate.suffix.lower() == ".npy":
                raw = _load_array_from_npy(candidate)
            elif candidate.suffix.lower() == ".csv":
                raw = _load_array_from_csv(candidate)
            else:
                continue

            windows = _prepare_training_data(raw, timesteps=timesteps, features=features)
            return windows, candidate
        except Exception as exc:  # pragma: no cover - data quality is runtime dependent
            last_error = exc
            print(f"Skipping unusable training file {candidate}: {exc}")

    _print_missing_training_paths(explicit_candidates)
    if last_error is not None:
        raise RuntimeError(f"All discovered training files were unusable. Last error: {last_error}") from last_error
    raise RuntimeError("All discovered training files were unusable")


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"Restructured model not found: {MODEL_PATH}")
        return 1

    print(f"Loading model for fine-tuning: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    input_shape = model.input_shape
    if len(input_shape) != 3:
        print(f"Unexpected model input shape: {input_shape}")
        return 1

    timesteps = int(input_shape[1])
    features = int(input_shape[2])

    try:
        x_train, used_path = _load_training_windows(timesteps=timesteps, features=features)
    except Exception as exc:
        print(str(exc))
        return 1

    print(f"Using training data: {used_path}")
    print(f"Training tensor shape: {x_train.shape}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )

    history = model.fit(
        x_train,
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        shuffle=True,
        verbose=2,
    )

    model.save(MODEL_PATH, include_optimizer=False)
    print(f"Saved fine-tuned model: {MODEL_PATH}")

    if "loss" in history.history and history.history["loss"]:
        print(f"Final train loss: {history.history['loss'][-1]:.6f}")
    if "val_loss" in history.history and history.history["val_loss"]:
        print(f"Final val loss: {history.history['val_loss'][-1]:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

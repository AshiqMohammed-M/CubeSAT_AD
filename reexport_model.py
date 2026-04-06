#!/usr/bin/env python3
"""Re-export a Keras model to TFLite with builtins-only ops for ESP32 compatibility."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Set

import tensorflow as tf

MODEL_PATH = Path("data/ai_models/model_complex/ai_model_lstm_autoencoder.h5")
OUTPUT_PATH = Path("data/ai_models/esp32_deployment/model_noselect.tflite")
TENSORLIST_ERROR_KEYS = (
    "TensorList",
    "FlexTensorListReserve",
    "TensorListReserve",
    "TensorListStack",
    "TensorListSetItem",
)


def _extract_ops_via_interpreter(model_path: Path) -> List[str]:
    """Read operation names from a TFLite model via interpreter internals when available."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    if hasattr(interpreter, "_get_ops_details"):
        ops_details = interpreter._get_ops_details()  # pylint: disable=protected-access
        ops: Set[str] = set()
        for op in ops_details:
            op_name = op.get("op_name", "UNKNOWN")
            if isinstance(op_name, bytes):
                op_name = op_name.decode("utf-8", errors="replace")
            ops.add(str(op_name))
        return sorted(ops)

    return []


def _extract_ops_via_flatbuffer(model_content: bytes) -> List[str]:
    """Fallback op extraction by reading TFLite FlatBuffer metadata."""
    try:
        from tensorflow.lite.python import schema_py_generated as schema_fb
    except Exception:
        return []

    model = schema_fb.Model.GetRootAsModel(model_content, 0)
    op_codes = model.OperatorCodesLength()

    builtin_name_map = {
        value: name
        for name, value in schema_fb.BuiltinOperator.__dict__.items()
        if not name.startswith("_") and isinstance(value, int)
    }

    ops: Set[str] = set()
    for i in range(op_codes):
        code = model.OperatorCodes(i)
        builtin_code = code.BuiltinCode()

        if builtin_code == schema_fb.BuiltinOperator.CUSTOM:
            custom_code = code.CustomCode()
            if custom_code:
                if isinstance(custom_code, bytes):
                    custom_code = custom_code.decode("utf-8", errors="replace")
                ops.add(f"CUSTOM:{custom_code}")
            else:
                ops.add("CUSTOM")
        else:
            ops.add(builtin_name_map.get(builtin_code, f"BUILTIN_{builtin_code}"))

    return sorted(ops)


def _is_tensorlist_error(message: str, keys: Iterable[str]) -> bool:
    lower = message.lower()
    return any(key.lower() in lower for key in keys)


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}")
        alternates = sorted(MODEL_PATH.parent.glob("*.h5")) if MODEL_PATH.parent.exists() else []
        if alternates:
            print("Available .h5 files in model directory:")
            for candidate in alternates:
                print(f"- {candidate}")
        return 1

    print(f"Loading Keras model: {MODEL_PATH}")
    # compile=False avoids deserializing training-only losses/metrics from legacy .h5 files.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Converting with TFLITE_BUILTINS only (no SELECT_TF_OPS)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    try:
        tflite_model = converter.convert()
    except Exception as exc:  # pragma: no cover - conversion failures depend on TF/runtime
        message = str(exc)
        if _is_tensorlist_error(message, TENSORLIST_ERROR_KEYS):
            print(message)
            return 1
        raise

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(tflite_model)

    size_kb = OUTPUT_PATH.stat().st_size / 1024.0
    print(f"Saved TFLite model: {OUTPUT_PATH}")
    print(f"File size: {size_kb:.2f} KB")

    print("Ops used in exported model:")
    ops = _extract_ops_via_interpreter(OUTPUT_PATH)
    if not ops:
        ops = _extract_ops_via_flatbuffer(tflite_model)

    if not ops:
        print("- Unable to extract operation list with current TensorFlow runtime.")
    else:
        for op in ops:
            print(f"- {op}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

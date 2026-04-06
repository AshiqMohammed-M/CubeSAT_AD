#!/usr/bin/env python3
"""Export the TFLite-native autoencoder with TFLITE_BUILTINS only and verify ops."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Set

import tensorflow as tf

MODEL_PATH = Path("data/ai_models/model_complex/lstm_autoencoder_tflite_native.h5")
OUTPUT_PATH = Path("data/ai_models/esp32_deployment/model_noselect.tflite")


def _extract_ops_via_interpreter(model_path: Path) -> List[str]:
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


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"Input model not found: {MODEL_PATH}")
        return 1

    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Converting with TFLITE_BUILTINS only (no SELECT_TF_OPS, no quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tflite_model = converter.convert()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(tflite_model)

    size_kb = OUTPUT_PATH.stat().st_size / 1024.0
    print(f"Saved TFLite model: {OUTPUT_PATH}")
    print(f"File size: {size_kb:.2f} KB")

    ops = _extract_ops_via_interpreter(OUTPUT_PATH)
    if not ops:
        ops = _extract_ops_via_flatbuffer(tflite_model)

    print("Ops used in exported model:")
    if not ops:
        print("- Unable to extract operation list with current TensorFlow runtime.")
    else:
        for op in ops:
            print(f"- {op}")

    has_custom = any(op == "CUSTOM" or op.startswith("CUSTOM:") for op in ops)
    if has_custom:
        print("CUSTOM ops detected in exported model.")
        return 1

    print("Zero CUSTOM ops confirmed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

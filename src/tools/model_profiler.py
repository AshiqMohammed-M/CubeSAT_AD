#!/usr/bin/env python3
"""Profile TensorFlow Lite model artifacts for embedded deployment checks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


def _default_model_path() -> Path:
    candidates = [
        Path("data/ai_models/model_complex/ai_model_lstm_autoencoder_quant.tflite"),
        Path("data/ai_models/model_complex/ai_model_lstm_autoencoder.tflite"),
        Path("data/ai_models/model_simple/ai_autoencoder.tflite"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No default TFLite model found under data/ai_models/")


def _dtype_str(dtype: Any) -> str:
    try:
        return np.dtype(dtype).name
    except Exception:
        return str(dtype)


def _numeric_tensor_bytes(shape: np.ndarray, dtype: Any) -> int:
    try:
        np_dtype = np.dtype(dtype)
        if np_dtype.kind not in {"b", "i", "u", "f"}:
            return 0
        count = int(np.prod(shape))
        if count <= 0:
            return 0
        return count * np_dtype.itemsize
    except Exception:
        return 0


def _quantization_label(input_dtype: str, tensor_dtypes: set[str]) -> str:
    if input_dtype in {"int8", "uint8"}:
        return "INT8"
    if "int8" in tensor_dtypes or "uint8" in tensor_dtypes:
        return "INT8"
    # The project report expects float32 vs INT8 only.
    return "float32"


def _random_input_for_tensor(input_detail: dict[str, Any]) -> np.ndarray:
    shape = [int(x) for x in input_detail["shape"]]
    dtype = np.dtype(input_detail["dtype"])

    if dtype.kind == "f":
        return np.random.randn(*shape).astype(dtype)
    if dtype.kind in {"i", "u"}:
        return np.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=np.float32)


def profile_model(model_path: Path, runs: int, warmup: int) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_detail = input_details[0]

    tensor_details = interpreter.get_tensor_details()
    tensor_dtypes = {_dtype_str(t["dtype"]) for t in tensor_details}
    total_tensor_bytes = sum(
        _numeric_tensor_bytes(t["shape"], t["dtype"]) for t in tensor_details
    )

    result: dict[str, Any] = {
        "model_path": str(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "model_size_kb": model_path.stat().st_size / 1024.0,
        "input_shape": [int(x) for x in input_detail["shape"]],
        "output_shape": [int(x) for x in output_details[0]["shape"]],
        "input_dtype": _dtype_str(input_detail["dtype"]),
        "tensor_dtypes": sorted(tensor_dtypes),
        "quantization": _quantization_label(_dtype_str(input_detail["dtype"]), tensor_dtypes),
        "estimated_tensor_memory_bytes": int(total_tensor_bytes),
        "estimated_tensor_memory_kb": total_tensor_bytes / 1024.0,
        "inference_time_ms": None,
        "inference_error": None,
    }

    try:
        interpreter.allocate_tensors()
        input_data = _random_input_for_tensor(input_detail)

        for _ in range(max(0, warmup)):
            interpreter.set_tensor(input_detail["index"], input_data)
            interpreter.invoke()

        timings_ms: list[float] = []
        for _ in range(max(1, runs)):
            start = time.perf_counter()
            interpreter.set_tensor(input_detail["index"], input_data)
            interpreter.invoke()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)

        result["inference_time_ms"] = statistics.mean(timings_ms)
        result["inference_p95_ms"] = (
            statistics.quantiles(timings_ms, n=20)[18] if len(timings_ms) >= 20 else max(timings_ms)
        )
    except Exception as exc:
        result["inference_error"] = str(exc)

    return result


def _human_summary(profile: dict[str, Any]) -> str:
    inference_value = profile["inference_time_ms"]
    if inference_value is None:
        inference_str = "NOT FOUND"
    else:
        inference_str = f"{inference_value:.2f} ms"

    return "\n".join(
        [
            f"Model size     : {profile['model_size_kb']:.2f} KB",
            f"Inference time : {inference_str}",
            f"RAM usage      : {profile['estimated_tensor_memory_kb']:.2f} KB",
            f"Quantization   : {profile['quantization']}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a TFLite model for embedded deployment")
    parser.add_argument("--model", type=Path, default=None, help="Path to .tflite model")
    parser.add_argument("--runs", type=int, default=20, help="Number of timed inference runs")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs before timing")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("data/analyse/model_profile_report.json"),
        help="Write profiler output JSON to this path",
    )
    args = parser.parse_args()

    model_path = args.model if args.model is not None else _default_model_path()
    profile = profile_model(model_path=model_path, runs=args.runs, warmup=args.warmup)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(_human_summary(profile))
    print(f"Profile JSON   : {args.json_out}")


if __name__ == "__main__":
    main()

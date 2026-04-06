#!/usr/bin/env python3
"""Restructure an LSTM autoencoder into a TFLite-native model without Select TF ops."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import tensorflow as tf

SOURCE_MODEL_CANDIDATES = (
    Path("data/ai_models/model_complex/lstm_autoencoder.h5"),
    Path("data/ai_models/model_complex/ai_model_lstm_autoencoder.h5"),
)
OUTPUT_MODEL_PATH = Path("data/ai_models/model_complex/lstm_autoencoder_tflite_native.h5")
PRESERVED_THRESHOLDS = {
    "normal": 0.6091,
    "warning": 0.8378,
    "critical": 1.0665,
}


def _find_existing_path(candidates: Sequence[Path], label: str) -> Path | None:
    checked: List[Path] = []
    for candidate in candidates:
        checked.append(candidate)
        if candidate.exists():
            return candidate

    print(f"{label} not found. Paths checked:")
    for candidate in checked:
        print(f"- {candidate}")
    return None


def _convert_layer(source_layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    if isinstance(source_layer, tf.keras.layers.LSTM):
        cell = tf.keras.layers.LSTMCell(
            units=source_layer.units,
            activation=source_layer.activation,
            recurrent_activation=source_layer.recurrent_activation,
            use_bias=source_layer.use_bias,
            unit_forget_bias=source_layer.unit_forget_bias,
            dropout=source_layer.dropout,
            recurrent_dropout=source_layer.recurrent_dropout,
            name=f"{source_layer.name}_cell",
        )
        return tf.keras.layers.RNN(
            cell,
            return_sequences=source_layer.return_sequences,
            go_backwards=source_layer.go_backwards,
            unroll=True,
            name=source_layer.name,
        )

    if isinstance(source_layer, tf.keras.layers.RepeatVector):
        return tf.keras.layers.RepeatVector(source_layer.n, name=source_layer.name)

    if isinstance(source_layer, tf.keras.layers.Dense):
        return tf.keras.layers.Dense(
            units=source_layer.units,
            activation=source_layer.activation,
            use_bias=source_layer.use_bias,
            name=source_layer.name,
        )

    if isinstance(source_layer, tf.keras.layers.TimeDistributed):
        inner = source_layer.layer
        if isinstance(inner, tf.keras.layers.Dense):
            wrapped_dense = tf.keras.layers.Dense(
                units=inner.units,
                activation=inner.activation,
                use_bias=inner.use_bias,
                name=inner.name,
            )
            return tf.keras.layers.TimeDistributed(wrapped_dense, name=source_layer.name)
        raise ValueError(
            f"Unsupported TimeDistributed inner layer in source model: {inner.__class__.__name__}"
        )

    raise ValueError(f"Unsupported source layer type: {source_layer.name} ({source_layer.__class__.__name__})")


def _weights_are_compatible(source_weights: Iterable, target_weights: Iterable) -> bool:
    source_list = list(source_weights)
    target_list = list(target_weights)

    if len(source_list) != len(target_list):
        return False

    for source_weight, target_weight in zip(source_list, target_list):
        if source_weight.shape != target_weight.shape:
            return False

    return True


def _build_restructured_model(
    source_model: tf.keras.Model,
) -> Tuple[tf.keras.Model, List[Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]]]:
    input_shape = tuple(source_model.input_shape[1:])
    if len(input_shape) != 2:
        raise ValueError(f"Expected 3D input shape [batch, timesteps, features], got: {source_model.input_shape}")

    inputs = tf.keras.Input(shape=input_shape, name="input_layer")
    x = inputs
    layer_pairs: List[Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]] = []

    for source_layer in source_model.layers:
        if isinstance(source_layer, tf.keras.layers.InputLayer):
            continue

        target_layer = _convert_layer(source_layer)
        x = target_layer(x)
        layer_pairs.append((source_layer, target_layer))

    model = tf.keras.Model(inputs=inputs, outputs=x, name="lstm_autoencoder_tflite_native")
    return model, layer_pairs


def _transfer_weights(layer_pairs: Sequence[Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]]) -> Tuple[int, List[str]]:
    transferred_count = 0
    skipped_messages: List[str] = []

    for source_layer, target_layer in layer_pairs:
        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()

        if not source_weights and not target_weights:
            continue

        if _weights_are_compatible(source_weights, target_weights):
            target_layer.set_weights(source_weights)
            transferred_count += 1
        else:
            skipped_messages.append(
                f"{source_layer.name} -> {target_layer.name}: "
                f"shape mismatch ({[w.shape for w in source_weights]} vs {[w.shape for w in target_weights]})"
            )

    return transferred_count, skipped_messages


def main() -> int:
    source_model_path = _find_existing_path(SOURCE_MODEL_CANDIDATES, "Source model")
    if source_model_path is None:
        return 1

    print(f"Loading source model: {source_model_path}")
    source_model = tf.keras.models.load_model(source_model_path, compile=False)
    print(f"Source input shape: {source_model.input_shape}")
    print(f"Source output shape: {source_model.output_shape}")

    target_model, layer_pairs = _build_restructured_model(source_model)
    transferred_count, skipped_messages = _transfer_weights(layer_pairs)

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    target_model.save(OUTPUT_MODEL_PATH, include_optimizer=False)

    print(f"Saved restructured model: {OUTPUT_MODEL_PATH}")
    print(f"Layers with transferred weights: {transferred_count}")
    if skipped_messages:
        print("Layers skipped during transfer:")
        for message in skipped_messages:
            print(f"- {message}")

    print(
        "Preserved thresholds: "
        f"normal={PRESERVED_THRESHOLDS['normal']}, "
        f"warning={PRESERVED_THRESHOLDS['warning']}, "
        f"critical={PRESERVED_THRESHOLDS['critical']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

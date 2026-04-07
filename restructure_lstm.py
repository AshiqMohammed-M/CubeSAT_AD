#!/usr/bin/env python3
"""Restructure an LSTM autoencoder into a TILE-free, TFLite-native architecture."""

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
TARGET_TIMESTEPS = 30
TARGET_FEATURES = 14
TARGET_LATENT = 32


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


def _source_layer_map(source_model: tf.keras.Model) -> dict[str, tf.keras.layers.Layer]:
    return {layer.name: layer for layer in source_model.layers}


def _weights_are_compatible(source_weights: Iterable, target_weights: Iterable) -> bool:
    source_list = list(source_weights)
    target_list = list(target_weights)

    if len(source_list) != len(target_list):
        return False

    for source_weight, target_weight in zip(source_list, target_list):
        if source_weight.shape != target_weight.shape:
            return False

    return True


def _build_restructured_model(source_model: tf.keras.Model) -> tf.keras.Model:
    input_shape = tuple(source_model.input_shape[1:])
    if input_shape != (TARGET_TIMESTEPS, TARGET_FEATURES):
        raise ValueError(
            "Expected input shape [30, 14], "
            f"got: {source_model.input_shape}"
        )

    source_layers = _source_layer_map(source_model)
    source_enc1 = source_layers.get("encoder_lstm1")
    if not isinstance(source_enc1, tf.keras.layers.LSTM):
        raise ValueError("Source layer 'encoder_lstm1' was not found as LSTM")

    inputs = tf.keras.Input(shape=input_shape, name="input_layer")
    x = tf.keras.layers.RNN(
        tf.keras.layers.LSTMCell(
            units=64,
            activation=source_enc1.activation,
            recurrent_activation=source_enc1.recurrent_activation,
            use_bias=source_enc1.use_bias,
            unit_forget_bias=source_enc1.unit_forget_bias,
            dropout=source_enc1.dropout,
            recurrent_dropout=source_enc1.recurrent_dropout,
            name="encoder_lstm1_cell",
        ),
        return_sequences=False,
        unroll=True,
        name="encoder_lstm1",
    )(inputs)

    latent = tf.keras.layers.Dense(TARGET_LATENT, activation="relu", name="latent_dense")(x)
    decoder_hidden = tf.keras.layers.Dense(TARGET_LATENT, activation="relu", name="decoder_dense")(latent)
    flat_output = tf.keras.layers.Dense(
        TARGET_TIMESTEPS * TARGET_FEATURES,
        activation="linear",
        name="decoder_projection",
    )(decoder_hidden)
    outputs = tf.keras.layers.Reshape((TARGET_TIMESTEPS, TARGET_FEATURES), name="output")(flat_output)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_autoencoder_tflite_native")


def _transfer_weights(
    source_model: tf.keras.Model,
    target_model: tf.keras.Model,
) -> Tuple[int, List[str]]:
    source_layers = _source_layer_map(source_model)
    target_layers = _source_layer_map(target_model)

    layer_pairs: List[Tuple[str, str]] = [
        ("encoder_lstm1", "encoder_lstm1"),
        ("bottleneck", "decoder_dense"),
    ]

    transferred_count = 0
    skipped_messages: List[str] = []

    for source_name, target_name in layer_pairs:
        source_layer = source_layers.get(source_name)
        target_layer = target_layers.get(target_name)

        if source_layer is None or target_layer is None:
            skipped_messages.append(f"{source_name} -> {target_name}: layer missing")
            continue

        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()

        if not source_weights and not target_weights:
            continue

        if _weights_are_compatible(source_weights, target_weights):
            target_layer.set_weights(source_weights)
            transferred_count += 1
        else:
            skipped_messages.append(
                f"{source_name} -> {target_name}: "
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

    target_model = _build_restructured_model(source_model)
    transferred_count, skipped_messages = _transfer_weights(source_model, target_model)

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

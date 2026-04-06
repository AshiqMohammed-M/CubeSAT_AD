#!/usr/bin/env bash
set -euo pipefail

RAW_HEADER="model_raw.h"
FINAL_HEADER="model.h"

INPUT_CANDIDATES=(
  "model_noselect.tflite"
  "model_complex/model_noselect.tflite"
  "../model_complex/model_noselect.tflite"
)

INPUT_TFLITE=""
for candidate in "${INPUT_CANDIDATES[@]}"; do
  if [[ -f "$candidate" ]]; then
    INPUT_TFLITE="$candidate"
    break
  fi
done

if [[ -z "$INPUT_TFLITE" ]]; then
  echo "Input TFLite file not found. Paths checked:" >&2
  for candidate in "${INPUT_CANDIDATES[@]}"; do
    echo "- $candidate" >&2
  done
  exit 1
fi

if ! command -v xxd >/dev/null 2>&1; then
  echo "xxd is required but was not found on PATH." >&2
  exit 1
fi

# Step 1: Generate raw header with xxd output.
xxd -i "$INPUT_TFLITE" > "$RAW_HEADER"

raw_array_name="$(sed -n 's/^unsigned char \([A-Za-z0-9_]*\)\[\] = {.*/\1/p' "$RAW_HEADER" | head -n1)"
raw_len_name="$(sed -n 's/^unsigned int \([A-Za-z0-9_]*\) = .*/\1/p' "$RAW_HEADER" | head -n1)"

if [[ -z "$raw_array_name" || -z "$raw_len_name" ]]; then
  echo "Failed to parse xxd output symbols in $RAW_HEADER" >&2
  exit 1
fi

{
  echo "#ifndef MODEL_H"
  echo "#define MODEL_H"
  echo
  sed -E \
    -e "s/^unsigned char ${raw_array_name}\[\] = \{/const unsigned char g_model_data[] = {/" \
    -e "s/^unsigned int ${raw_len_name} = /const unsigned int g_model_len = /" \
    "$RAW_HEADER"
  echo
  echo "#endif"
} > "$FINAL_HEADER"

bytes="$(stat -c%s "$INPUT_TFLITE")"
echo "model.h generated — ${bytes} bytes, array: g_model_data, len: g_model_len"

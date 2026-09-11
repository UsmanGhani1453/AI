#!/usr/bin/env bash
# convert_to_gguf.sh
#
# Converts the merged Hugging Face model (produced by
# training/merge_adapter.py) into an fp16 GGUF file, using llama.cpp's
# conversion script. No C++ build is required for this step.
#
# Run from the project root:
#   bash deployment/convert_to_gguf.sh
#
# Requires: git, Python (the project's venv is fine).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGED_MODEL="$PROJECT_ROOT/models/merged"
LLAMA_CPP_DIR="$PROJECT_ROOT/deployment/llama.cpp"
GGUF_OUT_DIR="$PROJECT_ROOT/models/gguf"
GGUF_OUT_FILE="$GGUF_OUT_DIR/truck-dispatch-f16.gguf"

if [ ! -d "$MERGED_MODEL" ]; then
    echo "Merged model not found at $MERGED_MODEL. Run 'python training/merge_adapter.py' first." >&2
    exit 1
fi

mkdir -p "$GGUF_OUT_DIR"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Cloning llama.cpp (conversion script only, no build needed)..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi

echo "Installing conversion script dependencies..."
pip install -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt"

echo "Converting merged model to fp16 GGUF..."
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_MODEL" \
    --outfile "$GGUF_OUT_FILE" \
    --outtype f16

echo
echo "Conversion complete: $GGUF_OUT_FILE"
echo
echo "Optional next step: quantize this file to reduce size (see deployment/README.md),"
echo "or point deployment/Modelfile directly at this fp16 file and run:"
echo "  ollama create truck-dispatch-ai -f deployment/Modelfile"

#!/bin/sh
# Download a Llama-architecture model from Hugging Face (single-file safetensors).
# usage: scripts/download.sh [hf_repo] [dir]
set -e
repo=${1:-HuggingFaceTB/SmolLM2-135M-Instruct}
dir=${2:-models/smollm2-135m}
mkdir -p "$dir"
for f in config.json tokenizer.json model.safetensors; do
  curl -fL --progress-bar -o "$dir/$f" "https://huggingface.co/$repo/resolve/main/$f"
done

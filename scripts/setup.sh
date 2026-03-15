#!/usr/bin/env bash
# scripts/setup.sh — download ONNX Runtime Web into extension/lib/
set -euo pipefail

ORT_VERSION="${1:-1.21.0}"
LIB_DIR="$(dirname "$0")/../extension/lib"
mkdir -p "$LIB_DIR"

BASE="https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist"

echo "Downloading ONNX Runtime Web v${ORT_VERSION}…"
for f in ort.min.js ort-wasm.wasm ort-wasm-simd.wasm; do
    echo "  → $f"
    curl -sL "${BASE}/${f}" -o "${LIB_DIR}/${f}"
done

echo ""
echo "Done. Saved to ${LIB_DIR}/"

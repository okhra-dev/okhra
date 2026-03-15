#!/usr/bin/env bash
set -euo pipefail

ORT_VERSION="1.17.3"
LIB_DIR="$(dirname "$0")/../extension/lib"
rm -rf "$LIB_DIR"
mkdir -p "$LIB_DIR"

BASE="https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist"

for f in ort.min.js ort-wasm.wasm ort-wasm-simd.wasm ort-wasm-simd-threaded.wasm; do
    echo "Downloading $f…"
    curl -sL "${BASE}/${f}" -o "${LIB_DIR}/${f}"
done

echo "Done → ${LIB_DIR}/"
ls -lh "$LIB_DIR/"

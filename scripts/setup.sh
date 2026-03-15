#!/usr/bin/env bash
set -euo pipefail

ORT_VERSION="${1:-1.22.0}"
LIB_DIR="$(dirname "$0")/../extension/lib"
rm -rf "$LIB_DIR"
mkdir -p "$LIB_DIR"

echo "Downloading ONNX Runtime Web v${ORT_VERSION} from npm…"

TMP=$(mktemp -d)
cd "$TMP"
npm pack "onnxruntime-web@${ORT_VERSION}" --pack-destination .
tar xzf onnxruntime-web-*.tgz

cp package/dist/*.min.js      "$LIB_DIR/" 2>/dev/null || true
cp package/dist/*.min.mjs     "$LIB_DIR/" 2>/dev/null || true
cp package/dist/*.wasm        "$LIB_DIR/" 2>/dev/null || true
cp package/dist/*.mjs         "$LIB_DIR/" 2>/dev/null || true

cd - >/dev/null
rm -rf "$TMP"

echo ""
echo "Files saved to ${LIB_DIR}/:"
ls -lh "$LIB_DIR/"
echo ""
echo "Done."

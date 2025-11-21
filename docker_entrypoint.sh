#!/bin/bash
# Docker entrypoint script for m4t
# Downloads models if needed, then starts the server

set -e

echo "🚀 m4t Docker Container Starting..."
echo "   Version: v1.1.0-gptsovits"

# Check if SKIP_MODEL_DOWNLOAD is set
if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then
    echo ""
    echo "Step 1: Checking pretrained models..."
    /app/docker_download_models.sh
else
    echo ""
    echo "⏩ Skipping model download (SKIP_MODEL_DOWNLOAD=true)"
    echo "   Ensure models are mounted at /app/third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models"
fi

# Verify critical models exist
MODEL_DIR="/app/third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models"
if [ ! -f "$MODEL_DIR/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt" ]; then
    echo "❌ ERROR: GPT model not found!"
    echo "   Expected: $MODEL_DIR/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    echo "   Either:"
    echo "   1. Remove SKIP_MODEL_DOWNLOAD environment variable to auto-download"
    echo "   2. Mount pretrained models directory with -v"
    exit 1
fi

if [ ! -f "$MODEL_DIR/s2G488k.pth" ]; then
    echo "❌ ERROR: SoVITS model not found!"
    echo "   Expected: $MODEL_DIR/s2G488k.pth"
    exit 1
fi

echo ""
echo "✓ All models verified"
echo ""
echo "Step 2: Starting m4t API server..."
echo "   Listening on: 0.0.0.0:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo ""

# Execute the CMD (python server.py)
exec "$@"

#!/bin/bash
#
# Docker Build Script for m4t v1.1
# Includes GPT-SoVITS integration and seed parameter support
#

set -e  # Exit on error

echo "🔨 Building m4t Docker Image v1.1"
echo "================================="
echo ""
echo "Features in this build:"
echo "  ✅ GPT-SoVITS voice cloning integration"
echo "  ✅ Random seed parameter for reproducibility"
echo "  ✅ Automatic model downloading from HuggingFace"
echo "  ✅ Volume-mount support for pretrained models"
echo ""

cd /home/camus/work/m4t

# Set proxy for Docker build (match systemd configuration)
export HTTP_PROXY=http://127.0.0.1:1091
export HTTPS_PROXY=http://127.0.0.1:1091

# Clean up any existing build
echo "Cleaning up previous build attempts..."
rm -f /tmp/m4t_docker_build_v1.1.log

# Link log file
ln -sf /tmp/m4t_docker_build_v1.1.log /home/camus/work/diary-job/logs/m4t_docker_build_v1.1.log

echo "Starting build (this will take 10-15 minutes)..."
echo "Build log: /tmp/m4t_docker_build_v1.1.log"
echo "Linked to: logs/m4t_docker_build_v1.1.log"
echo ""

# Build Docker image with proxy settings
echo "ppio" | sudo -S docker build \
  --network host \
  --build-arg HTTP_PROXY=http://127.0.0.1:1091 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:1091 \
  -f Dockerfile \
  -t kllambda/m4t:v1.1 \
  -t kllambda/m4t:latest \
  . 2>&1 | tee /tmp/m4t_docker_build_v1.1.log

BUILD_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================="

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "Image tags:"
    echo "  - kllambda/m4t:v1.1"
    echo "  - kllambda/m4t:latest"
    echo ""
    echo "To verify:"
    echo "  sudo docker images | grep kllambda/m4t"
    echo ""
    echo "To run (auto-download models):"
    echo "  sudo docker run -d --name m4t-server --gpus all -p 8000:8000 kllambda/m4t:v1.1"
    echo ""
    echo "To run (with volume-mounted models):"
    echo "  sudo docker run -d --name m4t-server --gpus all -p 8000:8000 \\"
    echo "    -v ~/work/hf-GPT-SoVITS:/app/third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models \\"
    echo "    -e SKIP_MODEL_DOWNLOAD=true \\"
    echo "    kllambda/m4t:v1.1"
    echo ""
else
    echo "❌ Build failed with exit code: $BUILD_EXIT_CODE"
    echo ""
    echo "Check the log: /tmp/m4t_docker_build_v1.1.log"
    echo ""
    echo "Common issues:"
    echo "  - Network connectivity (check proxy at http://localhost:1091)"
    echo "  - Disk space (need ~10GB free)"
    echo "  - Docker daemon running"
    exit $BUILD_EXIT_CODE
fi

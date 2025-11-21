#!/bin/bash
# Build m4t Docker Image v1.1
# This script requires sudo password

echo "🔨 Building m4t Docker Image v1.1"
echo "   This build includes:"
echo "   - GPT-SoVITS integration with automatic model downloading"
echo "   - Random seed parameter for voice cloning reproducibility"
echo "   - Volume-mount support for pretrained models"
echo ""
echo "Building..."

cd /home/camus/work/m4t

sudo docker build \
  --network host \
  -f Dockerfile \
  -t kllambda/m4t:v1.1 \
  -t kllambda/m4t:latest \
  . 2>&1 | tee /tmp/m4t_docker_build_v1.1.log

if [ $? -eq 0 ]; then
    echo ""
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
    echo "Build log: /tmp/m4t_docker_build_v1.1.log"
else
    echo ""
    echo "❌ Build failed!"
    echo "Check the log: /tmp/m4t_docker_build_v1.1.log"
    exit 1
fi

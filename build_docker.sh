#!/bin/bash
# Build m4t Docker image with GPT-SoVITS integration
# Usage: ./build_docker.sh [version]

set -e

VERSION="${1:-v1.1.0}"
IMAGE_NAME="kllambda/m4t"

echo "🔨 Building m4t Docker Image"
echo "   Version: $VERSION"
echo "   Image: $IMAGE_NAME:$VERSION"
echo ""

cd "$(dirname "$0")"

# Build the image (with sudo if needed)
if groups | grep -q docker; then
    echo "Building with docker (user in docker group)..."
    docker build \
        --network host \
        -t "$IMAGE_NAME:$VERSION" \
        -t "$IMAGE_NAME:latest" \
        .
else
    echo "Building with sudo docker (user not in docker group)..."
    echo "You may be prompted for your password"
    sudo -S docker build \
        --network host \
        -t "$IMAGE_NAME:$VERSION" \
        -t "$IMAGE_NAME:latest" \
        .
fi

echo ""
echo "✅ Build complete!"
echo "   Tagged as: $IMAGE_NAME:$VERSION"
echo "   Tagged as: $IMAGE_NAME:latest"
echo ""
echo "To run the container:"
echo "  docker run -d --name m4t-server --gpus all -p 8000:8000 $IMAGE_NAME:$VERSION"
echo ""
echo "Or with volume-mounted models (skip download):"
echo "  docker run -d --name m4t-server --gpus all -p 8000:8000 \\"
echo "    -v /path/to/pretrained_models:/app/third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models \\"
echo "    -e SKIP_MODEL_DOWNLOAD=true \\"
echo "    $IMAGE_NAME:$VERSION"

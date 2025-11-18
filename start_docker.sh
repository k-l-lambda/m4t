#!/bin/bash
# Build and start SeamlessM4T API server with Docker

echo "Building and Starting SeamlessM4T API Server (Docker)"
echo "====================================================="

# Build Docker image
echo "Building Docker image..."
docker build \
    --build-arg HTTP_PROXY=http://host.docker.internal:1091 \
    --build-arg HTTPS_PROXY=http://host.docker.internal:1091 \
    -t seamless-m4t-api:latest \
    .

# Stop and remove existing container if running
if [ "$(docker ps -aq -f name=seamless-m4t-api)" ]; then
    echo "Stopping existing container..."
    docker stop seamless-m4t-api
    docker rm seamless-m4t-api
fi

# Start container
echo "Starting container..."
docker run -d \
    --name seamless-m4t-api \
    --gpus '"device=0"' \
    -p 8000:8000 \
    -e HTTP_PROXY=http://host.docker.internal:1091 \
    -e HTTPS_PROXY=http://host.docker.internal:1091 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    seamless-m4t-api:latest

echo ""
echo "✅ Container started!"
echo "API URL: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "View logs: docker logs -f seamless-m4t-api"
echo "Stop server: docker stop seamless-m4t-api"

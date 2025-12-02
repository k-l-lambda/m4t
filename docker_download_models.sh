#!/bin/bash
# Download GPT-SoVITS pretrained models from HuggingFace
# This script is called by docker_entrypoint.sh at container startup

set -e

MODEL_DIR="/app/third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models"
HF_REPO="k-l-lambda/GPT-SoVITS-pretrained-models"

echo "🔍 Checking for pretrained models..."

# Check if models are already present (via volume mount or previous download)
if [ -f "$MODEL_DIR/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt" ] && \
   [ -f "$MODEL_DIR/s2G488k.pth" ] && \
   [ -f "$MODEL_DIR/s1v3.ckpt" ] && \
   [ -f "$MODEL_DIR/s2Gv3.pth" ] && \
   [ -d "$MODEL_DIR/chinese-hubert-base" ] && \
   [ -d "$MODEL_DIR/chinese-roberta-wwm-ext-large" ]; then
    echo "✓ All required models found (v1 + v3), skipping download"
    exit 0
fi

echo "📥 Downloading pretrained models from HuggingFace..."
echo "   Repository: $HF_REPO"
echo "   This may take several minutes (models are ~1.2 GB total)"

# Install huggingface-cli if not present
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-cli..."
    pip install -q huggingface-hub
fi

# Download models using huggingface-cli
cd "$MODEL_DIR"

# Core models (required)
echo "Downloading GPT model (s1bert25hz)..."
huggingface-cli download "$HF_REPO" \
    s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt \
    --local-dir . --local-dir-use-symlinks False

echo "Downloading SoVITS generator model (s2G488k)..."
huggingface-cli download "$HF_REPO" \
    s2G488k.pth \
    --local-dir . --local-dir-use-symlinks False

echo "Downloading SoVITS discriminator model (s2D488k)..."
huggingface-cli download "$HF_REPO" \
    s2D488k.pth \
    --local-dir . --local-dir-use-symlinks False

# Chinese HuBERT model
echo "Downloading Chinese HuBERT model..."
huggingface-cli download "$HF_REPO" \
    --include "chinese-hubert-base/*" \
    --local-dir . --local-dir-use-symlinks False

# Chinese RoBERTa model
echo "Downloading Chinese RoBERTa model..."
huggingface-cli download "$HF_REPO" \
    --include "chinese-roberta-wwm-ext-large/*" \
    --local-dir . --local-dir-use-symlinks False

# V3 models (mandatory for improved voice cloning)
echo "Downloading V3 GPT model (s1v3)..."
huggingface-cli download "$HF_REPO" \
    s1v3.ckpt \
    --local-dir . --local-dir-use-symlinks False

echo "Downloading V3 SoVITS model (s2Gv3)..."
huggingface-cli download "$HF_REPO" \
    s2Gv3.pth \
    --local-dir . --local-dir-use-symlinks False

# Download G2PW ONNX model (required for v3)
echo "Downloading G2PW ONNX model for v3 text processing..."
G2PW_DIR="/app/third_party/GPT-SoVITS/GPT_SoVITS/text/G2PWModel"
mkdir -p "$G2PW_DIR"
cd "$G2PW_DIR"
if [ ! -f "G2PWModel_1.1.onnx" ]; then
    # Download from GitHub release
    curl -L -o G2PWModel_1.1.zip \
        "https://github.com/Artrajz/G2PWModel/releases/download/1.1/G2PWModel_1.1.zip" || \
    curl -L -o G2PWModel_1.1.zip \
        "https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip"

    # Extract
    unzip -q G2PWModel_1.1.zip
    rm G2PWModel_1.1.zip
    echo "✓ G2PW model downloaded and extracted"
else
    echo "✓ G2PW model already exists"
fi

cd "$MODEL_DIR"

echo "✅ Model download complete!"
echo "   Models saved to: $MODEL_DIR"

# List downloaded files for verification
echo ""
echo "Downloaded files:"
du -sh "$MODEL_DIR"
ls -lh "$MODEL_DIR" | grep -E "\.ckpt|\.pth|^d"

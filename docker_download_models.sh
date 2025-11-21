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
   [ -d "$MODEL_DIR/chinese-hubert-base" ] && \
   [ -d "$MODEL_DIR/chinese-roberta-wwm-ext-large" ]; then
    echo "✓ All required models found, skipping download"
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

# Optional: V3 models (if available)
if huggingface-cli download "$HF_REPO" s1v3.ckpt --local-dir . --local-dir-use-symlinks False 2>/dev/null; then
    echo "✓ Downloaded V3 GPT model"
fi

if huggingface-cli download "$HF_REPO" s2Gv3.pth --local-dir . --local-dir-use-symlinks False 2>/dev/null; then
    echo "✓ Downloaded V3 SoVITS model"
fi

echo "✅ Model download complete!"
echo "   Models saved to: $MODEL_DIR"

# List downloaded files for verification
echo ""
echo "Downloaded files:"
du -sh "$MODEL_DIR"
ls -lh "$MODEL_DIR" | grep -E "\.ckpt|\.pth|^d"

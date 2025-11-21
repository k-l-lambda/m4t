# m4t Docker Integration - GPT-SoVITS Support

## Overview

Updated m4t Dockerfile (v1.1.0) to include GPT-SoVITS voice cloning support with automatic model downloading capabilities.

## Changes Made

### 1. Updated Dockerfile

**New features**:
- Clones GPT-SoVITS repository from GitHub during build
- Installs GPT-SoVITS dependencies
- Includes all m4t modules (gptsovits_local.py, voice_cloner.py, etc.)
- Adds entrypoint script for model management
- Supports both auto-download and volume-mounting of pretrained models

**Key sections**:
```dockerfile
# Clone GPT-SoVITS repository
RUN git clone https://github.com/k-l-lambda/GPT-SoVITS.git /app/third_party/GPT-SoVITS

# Install GPT-SoVITS dependencies
RUN pip install --no-cache-dir -r /app/third_party/GPT-SoVITS/requirements.txt

# Copy all necessary m4t modules
COPY config_m4t.py models.py server.py gptsovits_local.py voice_cloner.py voice_detector.py audio_separator.py env_loader.py ./

# Add model download and entrypoint scripts
COPY docker_download_models.sh docker_entrypoint.sh /app/
RUN chmod +x /app/docker_download_models.sh /app/docker_entrypoint.sh

ENTRYPOINT ["/app/docker_entrypoint.sh"]
CMD ["python", "server.py"]
```

### 2. Created docker_download_models.sh

**Purpose**: Downloads GPT-SoVITS pretrained models from HuggingFace

**Features**:
- Checks if models already exist (skip download if present)
- Downloads from HuggingFace repository: `k-l-lambda/GPT-SoVITS-pretrained-models`
- Installs huggingface-cli if needed
- Downloads core models (s1bert25hz, s2G488k, s2D488k, chinese-hubert-base, chinese-roberta-wwm-ext-large)
- Optionally downloads v3 models
- Provides progress feedback and verification

**Models downloaded** (~1.2 GB total):
- GPT model: s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt (~155 MB)
- SoVITS generator: s2G488k.pth (~106 MB)
- SoVITS discriminator: s2D488k.pth (~94 MB)
- Chinese HuBERT model
- Chinese RoBERTa model
- Optional v3 models (s1v3.ckpt, s2Gv3.pth)

### 3. Created docker_entrypoint.sh

**Purpose**: Container startup script that manages model downloading

**Workflow**:
1. Checks `SKIP_MODEL_DOWNLOAD` environment variable
2. If not set, calls docker_download_models.sh
3. Verifies critical models exist
4. Starts m4t API server (python server.py)

**Error handling**:
- Validates model files before starting server
- Provides clear error messages if models missing
- Suggests solutions (remove SKIP_MODEL_DOWNLOAD or mount volume)

### 4. Created build_docker.sh

**Purpose**: Convenient build script with sudo handling

**Features**:
- Detects if user is in docker group
- Uses sudo if needed (with -S flag for password input)
- Tags image with version and 'latest'
- Provides usage examples after build

**Usage**:
```bash
./build_docker.sh          # Builds v1.1.0 (default)
./build_docker.sh v1.2.0   # Builds custom version
```

### 5. Created .dockerignore

**Purpose**: Exclude unnecessary files from Docker build context

**Excluded**:
- Python cache (`__pycache__/`, `*.pyc`)
- Environment files (`.env`, `.env.local`)
- Git files (`.git/`, `.gitignore`)
- Third-party directory (cloned fresh in Dockerfile)
- Test files and examples
- Development scripts
- Logs

### 6. Updated README.md

**New sections**:
- Docker deployment options (v1.1.0)
- Method 1: Auto-download models (easiest)
- Method 2: Volume-mount models (faster startup)
- Method 3: Docker Compose (legacy, without GPT-SoVITS)
- Model files documentation
- Build and run instructions

## Usage

### Option 1: Auto-download Models (Recommended)

```bash
# Build image
./build_docker.sh v1.1.0

# Run container (models download on first start)
docker run -d --name m4t-server \
  --gpus all \
  -p 8000:8000 \
  kllambda/m4t:v1.1.0

# Monitor first startup (5-10 minutes for model download)
docker logs -f m4t-server
```

**Pros**:
- Easiest setup
- No manual model management
- Models cached in container for subsequent starts

**Cons**:
- First startup takes 5-10 minutes
- Requires internet connection on first start
- Models stored in container (not shared across containers)

### Option 2: Volume-mount Models (Production)

```bash
# Build image
./build_docker.sh v1.1.0

# Run with volume-mounted models
docker run -d --name m4t-server \
  --gpus all \
  -p 8000:8000 \
  -v ~/work/hf-GPT-SoVITS:/app/third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models \
  -e SKIP_MODEL_DOWNLOAD=true \
  kllambda/m4t:v1.1.0

# Instant startup
docker logs -f m4t-server
```

**Pros**:
- Instant startup (no model download)
- Models shared across containers
- Models persist across container recreations
- No internet required at runtime

**Cons**:
- Requires pre-downloaded models on host
- Need to manage model directory manually

## Testing

### Build Test

```bash
cd /home/camus/work/m4t
./build_docker.sh v1.1.0
```

Expected output:
- Docker build completes successfully
- Image tagged as `kllambda/m4t:v1.1.0` and `kllambda/m4t:latest`
- Build includes GPT-SoVITS and all dependencies

### Runtime Test (Auto-download)

```bash
# Start container with auto-download
docker run -d --name m4t-test \
  --gpus all \
  -p 8001:8000 \
  kllambda/m4t:v1.1.0

# Monitor logs
docker logs -f m4t-test

# Expected output:
# 🚀 m4t Docker Container Starting...
# 🔍 Checking for pretrained models...
# 📥 Downloading pretrained models from HuggingFace...
# ✅ Model download complete!
# ✓ All models verified
# Starting m4t API server...
```

### Runtime Test (Volume-mount)

```bash
# Start container with volume mount
docker run -d --name m4t-test-volume \
  --gpus all \
  -p 8002:8000 \
  -v ~/work/hf-GPT-SoVITS:/app/third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models \
  -e SKIP_MODEL_DOWNLOAD=true \
  kllambda/m4t:v1.1.0

# Monitor logs
docker logs -f m4t-test-volume

# Expected output:
# 🚀 m4t Docker Container Starting...
# ⏩ Skipping model download (SKIP_MODEL_DOWNLOAD=true)
# ✓ All models verified
# Starting m4t API server...
```

### API Test

```bash
# Test health endpoint
curl http://localhost:8001/health

# Test voice cloning endpoint
curl -s -X POST "http://localhost:8001/v1/voice-clone" \
  -F "audio=@reference.wav" \
  -F "text=Hello, this is a test." \
  -F "text_language=eng" \
  -F "prompt_text=Reference text" \
  -F "prompt_language=eng" \
  | jq -r '.output_audio_base64' | base64 -d > output.wav
```

## Files Created/Modified

**Created**:
- `/home/camus/work/m4t/docker_download_models.sh` - Model download script
- `/home/camus/work/m4t/docker_entrypoint.sh` - Container entrypoint
- `/home/camus/work/m4t/build_docker.sh` - Build convenience script
- `/home/camus/work/m4t/.dockerignore` - Docker build exclusions

**Modified**:
- `/home/camus/work/m4t/Dockerfile` - Added GPT-SoVITS integration
- `/home/camus/work/m4t/README.md` - Updated Docker documentation

## Version History

- **v1.0.0**: Original m4t Docker image (SeamlessM4T only)
- **v1.1.0**: Added GPT-SoVITS integration with automatic model downloading

## Future Enhancements

**Potential improvements**:
1. Create HuggingFace repository with pretrained models
2. Add multi-stage build to reduce image size
3. Support for custom model paths via environment variables
4. Add health check that verifies models loaded successfully
5. Create Docker Hub automated builds
6. Add docker-compose.yml for v1.1.0
7. Support for GPU selection (specific device IDs)

## Notes

- The container requires NVIDIA GPU with CUDA support
- GPT-SoVITS models are ~1.2 GB total
- First startup with auto-download takes 5-10 minutes
- Subsequent starts are instant (models cached)
- Volume mounting is recommended for production deployments

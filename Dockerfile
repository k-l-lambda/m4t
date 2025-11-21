FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set timezone non-interactively to prevent build hang
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .

# Set proxy for pip (if needed)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

# Install m4t Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone GPT-SoVITS repository
RUN git clone https://github.com/k-l-lambda/GPT-SoVITS.git /app/third_party/GPT-SoVITS

# Install GPT-SoVITS dependencies
RUN pip install --no-cache-dir -r /app/third_party/GPT-SoVITS/requirements.txt

# Copy application code
COPY config_m4t.py models.py server.py gptsovits_local.py voice_cloner.py voice_detector.py audio_separator.py env_loader.py ./

# Create directory for pretrained models
RUN mkdir -p /app/third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models

# Copy model download script
COPY docker_download_models.sh /app/

# Make the script executable
RUN chmod +x /app/docker_download_models.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint script that downloads models if needed, then starts server
COPY docker_entrypoint.sh /app/
RUN chmod +x /app/docker_entrypoint.sh

ENTRYPOINT ["/app/docker_entrypoint.sh"]
CMD ["python", "server.py"]

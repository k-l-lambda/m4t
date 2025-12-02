# SeamlessM4T Inference API

Multilingual speech and text translation API using Meta's **SeamlessM4T v2** model.

## Features

- **9 Translation & Speech Tasks:**
  - 🎤→📝 **S2TT**: Speech-to-Text Translation (e.g., Japanese audio → Chinese text)
  - 🎤→🔊 **S2ST**: Speech-to-Speech Translation (e.g., Japanese audio → Chinese audio)
  - 🎤→📝 **ASR**: Automatic Speech Recognition (e.g., Japanese audio → Japanese text)
  - 📝→📝 **T2TT**: Text-to-Text Translation (e.g., English text → Chinese text)
  - 📝→🔊 **TTS**: Text-to-Speech (e.g., Chinese text → Chinese audio)
  - 🎙️ **VAD**: Voice Activity Detection (detect speech segments in audio)
  - 🎵 **Vocal Separation**: Extract vocals from background music (optional Spleeter)
  - 🎭 **Voice Cloning**: Clone speaker voice with GPT-SoVITS (direct Python integration)
  - 🎼 **Audio Split**: Split audio into vocals + accompaniment (two separate streams)

- **Wide Language Support:** 101 languages for speech, 96 for text
- **High Quality:** 2.3B parameter model with state-of-the-art translation quality
- **Easy Deployment:** Standalone Python or Docker container
- **RESTful API:** FastAPI with automatic OpenAPI documentation

## Quick Start

### Option 1: Development Mode (Standalone Python)

```bash
# Start server
./start_dev.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

### Option 2: Docker (with GPT-SoVITS integrated)

**Latest version: v1.1.0** includes GPT-SoVITS voice cloning support built-in.

#### Method 1: Auto-download models (easiest)

```bash
# Build image with integrated GPT-SoVITS
./build_docker.sh v1.1.0

# Run container (models will download automatically on first start)
docker run -d --name m4t-server \
  --gpus all \
  -p 8000:8000 \
  kllambda/m4t:v1.1.0

# First startup takes ~5-10 minutes to download models (~1.2 GB)
# Subsequent starts are instant as models are cached in container
docker logs -f m4t-server
```

#### Method 2: Volume-mount pretrained models (faster startup)

```bash
# If you already have pretrained models on host:
docker run -d --name m4t-server \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/pretrained_models:/app/third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models \
  -e SKIP_MODEL_DOWNLOAD=true \
  kllambda/m4t:v1.1.0

# Example: Mount from host system
# -v ~/work/hf-GPT-SoVITS:/app/third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models
```

#### Method 3: Using Docker Compose (legacy)

```bash
# Using Docker Compose (without GPT-SoVITS)
docker-compose up -d

# Or using the script
./start_docker.sh
```

#### Model Files

The container includes both v1 and **v3 models** (~2.0 GB total):

**v1 models** (baseline):
- `s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt` (~155 MB) - GPT v1 model
- `s2G488k.pth` (~106 MB) - SoVITS v1 generator
- `s2D488k.pth` (~94 MB) - SoVITS v1 discriminator

**v3 models** (improved quality, included by default):
- `s1v3.ckpt` (~149 MB) - GPT v3 model with better duration control
- `s2Gv3.pth` (~734 MB) - SoVITS v3 generator (7x larger, higher quality)
- `G2PWModel/` (~562 MB) - G2PW ONNX model for advanced text processing

**Shared models**:
- `chinese-hubert-base/` - Chinese HuBERT model
- `chinese-roberta-wwm-ext-large/` - Chinese RoBERTa model for tokenization

**v3 Model Advantages**:
- Better voice quality and naturalness
- Improved pronunciation with ML-powered G2PW text processing
- More stable voice characteristics across different texts
- Better handling of Chinese text (tone and pronunciation)

Models are automatically downloaded from HuggingFace (`k-l-lambda/GPT-SoVITS-pretrained-models`) on first container start, or can be volume-mounted to skip download.

## Configuration

The server can be configured using environment variables or a `.env.local` file in the project root.

### Configuration File (.env.local)

Create a `.env.local` file to customize server settings:

```bash
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# Audio and Text Limits
MAX_AUDIO_LENGTH=300  # seconds
MAX_TEXT_LENGTH=2000  # characters

# GPT-SoVITS Configuration (for voice cloning)
GPTSOVITS_API_URL=http://localhost:9880

# Proxy Configuration (for model downloads)
HTTP_PROXY=http://localhost:1091
HTTPS_PROXY=http://localhost:1091

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Environment Variables

All configuration options can also be set via environment variables:

```bash
# Change server port
export SERVER_PORT=9000
python server.py

# Or inline
SERVER_PORT=9000 python server.py
```

### Restart Script

A convenience script is provided to restart the server with proper cache clearing:

```bash
./restart.sh
```

This script will:
- Stop existing server processes
- Clear Python cache
- Start server with nohup (runs in background)
- Wait for server readiness
- Check GPT-SoVITS availability
- Display server status and configuration

## API Endpoints

### Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Interactive API documentation (Swagger UI) |
| `/health` | GET | Health check |
| `/languages` | GET | List supported languages |
| `/tasks` | GET | List supported tasks |
| `/v1/speech-to-text-translation` | POST | Translate speech to text (S2TT) |
| `/v1/speech-to-speech-translation` | POST | Translate speech to speech (S2ST) |
| `/v1/transcribe` | POST | Transcribe speech (ASR) |
| `/v1/text-to-text-translation` | POST | Translate text to text (T2TT) |
| `/v1/text-to-speech` | POST | Convert text to speech (TTS) |
| `/v1/detect-voice` | POST | Detect speech segments (VAD) |
| `/v1/separate-vocals` | POST | Separate vocals from music (requires Spleeter) |

## Usage Examples

### 1. Text-to-Text Translation (T2TT)

```bash
curl -X POST "http://localhost:8000/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "こんにちは、今日は良い天気ですね。",
    "source_lang": "jpn",
    "target_lang": "cmn"
  }'
```

**Response:**
```json
{
  "task": "t2tt",
  "source_language": "jpn",
  "target_language": "cmn",
  "input_text": "こんにちは、今日は良い天気ですね。",
  "output_text": "你好，今天天气真好。",
  "processing_time": 0.45
}
```

### 2. Speech-to-Text Translation (S2TT)

```bash
curl -X POST "http://localhost:8000/v1/speech-to-text-translation" \
  -F "audio=@japanese_audio.wav" \
  -F "target_lang=cmn" \
  -F "source_lang=jpn"
```

**Response:**
```json
{
  "task": "s2tt",
  "source_language": "jpn",
  "target_language": "cmn",
  "input_duration": 5.2,
  "output_text": "你好，今天天气真好。",
  "processing_time": 1.85
}
```

### 3. Transcription (ASR)

```bash
curl -X POST "http://localhost:8000/v1/transcribe" \
  -F "audio=@japanese_audio.wav" \
  -F "language=jpn"
```

### 4. Speech-to-Speech Translation (S2ST)

```bash
# Get audio file directly
curl -X POST "http://localhost:8000/v1/speech-to-speech-translation" \
  -F "audio=@japanese_audio.wav" \
  -F "target_lang=cmn" \
  -F "source_lang=jpn" \
  -F "response_format=audio" \
  -o translated_audio.wav

# Or get JSON with base64-encoded audio
curl -X POST "http://localhost:8000/v1/speech-to-speech-translation" \
  -F "audio=@japanese_audio.wav" \
  -F "target_lang=cmn" \
  -F "response_format=json"
```

### 5. Text-to-Speech (TTS)

```bash
curl -X POST "http://localhost:8000/v1/text-to-speech" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，今天天气很好",
    "source_lang": "cmn"
  }'
```

**Response:**
```json
{
  "task": "tts",
  "language": "cmn",
  "input_text": "你好，今天天气很好",
  "output_audio": [0.001, -0.002, 0.003, ...],
  "output_sample_rate": 16000,
  "processing_time": 9.11
}
```

**Save audio to file (Python):**
```python
import requests
import numpy as np
import soundfile as sf

response = requests.post(
    "http://localhost:8000/v1/text-to-speech",
    json={
        "text": "Hello, how are you today?",
        "source_lang": "eng"
    }
)

result = response.json()
audio_array = np.array(result['output_audio'], dtype=np.float32)
sample_rate = result['output_sample_rate']

# Save to WAV file
sf.write('output_speech.wav', audio_array, sample_rate)
```

### 6. Voice Activity Detection (VAD)

Detect speech segments in audio files with precise timestamps.

**Basic usage:**
```bash
curl -X POST "http://localhost:8000/v1/detect-voice" \
  -F "audio=@audio_file.wav"
```

**With parameters:**
```bash
curl -X POST "http://localhost:8000/v1/detect-voice" \
  -F "audio=@audio_file.wav" \
  -F "threshold=0.5" \
  -F "min_speech_duration_ms=250" \
  -F "min_silence_duration_ms=300"
```

**Parameters:**
- `threshold` (float, default: 0.5): Speech detection threshold (0.0-1.0). Lower = more sensitive
- `min_speech_duration_ms` (int, default: 250): Minimum speech segment duration in milliseconds
- `min_silence_duration_ms` (int, default: 300): Minimum silence duration between segments

**Response:**
```json
{
  "task": "vad",
  "total_duration": 3.18,
  "speech_segments": [
    {
      "start": 0.258,
      "end": 2.846,
      "duration": 2.588
    }
  ],
  "segment_count": 1,
  "total_speech_duration": 2.588,
  "processing_time": 0.064
}
```

**Use cases:**
- Intelligent audio segmentation for long recordings
- Silence removal for efficient processing
- Speech quality analysis
- Preprocessing for translation workflows

**Performance:** ~40-50x faster than real-time (3s audio processed in 0.06s)

### 7. Vocal Separation (Audio Preprocessing)

Separate vocals from background music using Spleeter. Useful for improving speech recognition/translation quality on audio with music.

**Note:** Requires Spleeter installation: `pip install spleeter`

```bash
curl -X POST "http://localhost:8000/v1/separate-vocals" \
  -F "audio=@audio_with_music.wav" \
  | python3 -c "import sys, json, base64; d=json.load(sys.stdin); open('vocals.wav','wb').write(base64.b64decode(d['vocals_audio_base64']))"
```

**Response:**
```json
{
  "task": "separate",
  "input_duration": 5.2,
  "vocals_audio_base64": "UklGRiQAAABXQVZFZm10...",
  "sample_rate": 16000,
  "processing_time": 3.45,
  "separator_available": true
}
```

**Optional preprocessing for translation:**

You can automatically separate vocals before translation by adding `separate_vocals=true`:

```bash
# Speech-to-text translation with vocal separation
curl -X POST "http://localhost:8000/v1/speech-to-text-translation" \
  -F "audio=@audio_with_music.wav" \
  -F "target_lang=cmn" \
  -F "source_lang=jpn" \
  -F "separate_vocals=true"

# Speech-to-speech translation with vocal separation
curl -X POST "http://localhost:8000/v1/speech-to-speech-translation" \
  -F "audio=@audio_with_music.wav" \
  -F "target_lang=cmn" \
  -F "separate_vocals=true" \
  -F "response_format=audio" \
  -o translated_vocals.wav
```

If Spleeter is not installed, the parameter is ignored and processing continues without separation (with a warning).

### 8. Audio Split (Source Separation)

**Endpoint:** `POST /v1/audio-split`

Split audio into two separate streams: vocals and accompaniment (background music).

**Use cases:**
- Extract clean vocals for further processing
- Create karaoke versions (accompaniment only)
- Remix or mashup production
- Audio analysis and research

**Features:**
- ✅ Automatic chunked processing for long audio files (>10 minutes)
- ✅ No duration limit - processes audio of any length
- ✅ GPU-accelerated separation using Spleeter's 2-stem model

**Note:** Requires Spleeter installation: `pip install spleeter`

**Basic usage:**

```bash
# Split audio and save both streams
curl -X POST "http://localhost:8000/v1/audio-split" \
  -F "audio=@song.wav" \
  -o output.json

# Extract vocals only
jq -r '.vocals_audio_base64' output.json | base64 -d > vocals.wav

# Extract accompaniment only
jq -r '.accompaniment_audio_base64' output.json | base64 -d > accompaniment.wav

# Extract both streams in one command
cat output.json | jq -r '.vocals_audio_base64' | base64 -d > vocals.wav && \
cat output.json | jq -r '.accompaniment_audio_base64' | base64 -d > accompaniment.wav
```

**One-liner to extract both streams:**

```bash
curl -s -X POST "http://localhost:8000/v1/audio-split" \
  -F "audio=@song.wav" \
  | tee >(jq -r '.vocals_audio_base64' | base64 -d > vocals.wav) \
  | jq -r '.accompaniment_audio_base64' | base64 -d > accompaniment.wav
```

**Response:**

```json
{
  "task": "audio_split",
  "input_duration": 5.2,
  "vocals_audio_base64": "UklGRiQAAABXQVZFZm10...",
  "accompaniment_audio_base64": "UklGRkgBAABXQVZFZm10...",
  "sample_rate": 16000,
  "processing_time": 3.45,
  "separator_available": true
}
```

**Python example:**

```python
import requests
import base64

# Split audio
with open("song.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/audio-split",
        files={"audio": f}
    )

result = response.json()

# Save vocals
with open("vocals.wav", "wb") as f:
    f.write(base64.b64decode(result['vocals_audio_base64']))

# Save accompaniment
with open("accompaniment.wav", "wb") as f:
    f.write(base64.b64decode(result['accompaniment_audio_base64']))

print(f"Separated {result['input_duration']:.2f}s audio in {result['processing_time']:.2f}s")
```

**Performance:**
- Short audio (<10 min): ~0.6-0.7x real-time (5s audio processed in 3.5s on GPU)
- Long audio (>10 min): Processed in 5-minute chunks with automatic concatenation

**Difference from `/v1/separate-vocals`:**
- `/v1/separate-vocals`: Returns vocals only (single stream)
- `/v1/audio-split`: Returns BOTH vocals and accompaniment (two streams)

### 9. Voice Cloning (GPT-SoVITS)

**Endpoint:** `POST /v1/voice-clone`

Clone a speaker's voice from a reference audio and generate speech with the same voice characteristics.

**Features:**
- Direct Python integration (no external service needed)
- Supports multiple languages: Chinese (zh), English (en), Japanese (ja), Korean (ko), etc.
- **Automatic language code mapping**: Accepts both SeamlessM4T codes (eng, cmn, jpn) and GPT-SoVITS codes (en, zh, ja)
- High-quality voice cloning using GPT-SoVITS
- Auto-downloads language detection models on first use

**Installation:**

The voice cloning feature requires additional dependencies:

```bash
# Install GPT-SoVITS dependencies
cd /home/camus/work/m4t
./env/bin/pip install cn2an num2words eng_to_ipa fugashi[unidic-lite] unidic-lite

# GPT-SoVITS models are already included in third_party/GPT-SoVITS/
```

**First-time setup:**
On first use with Chinese text, fast-langdetect will automatically download a 126MB language detection model. This only happens once.

**Example (save directly to WAV file):**

```bash
# Using SeamlessM4T language codes (eng, cmn, jpn)
curl -s -X POST "http://localhost:8000/v1/voice-clone" \
  -F "audio=@reference_audio.wav" \
  -F "text=Hello, this is a voice cloning test." \
  -F "text_language=eng" \
  -F "prompt_text=Original text from reference audio" \
  -F "prompt_language=eng" \
  | jq -r '.output_audio_base64' | base64 -d > cloned_voice.wav

# Using GPT-SoVITS language codes (en, zh, ja) - also supported
curl -s -X POST "http://localhost:8000/v1/voice-clone" \
  -F "audio=@reference_audio.wav" \
  -F "text=Hello, this is a voice cloning test." \
  -F "text_language=en" \
  -F "prompt_text=Original text from reference audio" \
  -F "prompt_language=en" \
  | jq -r '.output_audio_base64' | base64 -d > cloned_voice.wav

# Chinese text with English reference audio
curl -s -X POST "http://localhost:8000/v1/voice-clone" \
  -F "audio=@reference_audio.wav" \
  -F "text=你好，这是一个中文语音克隆测试。" \
  -F "text_language=cmn" \
  -F "prompt_text=Original English text from reference" \
  -F "prompt_language=eng" \
  | jq -r '.output_audio_base64' | base64 -d > chinese_cloned.wav

# With fixed seed for reproducible results
curl -s -X POST "http://localhost:8000/v1/voice-clone" \
  -F "audio=@reference_audio.wav" \
  -F "text=Hello, this is a test." \
  -F "text_language=eng" \
  -F "prompt_text=Original text from reference audio" \
  -F "prompt_language=eng" \
  -F "seed=42" \
  | jq -r '.output_audio_base64' | base64 -d > reproducible_voice.wav
```

**Parameters:**
- `audio`: Reference audio file (WAV format recommended, 5-30 seconds)
- `text`: Text to synthesize in the target language
- `text_language`: Language code - Supports both:
  - SeamlessM4T codes: `eng` (English), `cmn` (Chinese), `jpn` (Japanese), `kor` (Korean)
  - GPT-SoVITS codes: `en` (English), `zh` (Chinese), `ja` (Japanese), `ko` (Korean)
- `prompt_text`: Transcription of the reference audio (what is being said)
- `prompt_language`: Language of the reference audio (supports both code formats)
- `cut_punc` (optional): Punctuation for text segmentation
- `seed` (optional): Random seed for reproducibility
  - `-1` (default): Random generation (different result each time)
  - `0-1000000`: Fixed seed for reproducible results (same result with same seed)

**Response:**
```json
{
  "task": "voice_clone",
  "output_audio_base64": "UklGRiQAAABXQVZFZm10...",
  "output_sample_rate": 32000,
  "text_length": 35,
  "output_duration": 3.5,
  "processing_time": 2.1,
  "service_available": true
}
```

**Performance:**
- English text: ~1-2 seconds processing time
- Chinese text (first time): ~28 seconds (includes model download)
- Chinese text (subsequent): ~1-2 seconds
- Output: 32kHz mono WAV audio

**Notes:**
- Reference audio should be clear and noise-free for best results
- Longer reference audio (10-30 seconds) generally produces better quality
- The cloned voice will maintain the speaker's characteristics but speak the new text
- GPU recommended for faster processing (uses ~4GB VRAM)
- **Reproducibility**: Set `seed` parameter to a fixed value (e.g., 42) for reproducible results across multiple runs
  - Useful for A/B testing, debugging, or when consistent output is needed
  - Use default `-1` for natural variety in voice generation

### Python Client Example

```python
import requests
import numpy as np
import soundfile as sf

# Text translation
response = requests.post(
    "http://localhost:8000/v1/text-to-text-translation",
    json={
        "text": "Hello",
        "source_lang": "eng",
        "target_lang": "cmn"
    }
)
print(response.json()["output_text"])

# Speech translation
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/speech-to-text-translation",
        files={"audio": f},
        data={"target_lang": "cmn", "source_lang": "jpn"}
    )
print(response.json()["output_text"])

# Text-to-Speech
response = requests.post(
    "http://localhost:8000/v1/text-to-speech",
    json={
        "text": "你好，今天天气很好",
        "source_lang": "cmn"
    }
)
result = response.json()
audio_array = np.array(result['output_audio'], dtype=np.float32)
sf.write('chinese_speech.wav', audio_array, result['output_sample_rate'])
```

**Full example script:** See `tts_example.py` for complete TTS usage examples.

## Supported Languages

Common language codes:

| Code | Language |
|------|----------|
| `jpn` | Japanese |
| `cmn` | Chinese (Simplified) |
| `cmn_Hant` | Chinese (Traditional) |
| `yue` | Cantonese |
| `kor` | Korean |
| `eng` | English |
| `fra` | French |
| `deu` | German |
| `spa` | Spanish |
| `rus` | Russian |
| `ara` | Arabic |
| `hin` | Hindi |
| `tha` | Thai |
| `vie` | Vietnamese |

**Full list:** GET `/languages` or check `config.py`

## System Requirements

### Hardware
- **GPU:** Recommended (NVIDIA with 24GB+ VRAM)
  - Model uses ~7GB VRAM in FP16
  - Can run on CPU but much slower
- **RAM:** 16GB+ recommended
- **Disk:** 10GB+ (for model cache)

### Software
- Python 3.8+
- CUDA 11.8+ (for GPU)
- Docker + nvidia-docker (for Docker deployment)

## Testing

Run test suite:

```bash
python test_api.py
```

Tests include:
- Health check
- Language and task listing
- Text-to-text translation
- Speech-to-text translation (requires audio file)
- Transcription (requires audio file)
- Speech-to-speech translation (requires audio file)

Add test audio file at `examples/test_audio.wav` for full testing.

## Configuration

Edit `config.py` to customize:

- **Model:** Change `MODEL_NAME` for different model sizes
- **Device:** Set `DEVICE` to "cuda" or "cpu"
- **Proxy:** Configure `HTTP_PROXY` for model downloads
- **Server:** Modify `SERVER_HOST` and `SERVER_PORT`
- **Languages:** Add/remove from `SUPPORTED_LANGUAGES`

## Troubleshooting

### Model Download Issues

If model download fails or hangs:

```bash
# Set proxy
export HTTP_PROXY=http://localhost:1091
export HTTPS_PROXY=http://localhost:1091

# Pre-download model
huggingface-cli download facebook/seamless-m4t-v2-large
```

### GPU Memory Issues

If running out of VRAM:
1. Use smaller model: `facebook/seamless-m4t-medium`
2. Reduce batch size (process one audio at a time)
3. Use CPU mode (set `DEVICE="cpu"` in config.py)

### Audio Format Issues

Supported formats: WAV, MP3, FLAC, M4A, OGG

If audio fails to process:
- Ensure audio is mono or stereo (will be converted)
- Check sample rate (will be resampled to 16kHz)
- Verify file is not corrupted

## API Documentation

Interactive documentation available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Performance

Typical processing times on NVIDIA H20 GPU:

| Task | Duration | Processing Time | Throughput |
|------|----------|-----------------|------------|
| T2TT | - | 0.3-0.5s | ~100 requests/min |
| TTS | - | 8-10s | ~6-8 requests/min |
| S2TT | 5s audio | 1-2s | ~30 audio files/min |
| ASR | 5s audio | 1-2s | ~30 audio files/min |
| S2ST | 5s audio | 3-5s | ~15 audio files/min |

## Project Structure

```
m4t/
├── config.py              # Configuration settings
├── models.py              # Model loading and inference
├── server.py              # FastAPI application
├── voice_detector.py      # Voice activity detection (Silero VAD)
├── audio_separator.py     # Audio source separation (Spleeter)
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker Compose configuration
├── start_dev.sh          # Development startup script
├── start_docker.sh       # Docker startup script
├── test_api.py           # API test suite
├── tts_example.py        # Text-to-Speech examples
├── commands.local.sh     # Quick test commands
├── README.md             # This file
└── examples/             # Example files
    └── test_audio.wav    # Test audio file
```

## License

This API wrapper is open source. SeamlessM4T v2 model is released under CC BY-NC 4.0 license (non-commercial use).

## References

- **SeamlessM4T:** https://github.com/facebookresearch/seamless_communication
- **Model Card:** https://huggingface.co/facebook/seamless-m4t-v2-large
- **Paper:** [SeamlessM4T—Massively Multilingual & Multimodal Machine Translation](https://ai.meta.com/research/publications/seamless-m4t/)

## Support

For issues and questions:
- Check `/docs` endpoint for API documentation
- Review logs: `docker logs seamless-m4t-api`
- Test with `test_api.py`

---

**Built with ❤️ using Meta SeamlessM4T v2**

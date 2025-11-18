# SeamlessM4T Inference API

Multilingual speech and text translation API using Meta's **SeamlessM4T v2** model.

## Features

- **5 Translation & Speech Tasks:**
  - 🎤→📝 **S2TT**: Speech-to-Text Translation (e.g., Japanese audio → Chinese text)
  - 🎤→🔊 **S2ST**: Speech-to-Speech Translation (e.g., Japanese audio → Chinese audio)
  - 🎤→📝 **ASR**: Automatic Speech Recognition (e.g., Japanese audio → Japanese text)
  - 📝→📝 **T2TT**: Text-to-Text Translation (e.g., English text → Chinese text)
  - 📝→🔊 **TTS**: Text-to-Speech (e.g., Chinese text → Chinese audio)

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

### Option 2: Docker

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or using the script
./start_docker.sh

# Or manually
docker build -t seamless-m4t-api .
docker run -d --gpus '"device=0"' -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  seamless-m4t-api
```

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

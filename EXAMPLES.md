# Example Usage Guide

## Quick Test Examples

### 1. Start the Server

```bash
cd ~/work/m4t

# Option A: Development mode
./start_dev.sh

# Option B: Docker mode
./start_docker.sh
```

### 2. Wait for Model Loading

First startup takes 2-5 minutes to download the model (~9GB).
Check health status:

```bash
curl http://localhost:8000/health
```

### 3. Test with Text Translation (No Audio Needed!)

```bash
# Japanese to Chinese (Simplified)
curl -X POST "http://localhost:8000/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "こんにちは、元気ですか？",
    "source_lang": "jpn",
    "target_lang": "cmn"
  }'

# Expected output:
# {"task":"t2tt","source_language":"jpn","target_language":"cmn",...,"output_text":"你好，你好吗？",...}
```

### 4. More Text Examples

```bash
# Japanese to Traditional Chinese
curl -X POST "http://localhost:8000/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "今日はいい天気です",
    "source_lang": "jpn",
    "target_lang": "cmn_Hant"
  }'

# Japanese to English
curl -X POST "http://localhost:8000/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "ありがとうございます",
    "source_lang": "jpn",
    "target_lang": "eng"
  }'

# Korean to Chinese
curl -X POST "http://localhost:8000/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요",
    "source_lang": "kor",
    "target_lang": "cmn"
  }'
```

### 5. Audio Translation (Requires Audio File)

```bash
# Record or prepare a Japanese audio file: test.wav

# Translate Japanese speech to Chinese text
curl -X POST "http://localhost:8000/v1/speech-to-text-translation" \
  -F "audio=@test.wav" \
  -F "target_lang=cmn" \
  -F "source_lang=jpn"

# Transcribe Japanese speech to Japanese text
curl -X POST "http://localhost:8000/v1/transcribe" \
  -F "audio=@test.wav" \
  -F "language=jpn"

# Translate Japanese speech to Chinese speech (saves as translated.wav)
curl -X POST "http://localhost:8000/v1/speech-to-speech-translation" \
  -F "audio=@test.wav" \
  -F "target_lang=cmn" \
  -F "source_lang=jpn" \
  -F "response_format=audio" \
  -o translated.wav
```

### 6. Check Supported Languages

```bash
# List all supported languages
curl http://localhost:8000/languages | python3 -m json.tool

# Check if specific language is supported
curl http://localhost:8000/languages | grep -i "japanese\|chinese"
```

### 7. Interactive API Documentation

Open in browser:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

Try the endpoints directly in the browser!

## Python Usage

```python
import requests

API_BASE = "http://localhost:8000"

# 1. Check health
response = requests.get(f"{API_BASE}/health")
print(response.json())

# 2. Text translation
response = requests.post(
    f"{API_BASE}/v1/text-to-text-translation",
    json={
        "text": "こんにちは、世界",
        "source_lang": "jpn",
        "target_lang": "cmn"
    }
)
result = response.json()
print(f"Translation: {result['output_text']}")
print(f"Time: {result['processing_time']}s")

# 3. Audio translation (if you have audio file)
with open("japanese_audio.wav", "rb") as f:
    response = requests.post(
        f"{API_BASE}/v1/speech-to-text-translation",
        files={"audio": ("audio.wav", f, "audio/wav")},
        data={"target_lang": "cmn", "source_lang": "jpn"}
    )
print(response.json())
```

## Monitoring

```bash
# Check server logs (Docker)
docker logs -f seamless-m4t-api

# Check GPU usage
nvidia-smi

# Monitor memory
watch -n 1 nvidia-smi
```

## Stopping the Server

```bash
# Development mode
# Press Ctrl+C in the terminal

# Docker mode
docker stop seamless-m4t-api
docker rm seamless-m4t-api

# Or using docker-compose
docker-compose down
```

## Performance Tips

1. **First request is slow** - Model loads lazily on first use
2. **GPU recommended** - 10-50x faster than CPU
3. **Batch processing** - Process multiple texts/audios sequentially
4. **Keep warm** - Send periodic requests to keep model in memory
5. **Audio preprocessing** - Convert to WAV 16kHz mono for best results

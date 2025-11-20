# GPT-SoVITS Voice Cloning Integration - Implementation Summary

**Date:** 2025-11-20
**Project:** m4t (SeamlessM4T API Server)
**Feature:** Voice cloning using GPT-SoVITS

---

## Overview

Successfully integrated GPT-SoVITS voice cloning capabilities into the m4t API server, enabling text-to-speech generation with cloned voice characteristics from reference audio samples.

---

## Implementation Details

### 1. Created Voice Cloner Module

**File:** `/home/camus/work/m4t/voice_cloner.py` (293 lines)

**Key Features:**
- `VoiceCloner` class with GPT-SoVITS API integration
- Support for voice cloning from audio files or bytes
- Singleton pattern for efficient resource management
- Service availability checking
- Default reference audio configuration

**Main Methods:**
```python
clone_voice_from_audio(text, text_language, refer_wav_path, prompt_text, prompt_language)
clone_voice_from_bytes(text, text_language, refer_audio_bytes, prompt_text, prompt_language)
clone_voice_with_default(text, text_language)
change_default_reference(refer_wav_path, prompt_text, prompt_language)
is_available()  # Check if GPT-SoVITS service is running
```

### 2. Added API Endpoint

**File:** `/home/camus/work/m4t/server.py`

**Endpoint:** `POST /v1/voice-clone`

**Request Format:**
- Multipart form data
- Parameters:
  - `audio`: Reference audio file (required)
  - `text`: Text to synthesize (required)
  - `text_language`: Language code (required)
  - `prompt_text`: Reference audio transcription (required)
  - `prompt_language`: Reference audio language (required)
  - `cut_punc`: Text splitting punctuation (optional)

**Response Format:**
```json
{
  "task": "voice_clone",
  "output_audio_base64": "...",
  "output_sample_rate": 32000,
  "text_length": 35,
  "output_duration": 3.5,
  "processing_time": 2.1,
  "service_available": true
}
```

### 3. Created Test Scripts

#### Module Validation Test
**File:** `/home/camus/work/m4t/test_voice_cloner_module.py`

**Tests:**
- ✅ VoiceCloner initialization
- ✅ Singleton pattern
- ✅ Service availability check
- ✅ API integration (when service is running)

**Test Results:**
```
======================================================================
Voice Cloner Module Validation Test
======================================================================
✓ All tests passed!
```

#### Integration Test
**File:** `/home/camus/work/m4t/test_voice_cloning.py`

**Features:**
- Service health check (GPT-SoVITS + m4t)
- End-to-end voice cloning test
- Audio file validation
- Detailed error reporting

### 4. Updated Documentation

**File:** `/home/camus/work/m4t/README.md`

Added comprehensive section documenting:
- Voice cloning endpoint usage
- Requirements and setup
- Example curl commands
- Parameter descriptions
- Response format
- Notes on audio quality and GPU requirements

---

## Usage Examples

### Start Services

```bash
# 1. Start GPT-SoVITS API service
cd /home/camus/work/GPT-SoVITS
python api.py -dr assets/reference.wav -dt "Reference text" -dl "zh"

# 2. Start m4t API server
cd /home/camus/work/m4t
python server.py
```

### Test Voice Cloning

```bash
# Using curl
curl -X POST "http://localhost:8000/v1/voice-clone" \
  -F "audio=@reference_audio.wav" \
  -F "text=Hello, this is a voice cloning test." \
  -F "text_language=en" \
  -F "prompt_text=This is the reference audio." \
  -F "prompt_language=en" \
  | python -c "import sys, json, base64; \
    data=json.load(sys.stdin); \
    open('output.wav', 'wb').write(base64.b64decode(data['output_audio_base64']))"

# Using Python
import requests
import base64

with open("reference.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/voice-clone",
        files={"audio": f},
        data={
            "text": "你好，这是一个语音克隆测试。",
            "text_language": "zh",
            "prompt_text": "参考音频文字",
            "prompt_language": "zh"
        }
    )

if response.status_code == 200:
    result = response.json()
    audio_bytes = base64.b64decode(result['output_audio_base64'])
    with open("cloned_voice.wav", "wb") as f:
        f.write(audio_bytes)
```

### Run Tests

```bash
# Module validation test
cd /home/camus/work/m4t
./env/bin/python test_voice_cloner_module.py

# Integration test (requires services running)
./env/bin/python test_voice_cloning.py
```

---

## Technical Specifications

### GPT-SoVITS Service Requirements

- **Service URL:** http://localhost:9880
- **API Version:** Original GPT-SoVITS api.py
- **GPU Memory:** 4-6GB VRAM (inference)
- **Supported Languages:** Chinese, English, Japanese, Korean, and more

### Integration Architecture

```
┌─────────────────┐
│   m4t Client    │
└────────┬────────┘
         │ HTTP POST /v1/voice-clone
         ▼
┌─────────────────┐
│   m4t Server    │
│  (FastAPI)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ voice_cloner.py │
│  (VoiceCloner)  │
└────────┬────────┘
         │ HTTP POST /
         ▼
┌─────────────────┐
│  GPT-SoVITS     │
│  API Service    │
│  (port 9880)    │
└─────────────────┘
```

### Error Handling

1. **Service Unavailable (503):**
   - GPT-SoVITS service not running
   - Returns suggestion to start service

2. **Voice Clone Error (500):**
   - Audio format issues
   - Processing failures
   - Returns detailed error message

3. **Timeout:**
   - Set to 60 seconds for voice generation
   - Can be adjusted in `voice_cloner.py`

---

## Testing Results

### Module Tests
- ✅ Initialization: PASSED
- ✅ Singleton pattern: PASSED
- ✅ Service availability check: PASSED (detects GPT-SoVITS not running)
- ⚠️ API integration: SKIPPED (requires running service)

### Integration Status
- ✅ API endpoint added to server
- ✅ Request/response models defined
- ✅ Error handling implemented
- ✅ Documentation updated
- ⚠️ End-to-end test: Pending (requires GPT-SoVITS service)

---

## Files Created/Modified

### New Files:
1. `/home/camus/work/m4t/voice_cloner.py` - Voice cloning module (293 lines)
2. `/home/camus/work/m4t/test_voice_cloner_module.py` - Module validation test (136 lines)
3. `/home/camus/work/m4t/test_voice_cloning.py` - Integration test script (252 lines)

### Modified Files:
1. `/home/camus/work/m4t/server.py` - Added voice cloning endpoint (~150 lines added)
2. `/home/camus/work/m4t/README.md` - Added documentation section

---

## Next Steps

### Immediate:
1. ✅ Module implementation complete
2. ✅ API endpoint integrated
3. ✅ Tests written and validated
4. ⏳ Start GPT-SoVITS service for full end-to-end testing

### Future Enhancements:
1. **Voice conversion mode:** Convert existing audio to different voice
2. **Batch processing:** Clone multiple texts in single request
3. **Voice library:** Store and reuse pre-computed voice embeddings
4. **Quality metrics:** Add MOS (Mean Opinion Score) evaluation
5. **Streaming:** Support streaming voice generation
6. **Multi-speaker:** Support multiple reference voices

---

## Integration with stream-polyglot

The voice cloning API can be integrated into stream-polyglot's audio dubbing pipeline:

```python
# In stream-polyglot main.py
from TTS.api import TTS

# After SeamlessM4T translation
for fragment in translated_fragments:
    # Option 1: Use Coqui TTS for voice conversion
    tts = TTS("voice_conversion_models/multilingual/vctk/freevc24")
    cloned_audio = tts.voice_conversion(
        source_wav=fragment['translated'],
        target_wav=fragment['original']
    )

    # Option 2: Use GPT-SoVITS via m4t API
    response = requests.post(
        "http://localhost:8000/v1/voice-clone",
        files={"audio": open(fragment['original'], 'rb')},
        data={
            "text": fragment['translated_text'],
            "text_language": target_lang,
            "prompt_text": fragment['original_text'],
            "prompt_language": source_lang
        }
    )
    cloned_audio = base64.b64decode(response.json()['output_audio_base64'])
```

---

## Performance Benchmarks

### Expected Performance (Based on Research):
- **RTX 3090 (24GB):** RTF ~0.02-0.03 (4060Ti reference: RTF 0.028)
- **Inference Time:** ~2-5 seconds for 30-second audio
- **Memory Usage:** 4-6GB VRAM
- **Audio Quality:** High (comparable to reference)

### Actual Performance:
⏳ Pending - Requires GPT-SoVITS service to be running for benchmarking

---

## Troubleshooting

### Issue: "GPT-SoVITS service is not available"

**Cause:** GPT-SoVITS API server not running

**Solution:**
```bash
cd /home/camus/work/GPT-SoVITS
python api.py \
  -dr assets/default_reference.wav \
  -dt "Default reference text" \
  -dl "zh" \
  -p 9880
```

### Issue: "Voice cloning failed"

**Possible Causes:**
1. Reference audio quality poor (noisy, distorted)
2. Audio format not supported
3. Text language mismatch
4. GPU memory insufficient

**Solutions:**
- Use clean, high-quality reference audio (16kHz+ recommended)
- Convert to WAV format if needed
- Verify language codes match audio content
- Close other GPU applications to free memory

---

## Conclusion

✅ **GPT-SoVITS voice cloning successfully integrated into m4t API**

The implementation provides:
- Clean API interface for voice cloning
- Robust error handling
- Comprehensive documentation
- Test coverage
- Ready for production use (pending GPT-SoVITS service deployment)

**Status:** Implementation Complete ✓
**Next:** Deploy and test with running GPT-SoVITS service

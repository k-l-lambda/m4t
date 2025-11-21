"""
FastAPI Server for SeamlessM4T Inference
Provides REST API endpoints for speech and text translation
"""
import io
import logging
import base64
from typing import Optional
import wave
import tempfile
import os

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np

from config_m4t import (
    SERVER_HOST,
    SERVER_PORT,
    SUPPORTED_LANGUAGES,
    SUPPORTED_TASKS,
    MAX_AUDIO_LENGTH,
    MAX_TEXT_LENGTH,
    TASK_S2TT,
    TASK_S2ST,
    TASK_ASR,
    TASK_T2TT,
    map_seamless_to_gptsovits_lang,
)
from models import get_model
from voice_detector import get_vad
from audio_separator import get_separator
from voice_cloner import get_voice_cloner
from gptsovits_local import get_gptsovits_local

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SeamlessM4T Inference API",
    description="Multilingual speech and text translation API using Meta's SeamlessM4T v2",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ==================== Pydantic Models ====================

class TextTranslationRequest(BaseModel):
    """Request model for text-to-text translation"""
    text: str = Field(..., description="Input text to translate", max_length=MAX_TEXT_LENGTH)
    source_lang: str = Field(..., description="Source language code (e.g., 'jpn')")
    target_lang: str = Field(..., description="Target language code (e.g., 'cmn')")


class TextToSpeechRequest(BaseModel):
    """Request model for text-to-speech"""
    text: str = Field(..., description="Input text to convert to speech", max_length=MAX_TEXT_LENGTH)
    source_lang: str = Field(..., description="Language code for the speech (e.g., 'eng', 'cmn', 'jpn')")
    speaker_id: int = Field(0, description="Speaker voice ID (0-199, default: 0)", ge=0, le=199)


class TranslationResponse(BaseModel):
    """Response model for all translation tasks"""
    task: str
    source_language: Optional[str] = None
    target_language: str
    input_text: Optional[str] = None
    input_duration: Optional[float] = None
    output_text: str
    processing_time: float


class SpeechTranslationResponse(BaseModel):
    """Response for speech-to-speech translation"""
    task: str
    source_language: Optional[str] = None
    target_language: str
    input_duration: Optional[float] = None
    output_text: str
    output_audio_base64: str
    output_sample_rate: int
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str


class LanguagesResponse(BaseModel):
    """Supported languages response"""
    count: int
    languages: dict


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(None, description="Additional error details")
    suggestion: Optional[str] = Field(None, description="Suggested solution or next steps")


class VADSegment(BaseModel):
    """Voice activity detection segment"""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    duration: float = Field(..., description="Duration in seconds")


class VADResponse(BaseModel):
    """Voice activity detection response"""
    task: str = "vad"
    total_duration: float = Field(..., description="Total audio duration in seconds")
    speech_segments: list[VADSegment]
    segment_count: int
    total_speech_duration: float = Field(..., description="Total speech duration in seconds")
    processing_time: float


class SeparatorResponse(BaseModel):
    """Audio separation response"""
    task: str = "separate"
    input_duration: float
    vocals_audio_base64: str = Field(..., description="Base64-encoded vocals audio (WAV)")
    sample_rate: int
    processing_time: float
    separator_available: bool


class AudioSplitResponse(BaseModel):
    """Audio split response with both vocals and accompaniment"""
    task: str = "audio_split"
    input_duration: float
    vocals_audio_base64: str = Field(..., description="Base64-encoded vocals audio (WAV)")
    accompaniment_audio_base64: str = Field(..., description="Base64-encoded accompaniment audio (WAV)")
    sample_rate: int
    processing_time: float
    separator_available: bool


# ==================== Helper Functions ====================

def create_error_response(
    error_type: str,
    message: str,
    details: Optional[str] = None,
    suggestion: Optional[str] = None
) -> JSONResponse:
    """Create a standardized error response"""
    error_data = {
        "error": error_type,
        "message": message,
    }
    if details:
        error_data["details"] = details
    if suggestion:
        error_data["suggestion"] = suggestion

    return error_data


def validate_language(lang_code: str, param_name: str = "language"):
    """Validate if language code is supported"""
    if lang_code not in SUPPORTED_LANGUAGES:
        error_data = create_error_response(
            error_type="UnsupportedLanguage",
            message=f"Language code '{lang_code}' is not supported",
            details=f"The {param_name} '{lang_code}' is not in the list of supported languages",
            suggestion=f"Use GET /languages to see all supported languages, or try common codes like 'eng', 'cmn', 'jpn', 'kor'"
        )
        raise HTTPException(status_code=400, detail=error_data)


async def read_audio_file(file: UploadFile) -> bytes:
    """Read and validate uploaded audio file"""
    try:
        audio_data = await file.read()
        if not audio_data:
            error_data = create_error_response(
                error_type="EmptyFile",
                message="The uploaded audio file is empty",
                suggestion="Please upload a valid audio file (WAV, MP3, FLAC, M4A, OGG)"
            )
            raise HTTPException(status_code=400, detail=error_data)
        return audio_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading audio file: {e}")
        error_data = create_error_response(
            error_type="FileReadError",
            message="Failed to read the uploaded audio file",
            details=str(e),
            suggestion="Ensure the file is a valid audio format and not corrupted"
        )
        raise HTTPException(status_code=400, detail=error_data)


def audio_array_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes"""
    buffer = io.BytesIO()

    # Normalize audio to int16
    if audio_array.dtype != np.int16:
        audio_array = (audio_array * 32767).astype(np.int16)

    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())

    buffer.seek(0)
    return buffer.read()


# ==================== API Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=model.device
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model_loaded": False, "error": str(e)}
        )


@app.get("/languages", response_model=LanguagesResponse)
async def list_languages():
    """List all supported languages"""
    return LanguagesResponse(
        count=len(SUPPORTED_LANGUAGES),
        languages=SUPPORTED_LANGUAGES
    )


@app.get("/tasks")
async def list_tasks():
    """List all supported tasks"""
    return {
        "count": len(SUPPORTED_TASKS),
        "tasks": {
            TASK_S2TT: "Speech-to-Text Translation",
            TASK_S2ST: "Speech-to-Speech Translation",
            TASK_ASR: "Automatic Speech Recognition (Transcription)",
            TASK_T2TT: "Text-to-Text Translation",
        }
    }


@app.post("/v1/speech-to-text-translation", response_model=TranslationResponse)
async def speech_to_text_translation(
    audio: UploadFile = File(..., description="Audio file to translate"),
    target_lang: str = Form(..., description="Target language code (e.g., 'cmn')"),
    source_lang: Optional[str] = Form(None, description="Source language code (optional)"),
    separate_vocals: bool = Form(False, description="Separate vocals from background music first")
):
    """
    Translate speech to text in a different language (S2TT)

    Example: Japanese audio -> Chinese text

    Optional: Use `separate_vocals=true` to extract vocals from background music first.
    """
    try:
        # Validate language
        validate_language(target_lang, "target_lang")
        if source_lang:
            validate_language(source_lang, "source_lang")

        # Read audio
        audio_data = await read_audio_file(audio)

        # Optional: Separate vocals from background music
        if separate_vocals:
            logger.info("Separating vocals from background music...")
            separator = get_separator()
            if separator.is_available():
                audio_data = separator.preprocess_audio_bytes(audio_data)
                logger.info("Vocals separated successfully")
            else:
                logger.warning("Spleeter not available, skipping vocal separation")

        # Get model and perform translation
        model = get_model()
        result = model.speech_to_text_translation(
            audio_data=audio_data,
            target_lang=target_lang,
            source_lang=source_lang
        )

        return TranslationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in S2TT: {e}")
        error_data = create_error_response(
            error_type="AudioProcessingError",
            message="Failed to process speech-to-text translation",
            details=str(e),
            suggestion="Check that your audio file is valid (WAV, MP3, FLAC) and not corrupted. For Japanese translation, ensure audio is clear."
        )
        raise HTTPException(status_code=500, detail=error_data)


@app.post("/v1/speech-to-speech-translation")
async def speech_to_speech_translation(
    audio: UploadFile = File(..., description="Audio file to translate"),
    target_lang: str = Form(..., description="Target language code"),
    source_lang: Optional[str] = Form(None, description="Source language code (optional)"),
    response_format: str = Form("json", description="Response format: 'json' or 'audio'"),
    separate_vocals: bool = Form(False, description="Separate vocals from background music first"),
    speaker_id: int = Form(0, description="Speaker voice ID (0-199, default: 0)")
):
    """
    Translate speech to speech in a different language (S2ST)

    Example: Japanese audio -> Chinese audio

    Optional: Use `separate_vocals=true` to extract vocals from background music first.

    Response formats:
    - 'json': Returns JSON with base64-encoded audio
    - 'audio': Returns WAV audio file directly
    """
    try:
        # Validate language
        validate_language(target_lang, "target_lang")
        if source_lang:
            validate_language(source_lang, "source_lang")

        # Read audio
        audio_data = await read_audio_file(audio)

        # Optional: Separate vocals from background music
        if separate_vocals:
            logger.info("Separating vocals from background music...")
            separator = get_separator()
            if separator.is_available():
                audio_data = separator.preprocess_audio_bytes(audio_data)
                logger.info("Vocals separated successfully")
            else:
                logger.warning("Spleeter not available, skipping vocal separation")

        # Get model and perform translation
        model = get_model()
        result = model.speech_to_speech_translation(
            audio_data=audio_data,
            target_lang=target_lang,
            source_lang=source_lang,
            speaker_id=speaker_id
        )

        # Extract audio array
        output_audio = result.pop("output_audio")
        output_sr = result["output_sample_rate"]

        # Return based on format
        if response_format == "audio":
            # Convert to WAV and return as audio file
            wav_bytes = audio_array_to_wav_bytes(output_audio, output_sr)
            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=translated_audio.wav"
                }
            )
        else:
            # Return JSON with base64 audio
            wav_bytes = audio_array_to_wav_bytes(output_audio, output_sr)
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

            return SpeechTranslationResponse(
                **result,
                output_audio_base64=audio_base64
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in S2ST: {e}")
        error_data = create_error_response(
            error_type="AudioProcessingError",
            message="Failed to process speech-to-speech translation",
            details=str(e),
            suggestion="Check that your audio file is valid. Ensure both source and target languages support speech generation."
        )
        raise HTTPException(status_code=500, detail=error_data)


@app.post("/v1/transcribe", response_model=TranslationResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form(..., description="Language code of the audio (e.g., 'jpn')")
):
    """
    Transcribe speech to text in the same language (ASR)

    Example: Japanese audio -> Japanese text
    """
    try:
        # Validate language
        validate_language(language, "language")

        # Read audio
        audio_data = await read_audio_file(audio)

        # Get model and perform transcription
        model = get_model()
        result = model.transcribe(
            audio_data=audio_data,
            language=language
        )

        return TranslationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ASR: {e}")
        error_data = create_error_response(
            error_type="AudioProcessingError",
            message="Failed to transcribe audio",
            details=str(e),
            suggestion="Check that your audio file is valid and the language code matches the audio language. Common codes: 'jpn', 'cmn', 'eng'."
        )
        raise HTTPException(status_code=500, detail=error_data)


@app.post("/v1/text-to-text-translation", response_model=TranslationResponse)
async def text_to_text_translation(request: TextTranslationRequest):
    """
    Translate text to text in a different language (T2TT)

    Example: Japanese text -> Chinese text
    """
    try:
        # Validate languages
        validate_language(request.source_lang, "source_lang")
        validate_language(request.target_lang, "target_lang")

        # Get model and perform translation
        model = get_model()
        result = model.text_to_text_translation(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )

        return TranslationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in T2TT: {e}")
        error_data = create_error_response(
            error_type="TranslationError",
            message="Failed to translate text",
            details=str(e),
            suggestion="Check that both source and target language codes are valid. Note: Japanese text translation has known limitations; consider using speech translation instead."
        )
        raise HTTPException(status_code=500, detail=error_data)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SeamlessM4T Inference API",
        "version": "1.0.0",
        "model": "facebook/seamless-m4t-v2-large",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "languages": "/languages",
            "tasks": "/tasks",
            "s2tt": "/v1/speech-to-text-translation",
            "s2st": "/v1/speech-to-speech-translation",
            "asr": "/v1/transcribe",
            "t2tt": "/v1/text-to-text-translation",
            "tts": "/v1/text-to-speech",
        }
    }


@app.post("/v1/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Text-to-Speech (TTS)

    Converts text into speech audio in the specified language.

    Args:
        text: Input text to convert to speech
        source_lang: Language code (e.g., "eng", "cmn", "jpn")

    Returns:
        JSON with generated speech audio and metadata
    """
    try:
        model = get_model()
        result = model.text_to_speech(
            text=request.text,
            language=request.source_lang,
            speaker_id=request.speaker_id
        )

        # Convert numpy array to list for JSON serialization
        result["output_audio"] = result["output_audio"].tolist()

        return result

    except Exception as e:
        logger.error(f"Error in TTS: {e}")
        error_data = create_error_response(
            error_type="SpeechGenerationError",
            message="Failed to generate speech from text",
            details=str(e),
            suggestion="Check that the language code is valid and supports speech generation. Common codes: 'cmn', 'eng', 'jpn', 'kor'."
        )
        raise HTTPException(status_code=500, detail=error_data)


# ==================== VAD and Audio Separation Endpoints ====================

@app.post("/v1/detect-voice", response_model=VADResponse)
async def detect_voice(
    audio: UploadFile = File(..., description="Audio file to analyze"),
    threshold: float = Form(0.5, description="Speech detection threshold (0.0-1.0)"),
    min_speech_duration_ms: int = Form(250, description="Minimum speech duration in ms"),
    min_silence_duration_ms: int = Form(300, description="Minimum silence duration in ms")
):
    """
    Detect voice activity in audio file

    Returns speech segments with timestamps.
    """
    import time
    start_time = time.time()

    try:
        # Read audio file
        audio_data = await audio.read()

        # Get VAD detector
        vad = get_vad()

        # Detect speech segments
        segments = vad.detect_speech_from_bytes(
            audio_data,
            sample_rate=16000,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms
        )

        # Calculate total durations
        total_speech_duration = sum(seg['duration'] for seg in segments)

        # Get total duration
        import soundfile as sf
        audio_array, sr = sf.read(io.BytesIO(audio_data))
        total_duration = len(audio_array) / sr

        processing_time = time.time() - start_time

        return VADResponse(
            total_duration=total_duration,
            speech_segments=[VADSegment(**seg) for seg in segments],
            segment_count=len(segments),
            total_speech_duration=total_speech_duration,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error in voice detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "VoiceDetectionError",
                "Failed to detect voice activity",
                str(e),
                "Ensure audio file is in a supported format (WAV, MP3, FLAC, etc.)"
            )
        )


@app.post("/v1/separate-vocals", response_model=SeparatorResponse)
async def separate_vocals(
    audio: UploadFile = File(..., description="Audio file to separate")
):
    """
    Separate vocals from background music

    Returns vocals-only audio in base64-encoded WAV format.
    Requires Spleeter to be installed.
    """
    import time
    start_time = time.time()

    try:
        # Get audio separator
        separator = get_separator()

        # Check if separator is available
        if not separator.is_available():
            raise HTTPException(
                status_code=503,
                detail=create_error_response(
                    "SeparatorUnavailable",
                    "Spleeter is not installed or unavailable",
                    "The audio separation feature requires Spleeter",
                    "Install Spleeter with: pip install spleeter"
                )
            )

        # Read audio file
        audio_data = await audio.read()

        # Get input duration
        import soundfile as sf
        audio_array, sr = sf.read(io.BytesIO(audio_data))
        input_duration = len(audio_array) / sr

        # Separate vocals
        vocals_array, vocals_sr = separator.separate_vocals_from_bytes(audio_data, sample_rate=16000)

        # Convert to base64
        buffer = io.BytesIO()
        sf.write(buffer, vocals_array, vocals_sr, format='WAV')
        buffer.seek(0)
        vocals_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        processing_time = time.time() - start_time

        return SeparatorResponse(
            input_duration=input_duration,
            vocals_audio_base64=vocals_base64,
            sample_rate=vocals_sr,
            processing_time=processing_time,
            separator_available=True
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error in audio separation: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "AudioSeparationError",
                "Failed to separate vocals from audio",
                str(e),
                "Ensure audio file is in a supported format and Spleeter is properly installed"
            )
        )


# ==================== Voice Cloning Endpoint ====================

class VoiceCloneRequest(BaseModel):
    """Request model for voice cloning"""
    text: str = Field(..., description="Text to synthesize")
    text_language: str = Field(..., description="Language of the text (zh, en, ja, etc.)")
    prompt_text: str = Field(..., description="Text content of the reference audio")
    prompt_language: str = Field(..., description="Language of the reference audio")
    cut_punc: Optional[str] = Field(None, description="Optional text splitting punctuation")


class VoiceCloneResponse(BaseModel):
    """Response model for voice cloning"""
    task: str = "voice_clone"
    output_audio_base64: str
    output_sample_rate: int
    text_length: int
    output_duration: float
    processing_time: float
    service_available: bool


@app.post("/v1/audio-split", response_model=AudioSplitResponse)
async def audio_split_endpoint(
    audio: UploadFile = File(..., description="Audio file to split into vocals and accompaniment")
):
    """
    Audio Source Separation - Split audio into vocals and accompaniment

    Separates an audio file into two streams:
    - Vocals: Human voice/singing
    - Accompaniment: Background music and other sounds

    Uses Spleeter's 2-stem model for separation.

    **Example Usage:**
    ```bash
    curl -X POST "http://localhost:8000/v1/audio-split" \\
      -F "audio=@song.wav" \\
      -o output.json

    # Extract vocals
    jq -r '.vocals_audio_base64' output.json | base64 -d > vocals.wav

    # Extract accompaniment
    jq -r '.accompaniment_audio_base64' output.json | base64 -d > accompaniment.wav
    ```

    **Returns:**
    - vocals_audio_base64: Base64-encoded vocals audio (WAV format)
    - accompaniment_audio_base64: Base64-encoded accompaniment audio (WAV format)
    - sample_rate: Sample rate of both audio streams
    - processing_time: Time taken to process
    """
    import time
    start_time = time.time()

    try:
        # Check if separator is available
        separator = get_separator()
        if not separator.is_available():
            raise HTTPException(
                status_code=503,
                detail=create_error_response(
                    "SeparatorUnavailable",
                    "Audio separation service is not available",
                    "Spleeter is not installed or failed to load",
                    "Install spleeter: pip install spleeter"
                )
            )

        # Read audio file
        audio_data = await read_audio_file(audio)
        logger.info(f"Processing audio split for file: {audio.filename} ({len(audio_data)} bytes)")

        # Get input duration
        import soundfile as sf
        input_duration = len(sf.read(io.BytesIO(audio_data))[0]) / sf.read(io.BytesIO(audio_data))[1]

        # Separate audio into vocals and accompaniment
        vocals_array, accompaniment_array, sr = separator.separate_audio_streams(audio_data, sample_rate=16000)

        # Convert arrays to WAV bytes
        vocals_buffer = io.BytesIO()
        accompaniment_buffer = io.BytesIO()

        sf.write(vocals_buffer, vocals_array, sr, format='WAV')
        sf.write(accompaniment_buffer, accompaniment_array, sr, format='WAV')

        vocals_buffer.seek(0)
        accompaniment_buffer.seek(0)

        # Encode to base64
        vocals_base64 = base64.b64encode(vocals_buffer.read()).decode('utf-8')
        accompaniment_base64 = base64.b64encode(accompaniment_buffer.read()).decode('utf-8')

        processing_time = time.time() - start_time

        logger.info(f"Audio split completed: {input_duration:.2f}s audio, {processing_time:.2f}s processing")

        return AudioSplitResponse(
            input_duration=input_duration,
            vocals_audio_base64=vocals_base64,
            accompaniment_audio_base64=accompaniment_base64,
            sample_rate=sr,
            processing_time=processing_time,
            separator_available=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in audio split: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "AudioSplitError",
                "Audio separation failed",
                str(e),
                "Check audio file format and Spleeter installation"
            )
        )


@app.post("/v1/voice-clone", response_model=VoiceCloneResponse)
async def voice_clone_endpoint(
    audio: UploadFile = File(..., description="Reference audio file"),
    text: str = Form(..., description="Text to synthesize"),
    text_language: str = Form(..., description="Language of the text"),
    prompt_text: str = Form(..., description="Text content of reference audio"),
    prompt_language: str = Form(..., description="Language of reference audio"),
    cut_punc: Optional[str] = Form(None, description="Text splitting punctuation"),
    seed: int = Form(-1, description="Random seed for reproducibility (-1 for random, 0-1000000 for fixed seed)")
):
    """
    Voice cloning endpoint using GPT-SoVITS (Direct Python Integration)

    **Clone a speaker's voice and generate speech from text**

    This endpoint uses GPT-SoVITS to clone the voice characteristics from a reference
    audio sample and generate speech with the same voice for the given text.

    **Parameters:**
    - **audio**: Reference audio file (WAV format recommended)
    - **text**: Text to synthesize in the target language
    - **text_language**: Language code (SeamlessM4T codes like 'eng', 'cmn', 'jpn' or GPT-SoVITS codes like 'en', 'zh', 'ja')
    - **prompt_text**: Transcription of the reference audio
    - **prompt_language**: Language code (SeamlessM4T codes like 'eng', 'cmn', 'jpn' or GPT-SoVITS codes like 'en', 'zh', 'ja')
    - **cut_punc**: Optional punctuation marks for text segmentation
    - **seed**: Random seed for reproducibility (-1 for random, 0-1000000 for fixed seed, default: -1)

    **Returns:**
    - **output_audio_base64**: Generated audio in base64 encoding
    - **output_sample_rate**: Sample rate of the generated audio
    - **processing_time**: Time taken for voice cloning

    **Example:**
    ```bash
    # Using SeamlessM4T language codes
    curl -X POST "http://localhost:8000/v1/voice-clone" \
      -F "audio=@reference.wav" \
      -F "text=Hello, this is a test." \
      -F "text_language=eng" \
      -F "prompt_text=This is the reference audio." \
      -F "prompt_language=eng"

    # Using GPT-SoVITS language codes (also supported)
    curl -X POST "http://localhost:8000/v1/voice-clone" \
      -F "audio=@reference.wav" \
      -F "text=你好，这是一个测试。" \
      -F "text_language=zh" \
      -F "prompt_text=This is the reference audio." \
      -F "prompt_language=en"

    # With fixed seed for reproducibility
    curl -X POST "http://localhost:8000/v1/voice-clone" \
      -F "audio=@reference.wav" \
      -F "text=Hello, this is a test." \
      -F "text_language=eng" \
      -F "prompt_text=This is the reference audio." \
      -F "prompt_language=eng" \
      -F "seed=42"
    ```

    **Notes:**
    - Automatically maps SeamlessM4T language codes (eng, cmn, jpn) to GPT-SoVITS codes (en, zh, ja)
    - Uses direct Python integration (no external GPT-SoVITS service needed)
    - Reference audio should be clear and noise-free for best results
    - Longer reference audio (10-30 seconds) generally produces better quality
    - Set seed to a fixed value (e.g., 42) for reproducible results, or -1 for random generation
    """
    import time
    import soundfile as sf
    start_time = time.time()

    temp_audio_path = None

    try:
        # Get local GPT-SoVITS instance (singleton, auto-initializes on first call)
        gptsovits = get_gptsovits_local()

        # Read reference audio bytes
        audio_bytes = await audio.read()

        # Save reference audio to temporary file (GPT-SoVITS needs file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        # Map SeamlessM4T language codes to GPT-SoVITS codes
        gptsovits_text_lang = map_seamless_to_gptsovits_lang(text_language)
        gptsovits_prompt_lang = map_seamless_to_gptsovits_lang(prompt_language)

        logger.info(f"Voice cloning: text='{text[:50]}...', text_lang={text_language}->{gptsovits_text_lang}, "
                   f"prompt_lang={prompt_language}->{gptsovits_prompt_lang}, ref_audio={temp_audio_path}, seed={seed}")

        # Perform voice cloning using local GPT-SoVITS
        # Balanced parameters for stable generation with good quality
        result_audio = gptsovits.generate_speech(
            text=text,
            text_language=gptsovits_text_lang,
            ref_wav_path=temp_audio_path,
            prompt_text=prompt_text,
            prompt_language=gptsovits_prompt_lang,
            top_k=12,
            top_p=0.7,
            temperature=0.5,
            speed=1.0,
            spk="default",
            seed=seed
        )

        if result_audio is None:
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    "VoiceCloneError",
                    "Voice cloning failed",
                    "Failed to generate audio with cloned voice",
                    "Ensure reference audio is clear and in supported format"
                )
            )

        # Calculate output duration
        audio_array, sr = sf.read(io.BytesIO(result_audio))
        output_duration = len(audio_array) / sr

        # Convert to base64
        audio_base64 = base64.b64encode(result_audio).decode('utf-8')

        processing_time = time.time() - start_time

        logger.info(f"Voice cloning completed: {len(result_audio)} bytes, "
                   f"{output_duration:.2f}s audio, {processing_time:.2f}s processing")

        return VoiceCloneResponse(
            output_audio_base64=audio_base64,
            output_sample_rate=sr,
            text_length=len(text),
            output_duration=output_duration,
            processing_time=processing_time,
            service_available=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in voice cloning: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "VoiceCloneError",
                "Voice cloning failed",
                str(e),
                "Check audio file format and GPT-SoVITS model status"
            )
        )
    finally:
        # Clean up temporary audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_audio_path}: {e}")


# ==================== Server Startup ====================

if __name__ == "__main__":
    logger.info(f"Starting SeamlessM4T API server on {SERVER_HOST}:{SERVER_PORT}")
    logger.info("Loading model on startup...")

    # Pre-load model
    try:
        model = get_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Server will start, but model loading will be attempted on first request")

    # Start server
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info"
    )

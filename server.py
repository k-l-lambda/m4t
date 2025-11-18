"""
FastAPI Server for SeamlessM4T Inference
Provides REST API endpoints for speech and text translation
"""
import io
import logging
import base64
from typing import Optional
import wave

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np

from config import (
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
)
from models import get_model

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
    source_lang: Optional[str] = Form(None, description="Source language code (optional)")
):
    """
    Translate speech to text in a different language (S2TT)

    Example: Japanese audio → Chinese text
    """
    try:
        # Validate language
        validate_language(target_lang, "target_lang")
        if source_lang:
            validate_language(source_lang, "source_lang")

        # Read audio
        audio_data = await read_audio_file(audio)

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
    response_format: str = Form("json", description="Response format: 'json' or 'audio'")
):
    """
    Translate speech to speech in a different language (S2ST)

    Example: Japanese audio → Chinese audio

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

        # Get model and perform translation
        model = get_model()
        result = model.speech_to_speech_translation(
            audio_data=audio_data,
            target_lang=target_lang,
            source_lang=source_lang
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

    Example: Japanese audio → Japanese text
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

    Example: Japanese text → Chinese text
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
            language=request.source_lang
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

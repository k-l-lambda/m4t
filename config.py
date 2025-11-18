"""
Configuration for SeamlessM4T Inference Server
"""
import os
from typing import Dict, List

# Model Configuration
MODEL_NAME = "facebook/seamless-m4t-v2-large"
MODEL_SIZE = "2.3B"
DEVICE = "cuda"  # or "cpu"
TORCH_DTYPE = "float16"  # or "float32"

# Proxy Configuration (for downloading models from HuggingFace)
HTTP_PROXY = os.getenv("HTTP_PROXY", "http://localhost:1091")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "http://localhost:1091")

# Set proxy for model downloads
os.environ["HTTP_PROXY"] = HTTP_PROXY
os.environ["HTTPS_PROXY"] = HTTPS_PROXY

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
MAX_AUDIO_LENGTH = 300  # seconds
MAX_TEXT_LENGTH = 2000  # characters

# Audio Configuration
TARGET_SAMPLE_RATE = 16000  # SeamlessM4T expects 16kHz audio
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

# Language Configuration
# Format: {language_code: language_name}
SUPPORTED_LANGUAGES: Dict[str, str] = {
    # East Asian Languages
    "jpn": "Japanese",
    "cmn": "Chinese (Simplified)",
    "cmn_Hant": "Chinese (Traditional)",
    "yue": "Cantonese (Traditional)",
    "kor": "Korean",

    # European Languages
    "eng": "English",
    "fra": "French",
    "deu": "German",
    "spa": "Spanish",
    "ita": "Italian",
    "por": "Portuguese",
    "rus": "Russian",

    # Other Asian Languages
    "hin": "Hindi",
    "tha": "Thai",
    "vie": "Vietnamese",
    "ind": "Indonesian",
    "msa": "Malay",
    "tgl": "Tagalog",

    # Middle Eastern Languages
    "ara": "Arabic",
    "heb": "Hebrew",
    "tur": "Turkish",
    "fas": "Persian",
}

# Task Types
TASK_S2TT = "s2tt"  # Speech-to-Text Translation
TASK_S2ST = "s2st"  # Speech-to-Speech Translation
TASK_ASR = "asr"    # Automatic Speech Recognition (transcription)
TASK_T2TT = "t2tt"  # Text-to-Text Translation

SUPPORTED_TASKS: List[str] = [TASK_S2TT, TASK_S2ST, TASK_ASR, TASK_T2TT]

# Model Cache Directory
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

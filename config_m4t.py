"""
Configuration for SeamlessM4T Inference Server
"""
import env_loader  # Load .env.local before any config values
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
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
MAX_AUDIO_LENGTH = int(os.getenv("MAX_AUDIO_LENGTH", "300"))  # seconds
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2000"))  # characters

# GPT-SoVITS Configuration
GPTSOVITS_API_URL = os.getenv("GPTSOVITS_API_URL", "http://localhost:9880")

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

# Language code mapping: SeamlessM4T -> GPT-SoVITS
# GPT-SoVITS supports: en, zh, ja, ko, yue, all_zh, all_ja
SEAMLESS_TO_GPTSOVITS_LANG_MAP: Dict[str, str] = {
    "eng": "en",           # English
    "cmn": "zh",           # Chinese (Simplified) -> Chinese
    "cmn_Hant": "zh",      # Chinese (Traditional) -> Chinese
    "yue": "yue",          # Cantonese (Traditional)
    "jpn": "ja",           # Japanese
    "kor": "ko",           # Korean
    "fra": "en",           # French -> English (fallback)
    "deu": "en",           # German -> English (fallback)
    "spa": "en",           # Spanish -> English (fallback)
    "ita": "en",           # Italian -> English (fallback)
    "por": "en",           # Portuguese -> English (fallback)
    "rus": "en",           # Russian -> English (fallback)
    "hin": "en",           # Hindi -> English (fallback)
    "tha": "en",           # Thai -> English (fallback)
    "vie": "en",           # Vietnamese -> English (fallback)
    "ind": "en",           # Indonesian -> English (fallback)
    "msa": "en",           # Malay -> English (fallback)
    "tgl": "en",           # Tagalog -> English (fallback)
    "ara": "en",           # Arabic -> English (fallback)
    "heb": "en",           # Hebrew -> English (fallback)
    "tur": "en",           # Turkish -> English (fallback)
    "fas": "en",           # Persian -> English (fallback)
}

def map_seamless_to_gptsovits_lang(seamless_lang: str) -> str:
    """
    Map SeamlessM4T language code to GPT-SoVITS language code

    Args:
        seamless_lang: SeamlessM4T language code (e.g., 'eng', 'cmn', 'jpn')

    Returns:
        GPT-SoVITS language code (e.g., 'en', 'zh', 'ja')
        Returns the input unchanged if it's already a GPT-SoVITS code
    """
    # If already a GPT-SoVITS code, return as-is
    if seamless_lang in ["en", "zh", "ja", "ko", "yue", "all_zh", "all_ja"]:
        return seamless_lang

    # Map SeamlessM4T code to GPT-SoVITS code
    return SEAMLESS_TO_GPTSOVITS_LANG_MAP.get(seamless_lang, seamless_lang)

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

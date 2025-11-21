"""
GPT-SoVITS Local Integration - Direct Python Calls

This module provides direct Python function calls to GPT-SoVITS TTS engine,
avoiding HTTP networking entirely by importing GPT-SoVITS modules directly.

Author: Claude Code
Date: 2025-11-20
Approach: Minimal dependency integration (validated working)
"""

import sys
import os
import logging
from typing import Optional
from io import BytesIO

# Add GPT-SoVITS paths FIRST before any imports
gptsovits_base = "/home/camus/work/m4t/third_party/GPT-SoVITS"
sys.path.insert(0, gptsovits_base)
sys.path.insert(0, os.path.join(gptsovits_base, "GPT_SoVITS"))
sys.path.insert(0, os.path.join(gptsovits_base, "GPT_SoVITS", "eres2net"))

import torch
import soundfile as sf

logger = logging.getLogger(__name__)


class GPTSoVITSLocal:
    """
    Local GPT-SoVITS inference without HTTP.

    Strategy: Import GPT-SoVITS's api module and initialize its global state,
    then call get_tts_wav() directly as a function.
    """

    _instance = None
    _initialized = False

    def __init__(
        self,
        device: str = "cuda",
        is_half: bool = True,
        gpt_path: str = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        sovits_path: str = "GPT_SoVITS/pretrained_models/s2G488k.pth",
        cnhubert_base_path: str = "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        bert_path: str = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    ):
        """
        Initialize GPT-SoVITS by setting up the api module's global state.

        This is a lightweight initialization - actual model loading happens
        on first inference call (lazy loading).
        """
        if GPTSoVITSLocal._initialized:
            logger.info("GPTSoVITS already initialized (singleton)")
            return

        logger.info(f"Initializing GPTSoVITS Local on {device} (half precision: {is_half})")

        self.device = device
        self.is_half = is_half
        self.gptsovits_base = gptsovits_base

        # Make paths absolute
        self.gpt_path = os.path.join(gptsovits_base, gpt_path)
        self.sovits_path = os.path.join(gptsovits_base, sovits_path)
        self.cnhubert_base_path = os.path.join(gptsovits_base, cnhubert_base_path)
        self.bert_path = os.path.join(gptsovits_base, bert_path)

        # Verify model files exist
        for name, path in [
            ("GPT model", self.gpt_path),
            ("SoVITS model", self.sovits_path),
            ("CNHubert", self.cnhubert_base_path),
            ("BERT", self.bert_path)
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at: {path}")

        # Import api module (this sets up all the infrastructure)
        # IMPORTANT: Change working directory before import because api.py
        # has module-level initialization code that uses relative paths
        try:
            logger.info("Importing GPT-SoVITS api module...")

            # Save current directory and change to GPT-SoVITS base
            original_cwd = os.getcwd()
            os.chdir(gptsovits_base)

            # Remove m4t's config from module cache to avoid conflicts
            # GPT-SoVITS needs to import its own config module
            if 'config' in sys.modules:
                import config as m4t_config  # Save reference to m4t's config
                del sys.modules['config']
                logger.info("Temporarily removed m4t's config from sys.modules")

            try:
                import api as gptsovits_api
                self.api = gptsovits_api

                # Set global configuration in api module
                self.api.device = device
                self.api.is_half = is_half

                logger.info("GPT-SoVITS api module imported successfully")
            finally:
                # Always restore original working directory
                os.chdir(original_cwd)

                # Restore m4t's config to sys.modules
                if 'm4t_config' in locals():
                    sys.modules['config'] = m4t_config
                    logger.info("Restored m4t's config to sys.modules")

        except ImportError as e:
            logger.error(f"Failed to import GPT-SoVITS api: {e}")
            raise

        # Initialize models using api's initialization logic
        self._initialize_models()

        GPTSoVITSLocal._initialized = True
        logger.info("GPTSoVITS Local initialized successfully!")

    def _initialize_models(self):
        """
        Initialize GPT and SoVITS models using api module's helper functions.

        The BERT and SSL models are already initialized when the api module is imported
        (at module level in api.py lines 1282-1291), so we just need to load the
        GPT and SoVITS models for the default speaker.
        """
        try:
            # Initialize speaker with GPT and SoVITS models using helper functions
            logger.info("Loading GPT and SoVITS models...")
            self._init_speaker(
                "default",
                self.gpt_path,
                self.sovits_path
            )

            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def _init_speaker(self, spk_name: str, gpt_path: str, sovits_path: str):
        """
        Initialize a speaker using api module's helper functions.

        This mirrors the logic from api.py's change_gpt_sovits_weights function.
        """
        # Use api module's helper functions to load model weights
        # These functions handle all the complex initialization logic
        gpt = self.api.get_gpt_weights(gpt_path)
        sovits = self.api.get_sovits_weights(sovits_path)

        # Create Speaker instance using api module's Speaker class
        Speaker = self.api.Speaker
        speaker = Speaker(name=spk_name, gpt=gpt, sovits=sovits)

        # Add to api's speaker_list
        self.api.speaker_list[spk_name] = speaker

        logger.info(f"Speaker '{spk_name}' initialized with GPT and SoVITS models")

    def _set_seed(self, seed: int):
        """
        Set random seed for reproducible generation.

        This sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (CPU and CUDA)
        - PYTHONHASHSEED environment variable

        Args:
            seed: Random seed value (if -1, generates random seed)
        """
        import random
        import numpy as np
        import os

        if seed == -1:
            seed = random.randint(0, 1000000)

        seed = int(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        logger.debug(f"Random seed set to: {seed}")

    def generate_speech(
        self,
        text: str,
        text_language: str,
        ref_wav_path: str,
        prompt_text: str,
        prompt_language: str,
        top_k: int = 15,
        top_p: float = 0.6,
        temperature: float = 0.6,
        speed: float = 1.0,
        spk: str = "default",
        seed: int = -1
    ) -> Optional[bytes]:
        """
        Generate speech audio from text using voice cloning.

        This directly calls api.get_tts_wav() which has all the complex
        TTS logic already implemented.

        Args:
            text: Text to synthesize
            text_language: Language code (zh, en, ja, etc.)
            ref_wav_path: Path to reference audio file
            prompt_text: Transcription of reference audio
            prompt_language: Language of reference audio
            top_k, top_p, temperature: Sampling parameters
            speed: Speech speed multiplier
            spk: Speaker name (default: "default")
            seed: Random seed for reproducibility (-1 for random, 0-1000000 for fixed)

        Returns:
            WAV audio bytes or None if generation failed
        """
        try:
            if not GPTSoVITSLocal._initialized:
                raise RuntimeError("GPTSoVITS not initialized! Call __init__() first.")

            # Set random seed if specified
            if seed != -1:
                self._set_seed(seed)
                logger.info(f"Set random seed to: {seed}")

            logger.info(f"Generating speech: '{text[:50]}...' ({text_language})")

            # Call api module's get_tts_wav function directly
            result = self.api.get_tts_wav(
                ref_wav_path=ref_wav_path,
                prompt_text=prompt_text,
                prompt_language=prompt_language,
                text=text,
                text_language=text_language,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                speed=speed,
                spk=spk
            )

            # result is a generator that yields audio chunks as bytes
            # Collect all chunks into a single bytes object
            audio_chunks = []
            for chunk in result:
                audio_chunks.append(chunk)

            if not audio_chunks:
                logger.error("No audio generated")
                return None

            # Concatenate all audio chunks
            # The chunks are already in WAV format (complete with headers)
            # Just use the last chunk which contains the complete WAV file
            wav_bytes = audio_chunks[-1] if audio_chunks else None

            if wav_bytes:
                logger.info(f"Generated {len(wav_bytes)} bytes of audio")
                return wav_bytes
            else:
                logger.error("No valid audio data")
                return None

        except Exception as e:
            logger.error(f"Speech generation failed: {e}", exc_info=True)
            return None


# Singleton accessor
_instance_lock = None

def get_gptsovits_local() -> GPTSoVITSLocal:
    """
    Get singleton instance of GPTSoVITSLocal.

    Thread-safe lazy initialization.

    Returns:
        Initialized GPTSoVITSLocal instance
    """
    global _instance_lock

    if GPTSoVITSLocal._instance is None:
        # Simple lock for thread safety
        if _instance_lock is None:
            import threading
            _instance_lock = threading.Lock()

        with _instance_lock:
            # Double-check pattern
            if GPTSoVITSLocal._instance is None:
                logger.info("Creating new GPTSoVITSLocal instance (singleton)")
                GPTSoVITSLocal._instance = GPTSoVITSLocal()

    return GPTSoVITSLocal._instance

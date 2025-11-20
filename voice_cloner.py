"""
Voice Cloning Module using GPT-SoVITS

This module provides voice cloning capabilities by integrating with GPT-SoVITS API.
It allows text-to-speech generation using a reference audio sample to clone the voice.
"""

import os
import logging
import httpx
from typing import Optional, Dict, Any
import base64
from config import GPTSOVITS_API_URL

logger = logging.getLogger(__name__)


class VoiceCloner:
    """Voice cloning using GPT-SoVITS API"""

    def __init__(
        self,
        api_url: str = "http://localhost:9880",
        default_refer_wav: Optional[str] = None,
        default_refer_text: Optional[str] = None,
        default_refer_language: str = "zh"
    ):
        """
        Initialize Voice Cloner

        Args:
            api_url: GPT-SoVITS API server URL
            default_refer_wav: Default reference audio path
            default_refer_text: Default reference audio text
            default_refer_language: Default reference audio language
        """
        self.api_url = api_url.rstrip('/')
        self.default_refer_wav = default_refer_wav
        self.default_refer_text = default_refer_text
        self.default_refer_language = default_refer_language

        self._available = None
        logger.info(f"VoiceCloner initialized with API URL: {self.api_url}")

    async def is_available(self) -> bool:
        """
        Check if GPT-SoVITS service is available

        Returns:
            True if service is running and accessible
        """
        # Always recheck availability (don't cache indefinitely)
        # This allows the service to become available after startup
        logger.info(f"[HTTPX-ASYNC] Checking GPT-SoVITS at {self.api_url}/control")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_url}/control",
                    timeout=2.0
                )
                self._available = response.status_code in [200, 404]  # 404 means server is running
                logger.info(f"[HTTPX-ASYNC] GPT-SoVITS service available: {self._available}, status_code: {response.status_code}")
                return self._available
        except httpx.HTTPStatusError as e:
            # HTTPStatusError means server responded with an error code
            # No longer need 503 workaround with async httpx
            self._available = e.response.status_code in [200, 404]
            logger.info(f"[HTTPX-ASYNC] GPT-SoVITS HTTPStatusError: {e.response.status_code}, available: {self._available}")
            return self._available
        except Exception as e:
            logger.warning(f"[HTTPX-ASYNC] GPT-SoVITS service not available: {e}")
            self._available = False
            return False

    async def clone_voice_from_audio(
        self,
        text: str,
        text_language: str,
        refer_wav_path: str,
        prompt_text: str,
        prompt_language: str,
        cut_punc: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Clone voice from reference audio and generate speech

        Args:
            text: Text to synthesize
            text_language: Language of the text ('zh', 'en', 'ja', etc.)
            refer_wav_path: Path to reference audio file
            prompt_text: Text content of the reference audio
            prompt_language: Language of the reference audio
            cut_punc: Optional text splitting punctuation

        Returns:
            Audio bytes (WAV format) or None if failed
        """
        if not await self.is_available():
            logger.error("GPT-SoVITS service is not available")
            return None

        try:
            # Prepare request data
            data = {
                "refer_wav_path": refer_wav_path,
                "prompt_text": prompt_text,
                "prompt_language": prompt_language,
                "text": text,
                "text_language": text_language
            }

            if cut_punc:
                data["cut_punc"] = cut_punc

            # Call GPT-SoVITS API with async httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/",
                    json=data,
                    timeout=60.0  # Voice generation can take time
                )

                if response.status_code == 200:
                    # Success: returns WAV audio stream
                    logger.info(f"Voice cloning successful, audio size: {len(response.content)} bytes")
                    return response.content
                else:
                    # Error: returns JSON with error message
                    error_msg = response.json() if 'application/json' in response.headers.get('content-type', '') else response.text
                    logger.error(f"Voice cloning failed: {error_msg}")
                    return None

        except httpx.TimeoutException:
            logger.error("Voice cloning request timeout")
            return None
        except Exception as e:
            logger.error(f"Voice cloning error: {e}")
            return None

    async def clone_voice_from_bytes(
        self,
        text: str,
        text_language: str,
        refer_audio_bytes: bytes,
        prompt_text: str,
        prompt_language: str,
        cut_punc: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Clone voice from reference audio bytes

        Args:
            text: Text to synthesize
            text_language: Language of the text
            refer_audio_bytes: Reference audio file bytes
            prompt_text: Text content of the reference audio
            prompt_language: Language of the reference audio
            cut_punc: Optional text splitting punctuation

        Returns:
            Audio bytes (WAV format) or None if failed
        """
        import tempfile

        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(refer_audio_bytes)
            tmp_path = tmp.name

        try:
            # Use the file-based method
            result = await self.clone_voice_from_audio(
                text=text,
                text_language=text_language,
                refer_wav_path=tmp_path,
                prompt_text=prompt_text,
                prompt_language=prompt_language,
                cut_punc=cut_punc
            )
            return result
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def clone_voice_with_default(
        self,
        text: str,
        text_language: str,
        cut_punc: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Clone voice using default reference audio

        Args:
            text: Text to synthesize
            text_language: Language of the text
            cut_punc: Optional text splitting punctuation

        Returns:
            Audio bytes (WAV format) or None if failed
        """
        if not self.default_refer_wav:
            logger.error("No default reference audio configured")
            return None

        return await self.clone_voice_from_audio(
            text=text,
            text_language=text_language,
            refer_wav_path=self.default_refer_wav,
            prompt_text=self.default_refer_text or "",
            prompt_language=self.default_refer_language,
            cut_punc=cut_punc
        )

    async def change_default_reference(
        self,
        refer_wav_path: str,
        prompt_text: str,
        prompt_language: str
    ) -> bool:
        """
        Change the default reference audio

        Args:
            refer_wav_path: Path to new reference audio
            prompt_text: Text content of the reference audio
            prompt_language: Language of the reference audio

        Returns:
            True if successful
        """
        if not await self.is_available():
            logger.error("GPT-SoVITS service is not available")
            return False

        try:
            data = {
                "refer_wav_path": refer_wav_path,
                "prompt_text": prompt_text,
                "prompt_language": prompt_language
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/change_refer",
                    json=data,
                    timeout=10.0
                )

                if response.status_code == 200:
                    # Update local defaults
                    self.default_refer_wav = refer_wav_path
                    self.default_refer_text = prompt_text
                    self.default_refer_language = prompt_language
                    logger.info("Default reference audio changed successfully")
                    return True
                else:
                    logger.error(f"Failed to change default reference: {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Error changing default reference: {e}")
            return False


# Singleton instance
_voice_cloner_instance: Optional[VoiceCloner] = None


def get_voice_cloner(
    api_url: Optional[str] = None,
    **kwargs
) -> VoiceCloner:
    """
    Get or create the VoiceCloner singleton instance

    Args:
        api_url: GPT-SoVITS API URL (defaults to GPTSOVITS_API_URL from config)
        **kwargs: Additional arguments for VoiceCloner constructor

    Returns:
        VoiceCloner instance
    """
    global _voice_cloner_instance

    if _voice_cloner_instance is None:
        if api_url is None:
            api_url = GPTSOVITS_API_URL
        _voice_cloner_instance = VoiceCloner(api_url=api_url, **kwargs)

    return _voice_cloner_instance

"""
SeamlessM4T Model Manager
Handles model loading, inference, and audio processing
"""
import logging
import time
from typing import Dict, Optional, Union, Tuple
import io

import torch
import torchaudio
import numpy as np
from transformers import (
    AutoProcessor,
    SeamlessM4Tv2ForSpeechToText,
    SeamlessM4Tv2ForTextToSpeech,
    SeamlessM4Tv2Model,
)

from config import (
    MODEL_NAME,
    DEVICE,
    TORCH_DTYPE,
    TARGET_SAMPLE_RATE,
    MAX_AUDIO_LENGTH,
    TASK_S2TT,
    TASK_S2ST,
    TASK_ASR,
    TASK_T2TT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeamlessM4TInference:
    """Unified inference class for SeamlessM4T model"""

    def __init__(self):
        self.device = DEVICE if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if TORCH_DTYPE == "float16" and self.device == "cuda" else torch.float32

        logger.info(f"Initializing SeamlessM4T on {self.device} with dtype {self.dtype}")

        # Models will be loaded lazily
        self.processor = None
        self.model_s2t = None  # For S2TT and ASR
        self.model_unified = None  # For S2ST (full model)
        self.model_t2s = None  # For T2ST if needed

        self._load_processor()

    def _load_processor(self):
        """Load the processor (always needed)"""
        if self.processor is None:
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
            logger.info("Processor loaded successfully")

    def _load_s2t_model(self):
        """Load Speech-to-Text model (for S2TT and ASR)"""
        if self.model_s2t is None:
            logger.info("Loading Speech-to-Text model...")
            self.model_s2t = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                MODEL_NAME,
                torch_dtype=self.dtype
            ).to(self.device)
            self.model_s2t.eval()
            logger.info("Speech-to-Text model loaded successfully")

    def _load_unified_model(self):
        """Load unified model (for S2ST)"""
        if self.model_unified is None:
            logger.info("Loading unified SeamlessM4T model for S2ST...")
            self.model_unified = SeamlessM4Tv2Model.from_pretrained(
                MODEL_NAME,
                torch_dtype=self.dtype
            ).to(self.device)
            self.model_unified.eval()
            logger.info("Unified model loaded successfully")

    def preprocess_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        sample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio to required format

        Args:
            audio_data: Audio bytes or numpy array
            sample_rate: Sample rate if audio_data is numpy array

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            if isinstance(audio_data, bytes):
                # Load audio from bytes using soundfile instead of torchaudio
                # to avoid TorchCodec requirement in torchaudio 2.9+
                import soundfile as sf
                audio_array, sr = sf.read(io.BytesIO(audio_data), dtype='float32')
                # Convert to torch tensor
                audio_tensor = torch.from_numpy(audio_array)
                # Ensure it's 2D (channels, samples)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
            elif isinstance(audio_data, np.ndarray):
                if sample_rate is None:
                    raise ValueError("sample_rate must be provided when audio_data is numpy array")
                audio_tensor = torch.from_numpy(audio_data)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                sr = sample_rate
            else:
                raise ValueError(f"Unsupported audio_data type: {type(audio_data)}")

            # Convert to mono if stereo
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            # Resample to 16kHz if needed
            if sr != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
                audio_tensor = resampler(audio_tensor)
                sr = TARGET_SAMPLE_RATE

            # Convert to numpy array
            audio_array = audio_tensor.squeeze().numpy()

            # Check audio length
            duration = len(audio_array) / sr
            if duration > MAX_AUDIO_LENGTH:
                logger.warning(f"Audio duration {duration:.2f}s exceeds maximum {MAX_AUDIO_LENGTH}s")
                # Truncate
                audio_array = audio_array[:MAX_AUDIO_LENGTH * sr]

            return audio_array, sr

        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise

    def speech_to_text_translation(
        self,
        audio_data: Union[bytes, np.ndarray],
        target_lang: str,
        source_lang: Optional[str] = None,
        sample_rate: Optional[int] = None
    ) -> Dict:
        """
        Speech-to-Text Translation (S2TT)

        Args:
            audio_data: Audio bytes or numpy array
            target_lang: Target language code (e.g., "cmn", "cmn_Hant")
            source_lang: Source language code (optional, for metadata)
            sample_rate: Sample rate if audio_data is numpy array

        Returns:
            Dict with translation result and metadata
        """
        start_time = time.time()

        try:
            # Load model if not loaded
            self._load_s2t_model()

            # Preprocess audio
            audio_array, sr = self.preprocess_audio(audio_data, sample_rate)
            duration = len(audio_array) / sr

            # Process audio
            audio_inputs = self.processor(
                audios=audio_array,
                sampling_rate=sr,
                return_tensors="pt"
            )

            # Move to device
            audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

            # Generate translation
            with torch.no_grad():
                output_tokens = self.model_s2t.generate(
                    **audio_inputs,
                    tgt_lang=target_lang
                )

            # Decode output
            translated_text = self.processor.decode(
                output_tokens[0].tolist(),
                skip_special_tokens=True
            )

            processing_time = time.time() - start_time

            return {
                "task": TASK_S2TT,
                "source_language": source_lang,
                "target_language": target_lang,
                "input_duration": round(duration, 2),
                "output_text": translated_text,
                "processing_time": round(processing_time, 2),
            }

        except Exception as e:
            logger.error(f"Error in S2TT: {e}")
            raise

    def speech_to_speech_translation(
        self,
        audio_data: Union[bytes, np.ndarray],
        target_lang: str,
        source_lang: Optional[str] = None,
        sample_rate: Optional[int] = None,
        speaker_id: int = 0
    ) -> Dict:
        """
        Speech-to-Speech Translation (S2ST)

        Args:
            audio_data: Audio bytes or numpy array
            target_lang: Target language code
            source_lang: Source language code (optional)
            sample_rate: Sample rate if audio_data is numpy array
            speaker_id: Speaker voice ID (0-199, default: 0)

        Returns:
            Dict with translation result including audio and text
        """
        start_time = time.time()

        try:
            # Load unified model
            self._load_unified_model()

            # Preprocess audio
            audio_array, sr = self.preprocess_audio(audio_data, sample_rate)
            duration = len(audio_array) / sr

            # Process audio
            audio_inputs = self.processor(
                audios=audio_array,
                sampling_rate=sr,
                return_tensors="pt"
            )

            # Move to device
            audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

            # Generate speech and text
            with torch.no_grad():
                output = self.model_unified.generate(
                    **audio_inputs,
                    tgt_lang=target_lang,
                    generate_speech=True,
                    return_intermediate_token_ids=True,
                    speaker_id=speaker_id
                )

            # Extract audio and text
            # output is a tuple: (audio_tensor, text_tokens)
            # audio_tensor has shape [batch_size, samples], need to squeeze batch dimension
            audio_samples = output[0].cpu().squeeze().numpy()

            # Handle text_tokens carefully - it might be a 0-dim tensor
            text_tokens = None
            if len(output) > 1 and output[1] is not None:
                # output[1] might be a tensor or a list
                if isinstance(output[1], torch.Tensor):
                    # If it's a 0-dim tensor, use .item() to get the value
                    if output[1].dim() == 0:
                        text_tokens = None  # Skip 0-dim tensors
                    else:
                        text_tokens = output[1][0] if output[1].dim() > 0 else None
                else:
                    # It's already a list or sequence
                    text_tokens = output[1][0] if len(output[1]) > 0 else None

            # Decode text if available
            translated_text = ""
            if text_tokens is not None:
                translated_text = self.processor.decode(
                    text_tokens.tolist(),
                    skip_special_tokens=True
                )

            processing_time = time.time() - start_time

            return {
                "task": TASK_S2ST,
                "source_language": source_lang,
                "target_language": target_lang,
                "input_duration": round(duration, 2),
                "output_text": translated_text,
                "output_audio": audio_samples,
                "output_sample_rate": self.model_unified.config.sampling_rate,
                "processing_time": round(processing_time, 2),
            }

        except Exception as e:
            logger.error(f"Error in S2ST: {e}")
            raise

    def transcribe(
        self,
        audio_data: Union[bytes, np.ndarray],
        language: str,
        sample_rate: Optional[int] = None
    ) -> Dict:
        """
        Automatic Speech Recognition (ASR) - Transcription

        Args:
            audio_data: Audio bytes or numpy array
            language: Language code for transcription
            sample_rate: Sample rate if audio_data is numpy array

        Returns:
            Dict with transcription result
        """
        # ASR is essentially S2TT with same source and target language
        return self.speech_to_text_translation(
            audio_data=audio_data,
            target_lang=language,
            source_lang=language,
            sample_rate=sample_rate
        )

    def text_to_text_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict:
        """
        Text-to-Text Translation (T2TT)

        Args:
            text: Input text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Dict with translation result
        """
        start_time = time.time()

        try:
            # Load unified model (needed for T2TT)
            self._load_unified_model()

            # Process text - processor handles text input
            text_inputs = self.processor(
                text=text,
                src_lang=source_lang,
                return_tensors="pt"
            )

            # Move to device
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

            # Generate translation
            with torch.no_grad():
                output = self.model_unified.generate(
                    **text_inputs,
                    tgt_lang=target_lang,
                    generate_speech=False
                )

            # Decode output - access sequences attribute
            if hasattr(output, 'sequences'):
                translated_text = self.processor.batch_decode(
                    output.sequences,
                    skip_special_tokens=True
                )[0]
            else:
                # Fallback if output structure is different
                translated_text = self.processor.batch_decode(
                    [output[0]],
                    skip_special_tokens=True
                )[0]

            processing_time = time.time() - start_time

            return {
                "task": TASK_T2TT,
                "source_language": source_lang,
                "target_language": target_lang,
                "input_text": text,
                "output_text": translated_text,
                "processing_time": round(processing_time, 2),
            }

        except Exception as e:
            logger.error(f"Error in T2TT: {e}")
            raise

    def text_to_speech(
        self,
        text: str,
        language: str,
        speaker_id: int = 0
    ) -> Dict:
        """
        Text-to-Speech (TTS)

        Args:
            text: Input text
            language: Language code for the speech output
            speaker_id: Speaker voice ID (0-199, default: 0)

        Returns:
            Dict with speech audio and metadata
        """
        start_time = time.time()

        try:
            # Load unified model (needed for TTS)
            self._load_unified_model()

            # Process text
            text_inputs = self.processor(
                text=text,
                src_lang=language,
                return_tensors="pt"
            )

            # Move to device
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

            # Generate speech
            with torch.no_grad():
                output = self.model_unified.generate(
                    **text_inputs,
                    tgt_lang=language,
                    generate_speech=True,
                    speaker_id=speaker_id
                )

            # Extract audio (output is tuple: (audio_tensor, text_tokens))
            audio_samples = output[0].cpu().squeeze().numpy()

            processing_time = time.time() - start_time

            return {
                "task": "tts",
                "language": language,
                "input_text": text,
                "output_audio": audio_samples,
                "output_sample_rate": self.model_unified.config.sampling_rate,
                "processing_time": round(processing_time, 2),
            }

        except Exception as e:
            logger.error(f"Error in TTS: {e}")
            raise


# Global model instance (singleton)
_model_instance: Optional[SeamlessM4TInference] = None


def get_model() -> SeamlessM4TInference:
    """Get or create the global model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = SeamlessM4TInference()
    return _model_instance

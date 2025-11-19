"""
Voice Activity Detection using Silero VAD
Detects speech segments in audio for intelligent segmentation
"""
import logging
import io
from typing import List, Dict, Union, Optional
import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Voice Activity Detection using Silero VAD"""

    def __init__(self):
        self.model = None
        self.utils = None
        self._load_model()

    def _load_model(self):
        """Load Silero VAD model from torch.hub"""
        try:
            logger.info("Loading Silero VAD model...")
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )

            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = self.utils

            logger.info("Silero VAD model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Silero VAD model: {e}")
            raise

    def detect_speech_from_bytes(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300
    ) -> List[Dict]:
        """
        Detect speech segments from audio bytes

        Args:
            audio_data: Audio data in bytes
            sample_rate: Sample rate of the audio
            threshold: Speech detection threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence duration between segments

        Returns:
            List of speech segments with start/end timestamps in seconds
        """
        try:
            # Load audio from bytes
            import soundfile as sf
            audio_array, sr = sf.read(io.BytesIO(audio_data), dtype='float32')

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_array)

            # Ensure it's 1D
            if audio_tensor.dim() > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0)

            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                audio_tensor = resampler(audio_tensor)

            # Detect speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                return_seconds=False  # Return in samples
            )

            # Convert to seconds
            speech_segments = []
            for ts in speech_timestamps:
                speech_segments.append({
                    'start': float(ts['start']) / sample_rate,
                    'end': float(ts['end']) / sample_rate,
                    'duration': float(ts['end'] - ts['start']) / sample_rate
                })

            logger.info(f"Detected {len(speech_segments)} speech segments")
            return speech_segments

        except Exception as e:
            logger.error(f"Error detecting speech: {e}")
            raise

    def detect_speech_from_file(
        self,
        audio_path: str,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300
    ) -> List[Dict]:
        """
        Detect speech segments from audio file

        Args:
            audio_path: Path to audio file
            threshold: Speech detection threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence duration between segments

        Returns:
            List of speech segments with start/end timestamps in seconds
        """
        try:
            # Read audio file
            wav = self.read_audio(audio_path, sampling_rate=16000)

            # Detect speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                wav,
                self.model,
                threshold=threshold,
                sampling_rate=16000,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                return_seconds=False
            )

            # Convert to seconds
            speech_segments = []
            for ts in speech_timestamps:
                speech_segments.append({
                    'start': float(ts['start']) / 16000,
                    'end': float(ts['end']) / 16000,
                    'duration': float(ts['end'] - ts['start']) / 16000
                })

            logger.info(f"Detected {len(speech_segments)} speech segments")
            return speech_segments

        except Exception as e:
            logger.error(f"Error detecting speech from file: {e}")
            raise

    def find_silence_gaps(
        self,
        speech_segments: List[Dict],
        total_duration: float,
        min_gap_duration: float = 0.3
    ) -> List[Dict]:
        """
        Find silence gaps between speech segments

        Args:
            speech_segments: List of speech segments
            total_duration: Total audio duration in seconds
            min_gap_duration: Minimum gap duration to consider

        Returns:
            List of silence gaps with position and duration
        """
        gaps = []

        for i in range(len(speech_segments) - 1):
            gap_start = speech_segments[i]['end']
            gap_end = speech_segments[i + 1]['start']
            gap_duration = gap_end - gap_start

            if gap_duration >= min_gap_duration:
                gaps.append({
                    'position': gap_start,
                    'duration': gap_duration,
                    'end': gap_end
                })

        logger.info(f"Found {len(gaps)} silence gaps (>{min_gap_duration}s)")
        return gaps

    def smart_segment(
        self,
        audio_data: Union[bytes, str],
        target_duration: float = 150.0,
        sample_rate: int = 16000,
        min_gap_duration: float = 0.3
    ) -> List[Dict]:
        """
        Intelligently segment audio based on speech detection

        Args:
            audio_data: Audio bytes or file path
            target_duration: Target segment duration in seconds
            sample_rate: Sample rate (for bytes input)
            min_gap_duration: Minimum gap duration for splitting

        Returns:
            List of segments with start/end timestamps
        """
        try:
            # Detect speech segments
            if isinstance(audio_data, bytes):
                speech_segments = self.detect_speech_from_bytes(
                    audio_data,
                    sample_rate=sample_rate
                )
                # Calculate total duration from audio
                import soundfile as sf
                audio_array, sr = sf.read(io.BytesIO(audio_data))
                total_duration = len(audio_array) / sr
            else:
                speech_segments = self.detect_speech_from_file(audio_data)
                # Get total duration from file
                import torchaudio
                info = torchaudio.info(audio_data)
                total_duration = info.num_frames / info.sample_rate

            if not speech_segments:
                # No speech detected, return whole audio
                return [{'start': 0.0, 'end': total_duration, 'duration': total_duration}]

            # Find silence gaps
            silence_gaps = self.find_silence_gaps(
                speech_segments,
                total_duration,
                min_gap_duration
            )

            if not silence_gaps:
                # No significant gaps, return whole audio
                return [{'start': 0.0, 'end': total_duration, 'duration': total_duration}]

            # Smart grouping
            segments = []
            current_start = 0.0
            accumulated_duration = 0.0

            for gap in silence_gaps:
                accumulated_duration = gap['position'] - current_start

                # If accumulated duration is close to target, split here
                if accumulated_duration >= target_duration * 0.8:
                    segments.append({
                        'start': current_start,
                        'end': gap['position'],
                        'duration': accumulated_duration
                    })
                    current_start = gap['end']
                    accumulated_duration = 0.0

            # Add final segment
            if current_start < total_duration:
                final_duration = total_duration - current_start
                segments.append({
                    'start': current_start,
                    'end': total_duration,
                    'duration': final_duration
                })

            logger.info(f"Segmented audio into {len(segments)} parts")
            return segments

        except Exception as e:
            logger.error(f"Error in smart segmentation: {e}")
            raise


# Global VAD instance (singleton)
_vad_instance: Optional[VoiceActivityDetector] = None


def get_vad() -> VoiceActivityDetector:
    """Get or create the global VAD instance"""
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = VoiceActivityDetector()
    return _vad_instance

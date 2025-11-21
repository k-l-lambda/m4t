"""
Audio Source Separation using Spleeter
Separates vocals from background music for better speech recognition
"""
import logging
import io
import os
import tempfile
from typing import Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioSeparator:
    """Audio source separation using Spleeter"""

    def __init__(self):
        self.separator = None
        self._spleeter_available = False
        self._try_load_spleeter()

    def _try_load_spleeter(self):
        """Try to load Spleeter, mark as unavailable if fails"""
        try:
            from spleeter.separator import Separator
            logger.info("Loading Spleeter model (2stems: vocals + accompaniment)...")
            self.separator = Separator('spleeter:2stems')
            self._spleeter_available = True
            logger.info("Spleeter loaded successfully")

        except ImportError:
            logger.warning(
                "Spleeter not installed. Audio separation features will be unavailable. "
                "Install with: pip install spleeter"
            )
            self._spleeter_available = False

        except Exception as e:
            logger.warning(f"Failed to load Spleeter: {e}")
            self._spleeter_available = False

    def is_available(self) -> bool:
        """Check if Spleeter is available"""
        return self._spleeter_available

    def separate_audio_streams(
        self,
        audio_data: bytes,
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Separate audio into vocals and accompaniment streams

        Args:
            audio_data: Audio data in bytes
            sample_rate: Target sample rate

        Returns:
            Tuple of (vocals_array, accompaniment_array, sample_rate)

        Raises:
            RuntimeError: If Spleeter is not available
        """
        if not self._spleeter_available:
            raise RuntimeError(
                "Spleeter is not available. Install with: pip install spleeter"
            )

        try:
            # Save audio bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
                tmp_input_path = tmp_input.name
                tmp_input.write(audio_data)

            # Create temporary output directory
            tmp_output_dir = tempfile.mkdtemp()

            try:
                # Separate audio
                logger.info("Separating vocals and accompaniment...")
                self.separator.separate_to_file(
                    tmp_input_path,
                    tmp_output_dir,
                    codec='wav'
                )

                # Read vocals and accompaniment files
                import soundfile as sf
                input_basename = os.path.splitext(os.path.basename(tmp_input_path))[0]
                vocals_path = os.path.join(tmp_output_dir, input_basename, 'vocals.wav')
                accompaniment_path = os.path.join(tmp_output_dir, input_basename, 'accompaniment.wav')

                vocals_array, sr_vocals = sf.read(vocals_path, dtype='float32')
                accompaniment_array, sr_accomp = sf.read(accompaniment_path, dtype='float32')

                # Resample if needed
                if sr_vocals != sample_rate:
                    import torchaudio
                    import torch
                    vocals_tensor = torch.from_numpy(vocals_array)
                    if vocals_tensor.dim() == 1:
                        vocals_tensor = vocals_tensor.unsqueeze(0)
                    elif vocals_tensor.dim() == 2:
                        vocals_tensor = vocals_tensor.T  # [channels, samples]

                    resampler = torchaudio.transforms.Resample(sr_vocals, sample_rate)
                    vocals_tensor = resampler(vocals_tensor)

                    if vocals_tensor.shape[0] == 1:
                        vocals_array = vocals_tensor.squeeze().numpy()
                    else:
                        vocals_array = vocals_tensor.T.numpy()  # [samples, channels]

                if sr_accomp != sample_rate:
                    import torchaudio
                    import torch
                    accomp_tensor = torch.from_numpy(accompaniment_array)
                    if accomp_tensor.dim() == 1:
                        accomp_tensor = accomp_tensor.unsqueeze(0)
                    elif accomp_tensor.dim() == 2:
                        accomp_tensor = accomp_tensor.T  # [channels, samples]

                    resampler = torchaudio.transforms.Resample(sr_accomp, sample_rate)
                    accomp_tensor = resampler(accomp_tensor)

                    if accomp_tensor.shape[0] == 1:
                        accompaniment_array = accomp_tensor.squeeze().numpy()
                    else:
                        accompaniment_array = accomp_tensor.T.numpy()  # [samples, channels]

                logger.info("Audio separation completed successfully")
                return vocals_array, accompaniment_array, sample_rate

            finally:
                # Clean up temporary files
                import shutil
                if os.path.exists(tmp_input_path):
                    os.unlink(tmp_input_path)
                if os.path.exists(tmp_output_dir):
                    shutil.rmtree(tmp_output_dir)

        except Exception as e:
            logger.error(f"Error separating audio: {e}")
            raise

    def separate_vocals_from_bytes(
        self,
        audio_data: bytes,
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, int]:
        """
        Separate vocals from audio bytes

        Args:
            audio_data: Audio data in bytes
            sample_rate: Target sample rate

        Returns:
            Tuple of (vocals_array, sample_rate)

        Raises:
            RuntimeError: If Spleeter is not available
        """
        if not self._spleeter_available:
            raise RuntimeError(
                "Spleeter is not available. Install with: pip install spleeter"
            )

        try:
            # Save audio bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
                tmp_input_path = tmp_input.name
                tmp_input.write(audio_data)

            # Create temporary output directory
            tmp_output_dir = tempfile.mkdtemp()

            try:
                # Separate audio
                logger.info("Separating vocals from background music...")
                self.separator.separate_to_file(
                    tmp_input_path,
                    tmp_output_dir,
                    codec='wav'
                )

                # Read vocals file
                import soundfile as sf
                input_basename = os.path.splitext(os.path.basename(tmp_input_path))[0]
                vocals_path = os.path.join(tmp_output_dir, input_basename, 'vocals.wav')

                vocals_array, sr = sf.read(vocals_path, dtype='float32')

                # Resample if needed
                if sr != sample_rate:
                    import torchaudio
                    import torch
                    vocals_tensor = torch.from_numpy(vocals_array)
                    if vocals_tensor.dim() == 1:
                        vocals_tensor = vocals_tensor.unsqueeze(0)

                    resampler = torchaudio.transforms.Resample(sr, sample_rate)
                    vocals_tensor = resampler(vocals_tensor)
                    vocals_array = vocals_tensor.squeeze().numpy()
                    sr = sample_rate

                logger.info("Vocals separated successfully")
                return vocals_array, sr

            finally:
                # Clean up temporary files
                import shutil
                if os.path.exists(tmp_input_path):
                    os.unlink(tmp_input_path)
                if os.path.exists(tmp_output_dir):
                    shutil.rmtree(tmp_output_dir)

        except Exception as e:
            logger.error(f"Error separating vocals: {e}")
            raise

    def separate_vocals_from_file(
        self,
        audio_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Separate vocals from audio file

        Args:
            audio_path: Path to input audio file
            output_path: Path to save vocals (optional)

        Returns:
            Path to vocals file

        Raises:
            RuntimeError: If Spleeter is not available
        """
        if not self._spleeter_available:
            raise RuntimeError(
                "Spleeter is not available. Install with: pip install spleeter"
            )

        try:
            # Determine output directory
            if output_path:
                output_dir = os.path.dirname(output_path)
            else:
                output_dir = tempfile.mkdtemp()

            # Separate audio
            logger.info(f"Separating vocals from: {audio_path}")
            self.separator.separate_to_file(
                audio_path,
                output_dir,
                codec='wav'
            )

            # Get vocals path
            input_basename = os.path.splitext(os.path.basename(audio_path))[0]
            vocals_path = os.path.join(output_dir, input_basename, 'vocals.wav')

            # Move to output path if specified
            if output_path and vocals_path != output_path:
                import shutil
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.move(vocals_path, output_path)
                vocals_path = output_path

            logger.info(f"Vocals saved to: {vocals_path}")
            return vocals_path

        except Exception as e:
            logger.error(f"Error separating vocals from file: {e}")
            raise

    def preprocess_audio_bytes(
        self,
        audio_data: bytes,
        sample_rate: int = 16000
    ) -> bytes:
        """
        Preprocess audio bytes by separating vocals

        Args:
            audio_data: Audio data in bytes
            sample_rate: Target sample rate

        Returns:
            Processed audio bytes (vocals only)
        """
        if not self._spleeter_available:
            logger.warning("Spleeter not available, returning original audio")
            return audio_data

        try:
            # Separate vocals
            vocals_array, sr = self.separate_vocals_from_bytes(audio_data, sample_rate)

            # Convert back to bytes
            import soundfile as sf
            buffer = io.BytesIO()
            sf.write(buffer, vocals_array, sr, format='WAV')
            buffer.seek(0)

            return buffer.read()

        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            logger.warning("Returning original audio")
            return audio_data


# Global separator instance (singleton)
_separator_instance: Optional[AudioSeparator] = None


def get_separator() -> AudioSeparator:
    """Get or create the global audio separator instance"""
    global _separator_instance
    if _separator_instance is None:
        _separator_instance = AudioSeparator()
    return _separator_instance

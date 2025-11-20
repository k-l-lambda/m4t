#!/usr/bin/env python3
"""
Test script for GPT-SoVITS voice cloning integration in m4t

This script tests the voice cloning API endpoint by:
1. Starting GPT-SoVITS service (if not running)
2. Testing voice cloning with a reference audio
3. Validating the generated audio
"""

import os
import sys
import time
import requests
import subprocess
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")


def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")


def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")


def check_gpt_sovits_service(api_url="http://localhost:9880"):
    """Check if GPT-SoVITS service is running"""
    try:
        response = requests.get(f"{api_url}/control", timeout=2)
        return True
    except:
        return False


def check_m4t_service(api_url="http://localhost:8000"):
    """Check if m4t service is running"""
    try:
        response = requests.get(f"{api_url}/docs", timeout=2)
        return response.status_code == 200
    except:
        return False


def test_voice_cloning(
    m4t_url="http://localhost:8000",
    reference_audio_path=None,
    reference_text="Hello, this is a test.",
    reference_language="en",
    target_text="Welcome to the voice cloning demonstration.",
    target_language="en",
    output_path="/tmp/voice_clone_test_output.wav"
):
    """
    Test voice cloning API

    Args:
        m4t_url: m4t API server URL
        reference_audio_path: Path to reference audio file
        reference_text: Transcription of reference audio
        reference_language: Language of reference audio
        target_text: Text to synthesize
        target_language: Language of target text
        output_path: Path to save output audio
    """
    print_info(f"Testing voice cloning API at {m4t_url}")
    print_info(f"Reference audio: {reference_audio_path}")
    print_info(f"Reference text: {reference_text}")
    print_info(f"Target text: {target_text}")

    try:
        # Prepare request
        with open(reference_audio_path, 'rb') as f:
            files = {
                'audio': ('reference.wav', f, 'audio/wav')
            }
            data = {
                'text': target_text,
                'text_language': target_language,
                'prompt_text': reference_text,
                'prompt_language': reference_language
            }

            print_info("Sending request to /v1/voice-clone...")
            start_time = time.time()

            response = requests.post(
                f"{m4t_url}/v1/voice-clone",
                files=files,
                data=data,
                timeout=120
            )

            elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()

            print_success(f"Voice cloning successful!")
            print_info(f"Processing time: {result.get('processing_time', elapsed):.2f}s")
            print_info(f"Sample rate: {result.get('output_sample_rate')} Hz")
            print_info(f"Text length: {result.get('text_length')} characters")
            print_info(f"Output duration: {result.get('output_duration', 0):.2f}s")

            # Decode and save audio
            import base64
            audio_base64 = result.get('output_audio_base64')
            if audio_base64:
                audio_bytes = base64.b64decode(audio_base64)

                with open(output_path, 'wb') as f:
                    f.write(audio_bytes)

                print_success(f"Audio saved to: {output_path}")
                print_info(f"File size: {len(audio_bytes) / 1024:.1f} KB")

                # Verify audio file
                try:
                    import soundfile as sf
                    audio_array, sr = sf.read(output_path)
                    print_success(f"Audio file verified: {len(audio_array)} samples, {sr} Hz")
                    return True
                except Exception as e:
                    print_error(f"Audio verification failed: {e}")
                    return False
            else:
                print_error("No audio data in response")
                return False
        elif response.status_code == 503:
            error_detail = response.json().get('detail', {})
            print_error("Service unavailable:")
            print_error(f"  {error_detail.get('message', 'GPT-SoVITS service not running')}")
            print_info(f"  {error_detail.get('suggestion', '')}")
            return False
        else:
            print_error(f"API error: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print_error("Request timeout (>120s)")
        return False
    except Exception as e:
        print_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("=" * 70)
    print("GPT-SoVITS Voice Cloning Integration Test")
    print("=" * 70)
    print()

    # Check services
    print_info("Checking services...")

    gpt_sovits_running = check_gpt_sovits_service()
    if gpt_sovits_running:
        print_success("GPT-SoVITS service is running (port 9880)")
    else:
        print_warning("GPT-SoVITS service not detected (port 9880)")
        print_info("Start it with: python /home/camus/work/GPT-SoVITS/api.py")

    m4t_running = check_m4t_service()
    if m4t_running:
        print_success("m4t API service is running (port 8000)")
    else:
        print_error("m4t API service not running (port 8000)")
        print_info("Start it with: python /home/camus/work/m4t/server.py")
        return 1

    print()

    # Check for test audio file
    test_audio_paths = [
        "/home/camus/work/m4t/assets/japanese_speech.wav",
        "/home/camus/work/m4t/assets/test_audio.wav",
        "/tmp/test_audio.wav"
    ]

    reference_audio = None
    for path in test_audio_paths:
        if os.path.exists(path):
            reference_audio = path
            break

    if reference_audio is None:
        print_error("No test audio file found")
        print_info("Looking for audio in:")
        for path in test_audio_paths:
            print_info(f"  - {path}")
        print()
        print_info("Please provide a reference audio file")
        return 1

    print_info(f"Using reference audio: {reference_audio}")
    print()

    # Run test
    print_info("Running voice cloning test...")
    print("-" * 70)

    success = test_voice_cloning(
        reference_audio_path=reference_audio,
        reference_text="This is a test audio.",
        reference_language="en",
        target_text="Hello, this is a voice cloning test. The cloned voice should sound similar to the reference.",
        target_language="en",
        output_path="/tmp/voice_clone_test_output.wav"
    )

    print("-" * 70)
    print()

    if success:
        print_success("All tests passed!")
        print()
        print_info("Next steps:")
        print_info("1. Listen to the output audio: /tmp/voice_clone_test_output.wav")
        print_info("2. Compare with reference audio to verify voice similarity")
        print_info("3. Try with different languages and longer text")
        return 0
    else:
        print_error("Tests failed!")
        print()
        print_info("Troubleshooting:")
        print_info("1. Ensure GPT-SoVITS service is running: http://localhost:9880")
        print_info("2. Check GPT-SoVITS logs for errors")
        print_info("3. Verify reference audio is valid WAV file")
        print_info("4. Check m4t logs: /tmp/m4t_server.log")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test script to demonstrate speaker_id parameter in TTS and S2ST
"""

import requests
import numpy as np
import soundfile as sf

API_URL = "http://localhost:8000"

def test_tts_with_different_speakers():
    """Test TTS with different speaker IDs"""

    text = "Hello, how are you today?"
    language = "eng"

    print("=" * 60)
    print("Testing TTS with Different Speaker IDs")
    print("=" * 60)
    print(f"Text: {text}")
    print(f"Language: {language}\n")

    # Test with speaker IDs 0, 50, 100, 150, 199
    speaker_ids = [0, 50, 100, 150, 199]

    for speaker_id in speaker_ids:
        print(f"Testing speaker_id={speaker_id}...")

        response = requests.post(
            f"{API_URL}/v1/text-to-speech",
            json={
                "text": text,
                "source_lang": language,
                "speaker_id": speaker_id
            }
        )

        if response.status_code == 200:
            result = response.json()

            # Save audio to file
            audio_array = np.array(result['output_audio'], dtype=np.float32)
            sample_rate = result['output_sample_rate']
            output_file = f"/tmp/tts_speaker_{speaker_id}.wav"

            sf.write(output_file, audio_array, sample_rate)

            print(f"  ✓ Success")
            print(f"    Processing time: {result['processing_time']}s")
            print(f"    Audio length: {len(audio_array) / sample_rate:.2f}s")
            print(f"    Saved to: {output_file}\n")
        else:
            print(f"  ✗ Failed: {response.status_code}")
            print(f"    {response.text}\n")


def test_s2st_with_different_speakers():
    """Test S2ST with different speaker IDs"""

    input_audio = "/tmp/chinese_speech.wav"
    target_lang = "eng"

    print("=" * 60)
    print("Testing S2ST with Different Speaker IDs")
    print("=" * 60)
    print(f"Input audio: {input_audio}")
    print(f"Target language: {target_lang}\n")

    # Test with speaker IDs 0 and 100
    speaker_ids = [0, 100]

    for speaker_id in speaker_ids:
        print(f"Testing speaker_id={speaker_id}...")

        with open(input_audio, 'rb') as f:
            response = requests.post(
                f"{API_URL}/v1/speech-to-speech-translation",
                files={"audio": ("input.wav", f, "audio/wav")},
                data={
                    "target_lang": target_lang,
                    "response_format": "json",
                    "speaker_id": speaker_id
                }
            )

        if response.status_code == 200:
            result = response.json()

            # Decode and save audio
            import base64
            audio_bytes = base64.b64decode(result['output_audio_base64'])

            output_file = f"/tmp/s2st_speaker_{speaker_id}.wav"
            with open(output_file, 'wb') as f:
                f.write(audio_bytes)

            print(f"  ✓ Success")
            print(f"    Processing time: {result['processing_time']}s")
            print(f"    Translated text: {result['output_text']}")
            print(f"    Saved to: {output_file}\n")
        else:
            print(f"  ✗ Failed: {response.status_code}")
            print(f"    {response.text}\n")


if __name__ == "__main__":
    print("\nSeamlessM4T Speaker ID Parameter Test\n")

    # Test TTS
    test_tts_with_different_speakers()

    # Test S2ST
    test_s2st_with_different_speakers()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nNote: Listen to the generated audio files to hear")
    print("different voices for different speaker_id values.")

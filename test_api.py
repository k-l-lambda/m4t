#!/usr/bin/env python3
"""
Test script for SeamlessM4T API endpoints
"""
import requests
import base64
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint...")
    print("="*60)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed!")


def test_languages():
    """Test languages endpoint"""
    print("\n" + "="*60)
    print("Testing /languages endpoint...")
    print("="*60)

    response = requests.get(f"{BASE_URL}/languages")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Total languages: {data['count']}")
    print(f"Sample languages: {list(data['languages'].items())[:5]}")

    assert response.status_code == 200
    assert "jpn" in data["languages"]
    assert "cmn" in data["languages"]
    print("✅ Languages endpoint passed!")


def test_tasks():
    """Test tasks endpoint"""
    print("\n" + "="*60)
    print("Testing /tasks endpoint...")
    print("="*60)

    response = requests.get(f"{BASE_URL}/tasks")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    print("✅ Tasks endpoint passed!")


def test_text_to_text_translation():
    """Test text-to-text translation (T2TT)"""
    print("\n" + "="*60)
    print("Testing Text-to-Text Translation (Japanese → Chinese)...")
    print("="*60)

    # Test data
    test_text = "こんにちは、今日は良い天気ですね。"
    print(f"Input text: {test_text}")

    payload = {
        "text": test_text,
        "source_lang": "jpn",
        "target_lang": "cmn"
    }

    response = requests.post(
        f"{BASE_URL}/v1/text-to-text-translation",
        json=payload
    )

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Translated text: {result['output_text']}")
        print(f"Processing time: {result['processing_time']}s")
        print("✅ T2TT test passed!")
    else:
        print(f"❌ Error: {response.text}")
        raise Exception("T2TT test failed")


def test_speech_to_text_translation(audio_file: Path):
    """Test speech-to-text translation (S2TT)"""
    print("\n" + "="*60)
    print("Testing Speech-to-Text Translation (Japanese audio → Chinese text)...")
    print("="*60)

    if not audio_file.exists():
        print(f"⚠️  Audio file not found: {audio_file}")
        print("Skipping S2TT test")
        return

    with open(audio_file, "rb") as f:
        files = {"audio": (audio_file.name, f, "audio/wav")}
        data = {
            "target_lang": "cmn",
            "source_lang": "jpn"
        }

        response = requests.post(
            f"{BASE_URL}/v1/speech-to-text-translation",
            files=files,
            data=data
        )

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Translated text: {result['output_text']}")
        print(f"Input duration: {result['input_duration']}s")
        print(f"Processing time: {result['processing_time']}s")
        print("✅ S2TT test passed!")
    else:
        print(f"❌ Error: {response.text}")
        raise Exception("S2TT test failed")


def test_transcribe(audio_file: Path):
    """Test transcription (ASR)"""
    print("\n" + "="*60)
    print("Testing Transcription (Japanese audio → Japanese text)...")
    print("="*60)

    if not audio_file.exists():
        print(f"⚠️  Audio file not found: {audio_file}")
        print("Skipping ASR test")
        return

    with open(audio_file, "rb") as f:
        files = {"audio": (audio_file.name, f, "audio/wav")}
        data = {"language": "jpn"}

        response = requests.post(
            f"{BASE_URL}/v1/transcribe",
            files=files,
            data=data
        )

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Transcribed text: {result['output_text']}")
        print(f"Input duration: {result['input_duration']}s")
        print(f"Processing time: {result['processing_time']}s")
        print("✅ ASR test passed!")
    else:
        print(f"❌ Error: {response.text}")
        raise Exception("ASR test failed")


def test_speech_to_speech_translation(audio_file: Path, output_file: Path):
    """Test speech-to-speech translation (S2ST)"""
    print("\n" + "="*60)
    print("Testing Speech-to-Speech Translation (Japanese audio → Chinese audio)...")
    print("="*60)

    if not audio_file.exists():
        print(f"⚠️  Audio file not found: {audio_file}")
        print("Skipping S2ST test")
        return

    # Test with audio response
    with open(audio_file, "rb") as f:
        files = {"audio": (audio_file.name, f, "audio/wav")}
        data = {
            "target_lang": "cmn",
            "source_lang": "jpn",
            "response_format": "audio"
        }

        response = requests.post(
            f"{BASE_URL}/v1/speech-to-speech-translation",
            files=files,
            data=data
        )

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        # Save audio output
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✅ S2ST test passed!")
        print(f"Output audio saved to: {output_file}")
    else:
        print(f"❌ Error: {response.text}")
        raise Exception("S2ST test failed")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SeamlessM4T API Test Suite")
    print("="*60)

    # Test basic endpoints
    test_health()
    test_languages()
    test_tasks()

    # Test translation endpoints
    test_text_to_text_translation()

    # Audio tests (optional, requires audio file)
    audio_file = Path("examples/test_audio.wav")
    output_file = Path("examples/output_audio.wav")

    if audio_file.exists():
        test_speech_to_text_translation(audio_file)
        test_transcribe(audio_file)
        test_speech_to_speech_translation(audio_file, output_file)
    else:
        print("\n" + "="*60)
        print("⚠️  No test audio file found at examples/test_audio.wav")
        print("Audio-based tests will be skipped")
        print("To test audio endpoints, add a Japanese audio file at:")
        print(f"  {audio_file.absolute()}")
        print("="*60)

    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()

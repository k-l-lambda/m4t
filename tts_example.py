#!/usr/bin/env python3
"""
Example: Text-to-Speech using SeamlessM4T API
Converts text to speech audio in any supported language
"""
import requests
import json
import numpy as np
import soundfile as sf

API_URL = "http://localhost:8000"

def text_to_speech(text, language, output_file):
    """
    Convert text to speech

    Args:
        text: Text to convert
        language: Language code (e.g., 'cmn', 'eng', 'jpn')
        output_file: Output audio file path
    """
    response = requests.post(
        f"{API_URL}/v1/text-to-speech",
        json={
            "text": text,
            "source_lang": language
        }
    )

    if response.status_code == 200:
        result = response.json()

        print(f"✓ TTS Success")
        print(f"  Language: {result['language']}")
        print(f"  Input text: {result['input_text']}")
        print(f"  Sample rate: {result['output_sample_rate']} Hz")
        print(f"  Processing time: {result['processing_time']}s")

        # Convert audio list to numpy array and save
        audio_array = np.array(result['output_audio'], dtype=np.float32)
        sample_rate = result['output_sample_rate']

        sf.write(output_file, audio_array, sample_rate)
        print(f"  Audio saved to: {output_file}")

        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("SeamlessM4T Text-to-Speech Examples")
    print("=" * 60)
    print()

    # Example 1: Chinese TTS
    print("1. Chinese TTS:")
    text_to_speech(
        text="你好，今天天气很好",
        language="cmn",
        output_file="/tmp/chinese_speech.wav"
    )
    print()

    # Example 2: English TTS
    print("2. English TTS:")
    text_to_speech(
        text="Hello, how are you today?",
        language="eng",
        output_file="/tmp/english_speech.wav"
    )
    print()

    # Example 3: Japanese TTS
    print("3. Japanese TTS:")
    text_to_speech(
        text="こんにちは、今日はいい天気ですね",
        language="jpn",
        output_file="/tmp/japanese_speech.wav"
    )
    print()

    print("=" * 60)
    print("All TTS examples completed!")
    print("=" * 60)

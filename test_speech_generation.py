#!/usr/bin/env python3
"""
Test Speech-to-Speech Translation (S2ST) - generates speech output
"""
import requests
import json
import numpy as np
import soundfile as sf

# Test audio file (you need to provide a real audio file)
# For now, let's create a simple test with the API

# Example 1: Test S2ST endpoint (requires audio input)
print("=" * 60)
print("Speech-to-Speech Translation Test")
print("=" * 60)

# You need an actual audio file to test S2ST
# Example usage:
print("""
# S2ST API - Translates speech to speech in another language
curl -X POST "http://localhost:8000/v1/speech-to-speech-translation" \\
  -F "audio=@japanese_audio.wav" \\
  -F "target_lang=cmn" \\
  --output output_chinese_speech.wav

# This will:
# 1. Take Japanese audio as input
# 2. Generate Chinese speech as output
# 3. Save the output audio to output_chinese_speech.wav
""")

print("\n" + "=" * 60)
print("Testing S2ST with Python requests")
print("=" * 60)

# If you have an audio file, you can test like this:
audio_file_path = "/tmp/test_audio.wav"  # Replace with actual path
api_url = "http://localhost:8000/v1/speech-to-speech-translation"

print(f"""
# Python code to test S2ST:
import requests

with open('{audio_file_path}', 'rb') as f:
    files = {{'audio': f}}
    data = {{'target_lang': 'cmn', 'source_lang': 'jpn'}}
    response = requests.post('{api_url}', files=files, data=data)

    # Save the output speech
    if response.status_code == 200:
        result = response.json()
        audio_data = result['output_audio']  # numpy array
        sample_rate = result['output_sample_rate']

        # Convert to audio file
        import numpy as np
        import soundfile as sf
        audio_array = np.array(audio_data, dtype=np.float32)
        sf.write('output_speech.wav', audio_array, sample_rate)
        print(f"Speech generated: output_speech.wav")
        print(f"Translation text: {{result['output_text']}}")
""")

print("\n" + "=" * 60)
print("Supported Speech Output Languages")
print("=" * 60)
print("""
The model can generate speech in these languages:
- English (eng)
- Chinese Simplified (cmn)
- Chinese Traditional (cmn_Hant)
- Japanese (jpn)
- Korean (kor)
- Spanish (spa)
- French (fra)
- German (deu)
- Italian (ita)
- Portuguese (por)
- Russian (rus)
- Arabic (arb)
- And many more...
""")

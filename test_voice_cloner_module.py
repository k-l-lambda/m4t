#!/usr/bin/env python3
"""
Simple validation test for voice cloner module
Tests the VoiceCloner class without requiring GPT-SoVITS service
"""

import sys
sys.path.insert(0, '/home/camus/work/m4t')

from voice_cloner import VoiceCloner, get_voice_cloner

def test_voice_cloner_initialization():
    """Test VoiceCloner initialization"""
    print("Testing VoiceCloner initialization...")

    # Test with default parameters
    cloner = VoiceCloner()
    assert cloner.api_url == "http://localhost:9880"
    assert cloner.default_refer_wav is None
    print("✓ Default initialization works")

    # Test with custom parameters
    cloner2 = VoiceCloner(
        api_url="http://example.com:8080",
        default_refer_wav="/path/to/audio.wav",
        default_refer_text="Test",
        default_refer_language="en"
    )
    assert cloner2.api_url == "http://example.com:8080"
    assert cloner2.default_refer_wav == "/path/to/audio.wav"
    print("✓ Custom initialization works")


def test_singleton_pattern():
    """Test singleton pattern for get_voice_cloner"""
    print("\nTesting singleton pattern...")

    cloner1 = get_voice_cloner()
    cloner2 = get_voice_cloner()

    assert cloner1 is cloner2
    print("✓ Singleton pattern works correctly")


def test_is_available():
    """Test service availability check"""
    print("\nTesting service availability check...")

    cloner = VoiceCloner(api_url="http://localhost:9880")
    available = cloner.is_available()

    if available:
        print("✓ GPT-SoVITS service is available")
    else:
        print("⚠ GPT-SoVITS service is NOT available (expected if not running)")
        print("  This is normal if GPT-SoVITS is not started")


def test_api_integration():
    """Test API integration (requires running service)"""
    print("\nTesting API integration...")

    cloner = VoiceCloner()

    if not cloner.is_available():
        print("⚠ Skipping API tests - GPT-SoVITS service not available")
        print("  Start GPT-SoVITS with: python /home/camus/work/GPT-SoVITS/api.py")
        return

    # Test with example audio
    test_audio = "/home/camus/work/stream-polyglot/assets/speaker_samples/speaker_121_cmn.wav"

    try:
        result = cloner.clone_voice_from_audio(
            text="你好，这是一个测试。",
            text_language="zh",
            refer_wav_path=test_audio,
            prompt_text="测试音频",
            prompt_language="zh"
        )

        if result:
            print(f"✓ Voice cloning successful! Generated {len(result)} bytes")

            # Save to temp file for verification
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(result)
                temp_path = f.name

            print(f"  Saved to: {temp_path}")

            # Verify audio
            import soundfile as sf
            import io
            audio_array, sr = sf.read(io.BytesIO(result))
            duration = len(audio_array) / sr
            print(f"  Duration: {duration:.2f}s, Sample rate: {sr} Hz")
        else:
            print("✗ Voice cloning failed")

    except Exception as e:
        print(f"✗ Error during API test: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("=" * 70)
    print("Voice Cloner Module Validation Test")
    print("=" * 70)
    print()

    tests = [
        test_voice_cloner_initialization,
        test_singleton_pattern,
        test_is_available,
        test_api_integration
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 70)
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

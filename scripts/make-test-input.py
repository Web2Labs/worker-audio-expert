#!/usr/bin/env python3
"""Build test_input.json with our real audio chunk encoded as base64.
   Use this to test the worker locally on actual niche-models-01 audio.
"""
import base64
import json
import sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: make-test-input.py <path-to-wav> <output-json>")
    sys.exit(1)

wav_path = sys.argv[1]
output_path = sys.argv[2]

with open(wav_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("ascii")

test_input = {
    "input": {
        "audio_base64": audio_b64,
        "model": "large-v3",
        "transcription": "plain_text",
        "translate": False,
        "word_timestamps": True,
        "force_align": True,
        "temperature": 0,
        "best_of": 5,
        "beam_size": 5,
        "suppress_tokens": "-1",
        "condition_on_previous_text": False,
        "temperature_increment_on_fallback": 0.2,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1,
        "no_speech_threshold": 0.6,
        "enable_vad": False,
    }
}

with open(output_path, "w") as f:
    json.dump(test_input, f)

print(f"Wrote {output_path} ({len(audio_b64)} chars b64, {len(audio_b64) * 3 // 4 // 1024 // 1024:.1f}MB audio)")

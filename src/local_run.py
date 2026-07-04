#!/usr/bin/env python3
"""Local-only runner: load test_input.json, call predict directly, write
JSON-clean output to /output/result.json. Avoids importing rp_handler
because that triggers runpod's auto-test-mode at import.

Used for the NP-SBV2 multi-clip local test.
"""
import json
import sys
import base64
import tempfile

import numpy as np

sys.path.insert(0, "/")
import predict  # noqa: E402


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def base64_to_tempfile(b64: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(base64.b64decode(b64))
        return f.name


def main() -> int:
    print("[local_run] loading model...", flush=True)
    p = predict.Predictor()
    p.setup()

    print("[local_run] reading test_input.json...", flush=True)
    with open("/test_input.json") as f:
        payload = json.load(f)
    job_input = payload["input"]

    audio_path = base64_to_tempfile(job_input["audio_base64"])
    print(f"[local_run] running predict on {audio_path}", flush=True)

    result = p.predict(
        audio=audio_path,
        model_name=job_input.get("model", "large-v3"),
        transcription=job_input.get("transcription", "plain_text"),
        translation=job_input.get("translation"),
        translate=job_input.get("translate", False),
        language=job_input.get("language"),
        temperature=job_input.get("temperature", 0),
        best_of=job_input.get("best_of", 5),
        beam_size=job_input.get("beam_size", 5),
        patience=job_input.get("patience", 1.0),
        length_penalty=job_input.get("length_penalty", 0.0),
        suppress_tokens=job_input.get("suppress_tokens", "-1"),
        initial_prompt=job_input.get("initial_prompt"),
        condition_on_previous_text=job_input.get("condition_on_previous_text", True),
        temperature_increment_on_fallback=job_input.get("temperature_increment_on_fallback", 0.2),
        compression_ratio_threshold=job_input.get("compression_ratio_threshold", 2.4),
        logprob_threshold=job_input.get("logprob_threshold", -1.0),
        no_speech_threshold=job_input.get("no_speech_threshold", 0.6),
        enable_vad=job_input.get("enable_vad", False),
        word_timestamps=job_input.get("word_timestamps", False),
        clap_queries=job_input.get("clap_queries"),
        force_align=job_input.get("force_align", False),
    )

    clean = to_jsonable(result)
    out_path = "/output/result.json"
    with open(out_path, "w") as f:
        json.dump(clean, f)

    n_words = len(clean.get("word_timestamps", []))
    n_with_onset = sum(1 for w in clean.get("word_timestamps", []) if "onset_start" in w)
    print(f"[local_run] wrote {out_path}", flush=True)
    print(f"[local_run] words: {n_words}, with onset_start: {n_with_onset}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

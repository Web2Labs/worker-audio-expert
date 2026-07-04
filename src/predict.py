"""
Audio Specialist — Whisper transcription + CLAP audio-text scoring.

Runs both models on the same audio in a single GPU worker call:
1. Faster-Whisper: speech-to-text with word-level timing + probability
2. CLAP: audio-text similarity scoring against natural language queries

Based on the Predictor class from the original Whisper worker,
extended with CLAP scoring for Web2Labs Studio.
"""

import gc
import threading
import numpy as np

from runpod.serverless.utils import rp_cuda

from faster_whisper import WhisperModel
from faster_whisper.utils import format_timestamp
from clap_scorer import ClapScorer
from aligner import Wav2Vec2Aligner

def parse_suppress_tokens(raw):
    """
    Parse the `suppress_tokens` input into the list of ints faster-whisper
    expects. Accepts the documented comma-separated string form ("-1" or
    "-1,0,50257"), a ready-made list, or None. Falls back to [-1] (whisper's
    "default non-speech suppression set" sentinel) on anything unparseable —
    matching the worker's historical behavior, which hardcoded [-1].
    """
    if raw is None:
        return [-1]
    if isinstance(raw, (list, tuple)):
        try:
            return [int(t) for t in raw]
        except (TypeError, ValueError):
            return [-1]
    try:
        return [int(t.strip()) for t in str(raw).split(",") if t.strip() != ""]
    except ValueError:
        return [-1]


# Define available models (for validation)
AVAILABLE_MODELS = {
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "distil-large-v2",
    "distil-large-v3",
    "distil-large-v3.5",
    "turbo",
}


class Predictor:
    """A Predictor class for the Whisper model with lazy loading"""

    def __init__(self):
        """Initializes the predictor with no models loaded."""
        self.models = {}
        self.model_lock = (
            threading.Lock()
        )  # Lock for thread-safe model loading/unloading
        self.clap_scorer = ClapScorer()
        self.aligner = Wav2Vec2Aligner()  # lazy-loaded on first force_align call

    def setup(self):
        """No models are pre-loaded. Setup is minimal."""
        pass

    def predict(
        self,
        audio,
        model_name="base",
        transcription="plain_text",
        translate=False,
        translation="plain_text",  # Added in a previous PR
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        # 1.0 = ctranslate2's neutral default. None crashes ctranslate2's
        # generate() with a TypeError — the handler always passes the schema
        # default so production never hit it, but direct callers did.
        length_penalty=1.0,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        enable_vad=False,
        word_timestamps=False,
        clap_queries=None,
        force_align=False,
    ):
        """
        Run a single prediction on the model, loading/unloading models as needed.
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. Available models are: {AVAILABLE_MODELS}"
            )

        with self.model_lock:
            model = None
            if model_name not in self.models:
                # Unload existing model if necessary
                if self.models:
                    existing_model_name = list(self.models.keys())[0]
                    print(f"Unloading model: {existing_model_name}...")
                    # Remove reference and clear dict
                    del self.models[existing_model_name]
                    self.models.clear()
                    # Hint Python to release memory
                    gc.collect()
                    if rp_cuda.is_available():
                        # If using PyTorch models, you might call torch.cuda.empty_cache()
                        # FasterWhisper uses CTranslate2; explicit cache clearing might not be needed
                        # but gc.collect() is generally helpful.
                        pass
                    print(f"Model {existing_model_name} unloaded.")

                # Load the requested model
                print(f"Loading model: {model_name}...")
                try:
                    loaded_model = WhisperModel(
                        model_name,
                        device="cuda" if rp_cuda.is_available() else "cpu",
                        compute_type="float16" if rp_cuda.is_available() else "int8",
                    )
                    self.models[model_name] = loaded_model
                    model = loaded_model
                    print(f"Model {model_name} loaded successfully.")
                except Exception as e:
                    print(f"Error loading model {model_name}: {e}")
                    raise ValueError(f"Failed to load model {model_name}: {e}") from e
            else:
                # Model already loaded
                model = self.models[model_name]
                print(f"Using already loaded model: {model_name}")

            # Ensure model is loaded before proceeding
            if model is None:
                raise RuntimeError(
                    f"Model {model_name} could not be loaded or retrieved."
                )

        # Model is now loaded and ready, proceed with prediction (outside the lock?)
        # Consider if transcribe is thread-safe or if it should also be within the lock
        # For now, keeping transcribe outside as it's CPU/GPU bound work

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        # Note: FasterWhisper's transcribe might release the GIL, potentially allowing
        # other threads to acquire the model_lock if transcribe is lengthy.
        # If issues arise, the lock might need to encompass the transcribe call too.
        segments_generator, info = model.transcribe(
            str(audio),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=parse_suppress_tokens(suppress_tokens),
            without_timestamps=False,
            max_initial_timestamp=1.0,
            word_timestamps=word_timestamps,
            vad_filter=enable_vad,
        )

        segments = list(segments_generator)

        # Format transcription
        transcription_output = format_segments(transcription, segments)

        # Handle translation if requested
        translation_output = None
        if translate:
            translation_segments, _ = model.transcribe(
                str(audio),
                task="translate",
                temperature=temperature,  # Reuse temperature settings for translation
            )
            translation_output = format_segments(
                translation, list(translation_segments)
            )

        results = {
            "segments": serialize_segments(segments),
            "detected_language": info.language,
            "transcription": transcription_output,
            "translation": translation_output,
            "device": "cuda" if rp_cuda.is_available() else "cpu",
            "model": model_name,
        }

        if word_timestamps:
            word_timestamps_list = []
            for segment in segments:
                # segment.words can be None for a no-speech segment depending on
                # the faster-whisper version — guard instead of crashing the job.
                for word in (segment.words or []):
                    word_entry = {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                    word_timestamps_list.append(word_entry)
            results["word_timestamps"] = word_timestamps_list

            # Wav2vec2 forced alignment — re-times each word against actual audio
            # (sub-50ms accuracy vs Whisper's 100-300ms cross-attention timing).
            # Only runs if explicitly requested via force_align: true input.
            if force_align and word_timestamps_list:
                print(
                    f"[Predictor] Running wav2vec2 forced alignment on "
                    f"{len(word_timestamps_list)} words..."
                )
                device = "cuda" if rp_cuda.is_available() else "cpu"
                self.aligner.setup(device=device)
                aligned_list = self.aligner.align(str(audio), word_timestamps_list)
                # Replace the original timestamps with the aligned ones.
                # The original (cross-attention) timing is gone — if you want both,
                # this is where to add a `word_timestamps_original` field.
                results["word_timestamps"] = aligned_list
                results["word_timestamps_aligned"] = True  # flag so callers know

        # CLAP audio-text similarity scoring (optional — only if queries provided)
        if clap_queries and isinstance(clap_queries, dict) and len(clap_queries) > 0:
            print(f"[AudioSpecialist] Running CLAP scoring with {len(clap_queries)} queries...")
            clap_result = self.clap_scorer.score(str(audio), clap_queries)
            if clap_result:
                results["clap_scores"] = clap_result
                print(f"[AudioSpecialist] CLAP scoring complete: {clap_result['duration']}s, device={clap_result['device']}")
            else:
                results["clap_scores"] = None
                print("[AudioSpecialist] CLAP scoring returned no results")

        return results


def serialize_segments(transcript):
    """
    Serialize the segments to be returned in the API response.
    """
    return [
        {
            "id": segment.id,
            "seek": segment.seek,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "tokens": segment.tokens,
            "temperature": segment.temperature,
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob,
        }
        for segment in transcript
    ]


def format_segments(format_type, segments):
    """
    Format the segments to the desired format
    """

    if format_type == "plain_text":
        return " ".join([segment.text.lstrip() for segment in segments])
    elif format_type == "formatted_text":
        return "\n".join([segment.text.lstrip() for segment in segments])
    elif format_type == "srt":
        return write_srt(segments)
    elif format_type == "vtt":  # Added VTT case
        return write_vtt(segments)
    else:  # Default or unknown format
        print(f"Warning: Unknown format '{format_type}', defaulting to plain text.")
        return " ".join([segment.text.lstrip() for segment in segments])


def write_vtt(transcript):
    """
    Write the transcript in VTT format.
    """
    result = ""

    for segment in transcript:
        # Using the consistent timestamp format from previous PR
        result += f"{format_timestamp(segment.start, always_include_hours=True)} --> {format_timestamp(segment.end, always_include_hours=True)}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result


def write_srt(transcript):
    """
    Write the transcript in SRT format.
    """
    result = ""

    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result

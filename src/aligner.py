"""Wav2vec2 forced alignment for word-level timestamps (NP-SBV1, 2026-05-04).

Takes faster-whisper word_timestamps + audio file, re-times each word against
the actual audio waveform using torchaudio's CTC forced alignment with a
wav2vec2 acoustic model.

Reduces word-timing error from Whisper's ~100-300ms (cross-attention-based) to
~30-50ms (forced alignment against actual acoustic frames). Most importantly,
it eliminates the "abrupt mid-sentence cut" effect at clip boundaries by
ensuring word_start matches the actual audio onset.

Process:
  1. Load wav2vec2-large-960h CTC model (one-time)
  2. Chunk audio into 60s windows with 5s overlap
  3. Per chunk:
     a. Forward pass → CTC log-prob emissions (shape T x vocab)
     b. Tokenize each Whisper word's text to char-level indices
     c. torchaudio.functional.forced_align finds optimal frame→token mapping
     d. Convert frame indices → seconds (relative to chunk start)
  4. Stitch chunks, dropping overlap regions
  5. Words with un-tokenizable chars (numbers, special) keep original timing

Notes:
  - We use the FASTER-WHISPER text as the source of truth (already cleaned by
    the no-VAD + no-condition_on_previous_text config). Forced alignment ONLY
    re-times — it doesn't change the words.
  - Model: torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H (~1.2 GB).
    English-only, librispeech-trained. No HuggingFace token required.
  - Memory: emissions tensor is (T, 32) ~5KB/sec. A 60s chunk uses ~300KB.
  - Speed: ~0.05x realtime on a single A40. 60s chunk = ~3s wall.
"""

import re
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_ASR_LARGE_LV60K_960H as BUNDLE


# CTC vocab for the model. Includes A-Z, ', |, and a few special tokens.
# Blank is at index 0 in this model's vocab.
BLANK_IDX = 0

# Words closer than this to a chunk's audio edge are left for a neighboring
# chunk (or keep original timing at the very start/end of the file). Absorbs
# Whisper's own 100-300ms timing error in the selection windows.
EDGE_MARGIN_SEC = 0.5


class Wav2Vec2Aligner:
    """Forced alignment via wav2vec2 CTC. Lazy-loaded on first call."""

    def __init__(self):
        self.model = None
        self.labels: Optional[Tuple[str, ...]] = None
        self.label_to_idx: Optional[dict] = None
        self.sample_rate: Optional[int] = None
        self.device: Optional[str] = None
        self.word_separator_idx: Optional[int] = None

    def setup(self, device: str = "cuda"):
        """Load the model. Idempotent — safe to call multiple times."""
        if self.model is not None and self.device == device:
            return
        print(f"[Wav2Vec2Aligner] loading WAV2VEC2_ASR_LARGE_LV60K_960H on {device}...", flush=True)
        self.device = device
        self.model = BUNDLE.get_model().to(device).eval()
        self.labels = BUNDLE.get_labels()
        self.sample_rate = BUNDLE.sample_rate
        self.label_to_idx = {ch: i for i, ch in enumerate(self.labels)}
        # Word boundary char in this model's vocab is '|'
        self.word_separator_idx = self.label_to_idx.get("|", 4)
        print(
            f"[Wav2Vec2Aligner] loaded | vocab={len(self.labels)} sr={self.sample_rate} "
            f"blank={BLANK_IDX} word_sep={self.word_separator_idx}",
            flush=True,
        )

    @staticmethod
    def normalize_word(word: str) -> str:
        """Strip non-alpha (keep apostrophe), uppercase. Returns "" if no chars."""
        return re.sub(r"[^A-Za-z']", "", word).upper()

    def align(
        self,
        audio_path: str,
        words: List[dict],
        chunk_sec: float = 60.0,
        overlap_sec: float = 5.0,
    ) -> List[dict]:
        """Re-time each word against the audio.

        words: list of {"word": str, "start": float, "end": float, ...}
               from faster-whisper transcribe(word_timestamps=True).
        Returns: same list with start/end re-timed via forced alignment.
                 Words that couldn't be aligned (no valid chars, fell off chunk
                 boundaries) keep their original timing — preserving order +
                 count exactly.

        chunk_sec: how much audio to process at once (frames in memory).
        overlap_sec: chunk overlap so words near chunk seams have full context.

        Word→chunk assignment: every word STARTING in a chunk's window is part
        of that chunk's CTC target sequence (so all speech in the chunk's audio
        has tokens to map onto), but a word's timing is WRITTEN by exactly one
        chunk — the first chunk that fully contains it with EDGE_MARGIN_SEC to
        spare (the last chunk, having no successor, writes everything it
        contains). Words already written act as anchors only; re-aligning them
        at the LEFT EDGE of the next chunk (where preceding audio is truncated)
        measurably degrades their timing, and that degraded version used to
        overwrite the good mid-chunk one. Words too close to a chunk's RIGHT
        edge anchor there and are written by the next chunk, which contains
        them mid-chunk thanks to the overlap.
        """
        if not words:
            return words
        if self.model is None:
            self.setup()

        # Load audio at the model's sample rate. Use librosa instead of
        # torchaudio.load() because torchaudio 2.7+ requires the torchcodec
        # backend, which isn't installed (we'd need an extra ~50MB dependency
        # for what librosa already does on CPU). librosa handles resampling
        # and mono conversion in one call.
        audio_np, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        # Shape: (samples,) — convert to torch (1, samples) for the model
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
        total_samples = waveform.shape[1]
        total_dur = total_samples / self.sample_rate

        # Output buffer — will overwrite for each word
        aligned: List[Optional[dict]] = [None] * len(words)

        chunk_step = chunk_sec - overlap_sec
        if chunk_step <= 0:
            raise ValueError(f"overlap_sec ({overlap_sec}) must be < chunk_sec ({chunk_sec})")

        chunk_idx = 0
        chunk_start_sec = 0.0
        chunks_processed = 0
        words_aligned = 0
        words_unalignable = 0

        while chunk_start_sec < total_dur:
            chunk_end_sec = min(chunk_start_sec + chunk_sec, total_dur)
            chunk_dur = chunk_end_sec - chunk_start_sec
            if chunk_dur < 1.0:
                break  # Skip tiny tail chunks
            is_last_chunk = chunk_end_sec >= total_dur - 1e-6

            # Every word STARTING in this chunk's window goes into the CTC
            # target sequence — INCLUDING words already written by the previous
            # chunk. The overlap audio contains their speech; without their
            # tokens as anchors, forced_align smears the first unwritten word's
            # start across that target-less speech (observed: a word at 59.3s
            # dragged to the 55.0s chunk boundary). Anchors are aligned but
            # their timings are discarded — only `writable` words get written.
            words_in_chunk = [
                (i, w) for i, w in enumerate(words)
                if chunk_start_sec <= w["start"] < chunk_end_sec - EDGE_MARGIN_SEC
            ]

            # A word's timing is written by the FIRST chunk that fully contains
            # it (end inside the margin too; the last chunk has no successor,
            # so it writes everything it contains). Words at the right edge
            # anchor here and are written by the next chunk, where the overlap
            # places them mid-chunk with full acoustic context instead of
            # force-squeezing their tokens into truncated audio.
            writable = {
                i for i, w in words_in_chunk
                if aligned[i] is None
                and (is_last_chunk or w["end"] <= chunk_end_sec - EDGE_MARGIN_SEC)
            }

            if not writable:
                chunk_start_sec += chunk_step
                chunk_idx += 1
                continue

            # Build flat list of CHAR tokens (no word separators — wav2vec2 model
            # implicitly predicts | between words from the audio). Track per-word
            # char counts so we can re-group TokenSpans back into words later.
            # Per torchaudio's CTC forced alignment tutorial.
            tokens: List[int] = []
            word_char_counts: List[Tuple[int, int, str]] = []  # (n_chars, words_idx, original_text)
            for words_idx, w in words_in_chunk:
                clean = self.normalize_word(w["word"])
                n_chars_added = 0
                if clean:
                    for ch in clean:
                        if ch in self.label_to_idx:
                            tokens.append(self.label_to_idx[ch])
                            n_chars_added += 1
                if n_chars_added == 0:
                    # No CTC-representable chars (numbers, symbols). Keeps its
                    # original whisper timing; contributes nothing as an anchor.
                    if aligned[words_idx] is None:
                        words_unalignable += 1
                        aligned[words_idx] = w
                    writable.discard(words_idx)
                    continue
                word_char_counts.append((n_chars_added, words_idx, w["word"]))

            if not writable or not word_char_counts:
                chunk_start_sec += chunk_step
                chunk_idx += 1
                continue

            # Forward pass — get CTC emissions. Done AFTER the word selection so
            # chunks with nothing to align (silence, music) skip the GPU work.
            sample_start = int(chunk_start_sec * self.sample_rate)
            sample_end = int(chunk_end_sec * self.sample_rate)
            chunk_audio = waveform[:, sample_start:sample_end].to(self.device)

            with torch.inference_mode():
                emissions, _ = self.model(chunk_audio)
            emissions = torch.log_softmax(emissions, dim=-1)
            emission = emissions[0]  # (T, vocab)
            sec_per_frame = chunk_dur / emission.shape[0]

            targets = torch.tensor([tokens], device=self.device, dtype=torch.int32)

            # Forced align — returns frame-level alignment (which target-token is
            # at each emission frame, may include blanks + repeats).
            try:
                aligned_tokens, alignment_scores = torchaudio.functional.forced_align(
                    emission.unsqueeze(0), targets, blank=BLANK_IDX,
                )
            except RuntimeError as e:
                print(f"[Wav2Vec2Aligner] chunk {chunk_idx} forced_align failed: {e}", flush=True)
                # Leave the words unmarked: those in the overlap region get a
                # second chance in the next chunk; the rest fall back to their
                # original timing in the final sweep below.
                chunk_start_sec += chunk_step
                chunk_idx += 1
                continue

            # Collapse repeats/blanks into per-token spans. Each TokenSpan has
            # .token (target index), .start (frame), .end (frame), .score.
            # One TokenSpan per CHAR in the targets sequence, in order.
            token_spans = torchaudio.functional.merge_tokens(
                aligned_tokens[0], alignment_scores[0]
            )

            # token_spans count should equal len(targets). If forced_align skipped
            # some chars (rare), fall back to original timing for affected words.
            if len(token_spans) != len(tokens):
                # Sanity check — log + skip chunk to avoid wrong alignments.
                # Unmarked words are retried by the next chunk or swept to
                # original timing at the end.
                print(
                    f"[Wav2Vec2Aligner] chunk {chunk_idx}: token_spans={len(token_spans)} "
                    f"vs targets={len(tokens)} — sequences out of sync, falling back",
                    flush=True,
                )
                chunk_start_sec += chunk_step
                chunk_idx += 1
                continue

            # Per-frame token assignments (includes blanks). Used to find
            # silence-run boundaries around each word for cut-friendly timing.
            aligned_tokens_arr = aligned_tokens[0].cpu()  # shape (T_emission,)
            n_emission_frames = int(aligned_tokens_arr.shape[0])

            # Re-group token_spans into per-word lists using char counts.
            cursor = 0
            for n_chars, words_idx, word_text in word_char_counts:
                spans_for_word = token_spans[cursor:cursor + n_chars]
                cursor += n_chars
                if words_idx not in writable:
                    # Anchor word — its timing was already written by an earlier
                    # chunk (or is deferred to the next); the spans only served
                    # to absorb its speech in the CTC path.
                    continue
                if not spans_for_word:
                    aligned[words_idx] = words[words_idx]
                    continue
                start_frame = spans_for_word[0].start
                end_frame = spans_for_word[-1].end

                # ── Acoustic onset/offset (Fix 3 / NP-SBV2) ─────────────────
                # The model's `start_frame` is when wav2vec2 was most CONFIDENT
                # about the first char (typically mid-phoneme). The actual
                # phoneme ONSET is in the CTC-blank run immediately before, where
                # the model was building confidence. Walking backward through
                # contiguous blank frames gives us the silence-run boundary —
                # the ideal place to CUT for clean audio without slicing
                # mid-phoneme.
                #
                # Same logic forward for the offset end.
                onset_frame = start_frame
                while (
                    onset_frame > 0
                    and int(aligned_tokens_arr[onset_frame - 1]) == BLANK_IDX
                ):
                    onset_frame -= 1
                # onset_frame now == start_frame (no preceding blanks) OR
                # the first frame of the contiguous blank run before this word
                # (which is the frame right after the previous non-blank).

                offset_frame = end_frame
                while (
                    offset_frame < n_emission_frames - 1
                    and int(aligned_tokens_arr[offset_frame + 1]) == BLANK_IDX
                ):
                    offset_frame += 1
                # offset_frame now == end_frame (no trailing blanks) OR the
                # last frame of the contiguous blank run after this word.

                abs_start = chunk_start_sec + start_frame * sec_per_frame
                abs_end = chunk_start_sec + end_frame * sec_per_frame
                abs_onset = chunk_start_sec + onset_frame * sec_per_frame
                # +1 because end_frame/offset_frame is inclusive
                abs_offset = chunk_start_sec + (offset_frame + 1) * sec_per_frame

                new_word = dict(words[words_idx])
                new_word["start"] = float(abs_start)
                new_word["end"] = float(abs_end)
                # Acoustic onset/offset — boundaries of the silence-run around
                # the word. Render-side can cut anywhere in [onset_start, start]
                # or [end, offset_end] without slicing mid-phoneme.
                new_word["onset_start"] = float(abs_onset)
                new_word["offset_end"] = float(abs_offset)
                aligned[words_idx] = new_word
                words_aligned += 1

            chunks_processed += 1
            chunk_start_sec += chunk_step
            chunk_idx += 1

        # Any words not aligned (off the end, between chunk overlaps) keep originals
        for i, w in enumerate(words):
            if aligned[i] is None:
                aligned[i] = w

        print(
            f"[Wav2Vec2Aligner] aligned {words_aligned}/{len(words)} words "
            f"({words_unalignable} unalignable, "
            f"{chunks_processed} chunks of {chunk_sec}s)",
            flush=True,
        )
        return aligned  # type: ignore

from faster_whisper.utils import download_model

# ── Whisper Models ──────────────────────────────────────────────────
# Pre-download models used in production.
# Other models in AVAILABLE_MODELS (predict.py) download on first request.
whisper_models = [
    "large-v3",
    "turbo",
]

for model_name in whisper_models:
    print(f"Downloading Whisper model: {model_name}...")
    download_model(model_name, cache_dir=None)
    print(f"Finished downloading {model_name}.")

# ── CLAP Model ──────────────────────────────────────────────────────
# ~1.5 GB, pre-downloaded for zero cold-start on CLAP scoring requests.
CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
print(f"Downloading CLAP model: {CLAP_MODEL_ID}...")

from transformers import ClapModel, ClapProcessor
ClapProcessor.from_pretrained(CLAP_MODEL_ID)
ClapModel.from_pretrained(CLAP_MODEL_ID)
print(f"Finished downloading CLAP model.")

# ── Wav2Vec2 Forced Alignment Model ──────────────────────────────────
# ~1.2 GB, pre-downloaded for zero cold-start on word-level forced alignment.
# Used when input has `force_align: true`. Re-times Whisper word_timestamps
# from ~100-300ms accuracy (Whisper cross-attention) to ~30-50ms (CTC forced
# alignment against actual audio). English-only (librispeech-trained).
print("Downloading wav2vec2 alignment model: WAV2VEC2_ASR_LARGE_LV60K_960H...")
from torchaudio.pipelines import WAV2VEC2_ASR_LARGE_LV60K_960H as W2V_BUNDLE
_ = W2V_BUNDLE.get_model()  # downloads + caches the .pth into ~/.cache/torch/hub/checkpoints
print("Finished downloading wav2vec2 alignment model.")

print("All models downloaded.")

# faster-whisper turbo needs cudnn >= 9
# see https://github.com/runpod-workers/worker-faster_whisper/pull/44
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Set working directory
WORKDIR /

# Fix stale Ubuntu mirrors in the NVIDIA base image
RUN sed -i 's|http://archive.ubuntu.com|http://us.archive.ubuntu.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://us.archive.ubuntu.com|g' /etc/apt/sources.list

# Update and install system packages (combined to reduce layers)
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends \
        sudo ca-certificates git wget curl bash \
        libgl1 libx11-6 software-properties-common \
        ffmpeg build-essential libsndfile1 \
        python3.10 python3.10-dev python3.10-venv python3-pip && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA (needed for CLAP GPU scoring + wav2vec2 forced alignment).
# torchaudio is now required — used by aligner.py for the WAV2VEC2_ASR_LARGE_LV60K_960H
# pipeline that re-times Whisper word_timestamps with sub-50ms accuracy.
#
# 2026-05-23: pinned to torch==2.7.1 / torchaudio==2.7.1 on cu128 wheels.
# Unpinned cu124 worked through ~2026-05-21, then RunPod silently started
# routing "AMPERE_24" jobs to NVIDIA RTX PRO 6000 Blackwell MIG slices
# (sm_120, 2025 architecture). The cu124 wheel does not ship compiled
# torchaudio kernels for sm_120 → "no kernel image is available for execution
# on the device" inside torchaudio.pipelines._wav2vec2.utils.layer_norm (CUDA
# kernel runtime mismatch). cu128 wheels (2.7.1+) ship sm_120 kernels and
# retain sm_86/sm_89/sm_90 → covers every GPU RunPod might assign.
# See web2labs/docs/project/relaunch/sessions/ for the diagnosis.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir huggingface_hub[hf_xet] && \
    pip install --no-cache-dir -r /requirements.txt

# Pre-download all models into the image (no network volume needed)
COPY builder/fetch_models.py /fetch_models.py
RUN python /fetch_models.py && \
    rm /fetch_models.py

# Copy handler and other code
COPY src .

# test input that will be used when the container runs outside of runpod
COPY test_input.json .

# Set default command
CMD python -u /rp_handler.py

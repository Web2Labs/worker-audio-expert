"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""
import base64
import json
import os
import shutil
import tempfile

import numpy as np

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict


MODEL = predict.Predictor()
MODEL.setup()


def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name


def cleanup_job_artifacts(job_id, base64_temp_path=None):
    '''
    Remove all per-job disk artifacts.

    download_files_from_urls saves into jobs/<job_id>/downloaded_files/ and the
    runpod SDK never deletes that directory — rp_cleanup.clean() only touches
    input_objects/, output_objects/, job_files/ and output.zip, none of which
    this worker uses. Before this function existed, every URL job leaked its
    full audio chunk (~10-25 MB per 120s WAV) on the warm worker until the
    container disk filled and jobs started failing with opaque disk errors.
    Base64 jobs leaked their tempfile the same way.
    '''
    if job_id:
        shutil.rmtree(os.path.join("jobs", str(job_id)), ignore_errors=True)
    if base64_temp_path:
        try:
            os.unlink(base64_temp_path)
        except OSError:
            pass
    rp_cleanup.clean(['input_objects'])


def to_jsonable(o):
    '''Convert numpy types to plain Python so json.dumps doesn't choke.'''
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [to_jsonable(x) for x in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    return o


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Returns:
    dict: The result of the prediction
    '''
    job_input = job['input']

    # Extract clap_queries before validation — rp_validator chokes on dict types
    raw_clap_queries = job_input.pop('clap_queries', None)

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)

        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    # Restore clap_queries after validation
    if raw_clap_queries and isinstance(raw_clap_queries, dict):
        job_input['clap_queries'] = raw_clap_queries

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64, not both'}

    base64_temp_path = None
    try:
        if job_input.get('audio', False):
            with rp_debugger.LineTimer('download_step'):
                audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]
            if not audio_input:
                # download_files_from_urls returns None for a failed download
                # (after 3 internal retries with backoff). Without this guard the
                # None reaches av.open('None') deep inside faster_whisper and
                # crashes with a misleading FileNotFoundError (2026-07-03
                # web2labs abstain-matrix post-mortem). Fail with a clear,
                # classifiable signature instead — the web2labs server
                # recognizes MEDIA_FETCH_FAILED and skips its model-fallback retry.
                return {'error': f"MEDIA_FETCH_FAILED: could not download audio from {job_input['audio']}"}

        if job_input.get('audio_base64', False):
            base64_temp_path = base64_to_tempfile(job_input['audio_base64'])
            audio_input = base64_temp_path

        with rp_debugger.LineTimer('prediction_step'):
            whisper_results = MODEL.predict(
                audio=audio_input,
                model_name=job_input["model"],
                transcription=job_input["transcription"],
                translation=job_input["translation"],
                translate=job_input["translate"],
                language=job_input["language"],
                temperature=job_input["temperature"],
                best_of=job_input["best_of"],
                beam_size=job_input["beam_size"],
                patience=job_input["patience"],
                length_penalty=job_input["length_penalty"],
                suppress_tokens=job_input.get("suppress_tokens", "-1"),
                initial_prompt=job_input["initial_prompt"],
                condition_on_previous_text=job_input["condition_on_previous_text"],
                temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
                compression_ratio_threshold=job_input["compression_ratio_threshold"],
                logprob_threshold=job_input["logprob_threshold"],
                no_speech_threshold=job_input["no_speech_threshold"],
                enable_vad=job_input["enable_vad"],
                word_timestamps=job_input["word_timestamps"],
                clap_queries=job_input.get("clap_queries"),
                force_align=job_input.get("force_align", False),
            )
    finally:
        # Always clean up job artifacts — success, MEDIA_FETCH_FAILED return, or
        # a predict() exception. Before the try/finally, any exception skipped
        # cleanup and leaked the downloaded audio on the warm worker.
        with rp_debugger.LineTimer('cleanup_step'):
            cleanup_job_artifacts(job.get('id'), base64_temp_path)

    # If TEST_OUTPUT_PATH is set (local Docker test mode), dump the full result
    # as JSON to that path so we can inspect all word timestamps without hitting
    # the docker stdout buffer limit on long audio.
    output_path = os.environ.get("TEST_OUTPUT_PATH")
    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(to_jsonable(whisper_results), f)
            print(f"[Test] Full result written to {output_path}", flush=True)
        except Exception as e:
            print(f"[Test] Failed to write {output_path}: {e}", flush=True)

    return whisper_results


runpod.serverless.start({"handler": run_whisper_job})

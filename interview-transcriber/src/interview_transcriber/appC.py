"""
High-performance Flask service:
1. Extracts audio from video
2. Transcribes with Whisper (ru-large-v3)
3. Performs speaker diarization (interviewer / interviewee)
4. Scores interviewee answers
"""

from __future__ import annotations
import os
import uuid
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict

from flask import Flask, request, jsonify
import torch, gc
torch.cuda.empty_cache()
gc.collect()    
from transformers import pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
from sentence_transformers import SentenceTransformer
from pydub import AudioSegment, silence
from pydub.effects import normalize
from moviepy import VideoFileClip
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
load_dotenv()
# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("app")

# ------------------------------------------------------------------ #
# Config (можно заменить на yaml / pydantic-settings)
# ------------------------------------------------------------------ #
from .config import (
    WHISPER_MODEL,
    DIARIZATION_MODEL,
    SEMANTIC_MODEL,
    DEVICE,
    CHUNK_LENGTH_S,
    SAMPLE_RATE,
    HEADROOM_DB,
    MIN_SILENCE_MS,
    SILENCE_THRESH_DB,
)

# ------------------------------------------------------------------ #
# NLTK resources
# ------------------------------------------------------------------ #
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ------------------------------------------------------------------ #
# Global objects
# ------------------------------------------------------------------ #
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=WHISPER_MODEL,
    device=DEVICE,
    chunk_length_s=CHUNK_LENGTH_S,
    batch_size=16,
    #model_kwargs={"language": "ru"},
)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN to use diarization.")

diarization_pipe = DiarizationPipeline.from_pretrained(
    DIARIZATION_MODEL, use_auth_token=HF_TOKEN
).to(torch.device(DEVICE))

semantic_model = SentenceTransformer(SEMANTIC_MODEL, device=DEVICE)

# ------------------------------------------------------------------ #
# Utility functions
# ------------------------------------------------------------------ #
def extract_audio(video_path: Path) -> Path:
    """Extract mono 16 kHz WAV from video."""
    tmp_audio = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.wav"
    with VideoFileClip(str(video_path)) as clip:
        if clip.audio is None:
            raise ValueError("No audio stream in video")
        clip.audio.write_audiofile(
            str(tmp_audio),
            codec="pcm_s16le",
            fps=SAMPLE_RATE,
            nbytes=2,
            buffersize=2000,
            logger=None,
        )
    log.info("Audio extracted → %s", tmp_audio)
    return tmp_audio


def optimize_audio(audio_path: Path) -> Path:
    """Pre-process audio: filter, normalize, strip silence."""
    audio = AudioSegment.from_file(audio_path).set_frame_rate(SAMPLE_RATE).set_channels(1)

    # High/low-pass
    audio = audio.low_pass_filter(8000).high_pass_filter(200)

    # Normalization
    audio = normalize(audio, headroom=HEADROOM_DB)

    # Silence trimming
    chunks = silence.detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=audio.dBFS + SILENCE_THRESH_DB,
    )
    if chunks:
        audio = audio[chunks[0][0] : chunks[-1][1]]

    out_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.wav"
    audio.export(out_path, format="wav")
    log.info("Audio optimized → %s", out_path)
    return out_path


def perform_diarization(audio_path: Path) -> List[Tuple[float, float, str]]:
    """Return list of (start, end, speaker_label)."""
    diarization = diarization_pipe(str(audio_path))
    segments = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
    log.info("Diarization finished: %d segments", len(segments))
    return segments


def transcribe_segment(audio_path: Path, start: float, end: float) -> str:
    """Transcribe audio slice."""
    audio = AudioSegment.from_file(audio_path)[start * 1000 : end * 1000]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio.export(f.name, format="wav")
        text = whisper_pipe(f.name)["text"].strip()
    os.remove(f.name)
    return text


def score_answer(answer: str, keywords: List[str], model_answer_emb) -> Dict[str, float]:
    """Compute similarity score (0-100)."""
    if not answer:
        return {"score": 0.0, "reason": "empty answer"}

    # keyword coverage
    kw_hit = sum(1 for kw in keywords if kw.lower() in answer.lower())
    kw_score = (kw_hit / max(len(keywords), 1)) * 100

    # semantic similarity
    ans_emb = semantic_model.encode(answer, convert_to_tensor=True)
    sem_score = float(torch.nn.functional.cosine_similarity(ans_emb, model_answer_emb).item()) * 100

    # weighted average
    final_score = 0.6 * sem_score + 0.4 * kw_score
    return {"score": round(final_score, 1), "semantic": round(sem_score, 1), "keywords": round(kw_score, 1)}


# ------------------------------------------------------------------ #
# Flask routes
# ------------------------------------------------------------------ #
app = Flask(__name__)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "video" not in request.files:
        return jsonify(error="No video file provided"), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify(error="Empty filename"), 400

    model_answer = request.form.get("model_answer", "")
    keywords = [k.strip() for k in request.form.get("keywords", "").split(",") if k.strip()]

    temp_files: List[Path] = []

    # --- One global progress bar (0 → 100 %) ----------------------------
    pbar = tqdm(total=100, desc="Overall", unit="%", ncols=100)

    def tick(percent: int, msg: str):
        pbar.set_description(msg)
        pbar.update(percent - pbar.n)
        log.info(msg)

    try:
        tick(5, "Saving uploaded video")
        video_tmp = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.mp4"
        file.save(video_tmp)
        temp_files.append(video_tmp)

        tick(15, "Extracting audio")
        audio_raw = extract_audio(video_tmp)
        temp_files.append(audio_raw)

        tick(25, "Optimizing audio")
        audio_clean = optimize_audio(audio_raw)
        temp_files.append(audio_clean)

        tick(35, "Speaker diarization")
        segments = perform_diarization(audio_clean)

        tick(40, "Resolving speaker roles")
        speakers = sorted({spk for _, _, spk in segments})
        interviewer = speakers[0] if speakers else "SPEAKER_00"

        model_emb = semantic_model.encode(model_answer, convert_to_tensor=True) if model_answer else None

        transcript = []
        seg_weight = 60 / max(len(segments), 1)  # remaining 60 % divided by segments

        for idx, (start, end, speaker) in enumerate(segments, 1):
            role = "interviewer" if speaker == interviewer else "interviewee"
            text = transcribe_segment(audio_clean, start, end)

            entry = {
                "start": round(start, 2),
                "end": round(end, 2),
                "speaker": role,
                "text": text,
            }
            if role == "interviewee" and model_emb is not None:
                entry["score"] = score_answer(text, keywords, model_emb)

            transcript.append(entry)
            tick(int(seg_weight * idx + 40), f"Transcribing segment {idx}/{len(segments)}")

        tick(100, "Finished")
        return jsonify(transcript=transcript)

    except Exception as e:
        log.exception("Processing failed")
        return jsonify(error=str(e)), 500

    finally:
        pbar.close()
        for path in temp_files:
            if path.exists():
                try:
                    path.unlink(missing_ok=True)
                except Exception as e:
                    log.warning("Failed to delete %s: %s", path, e)


def main() -> None:
    """Entry-point for `uv run serve`"""
    app.run(host="0.0.0.0", port=5000, debug=False)
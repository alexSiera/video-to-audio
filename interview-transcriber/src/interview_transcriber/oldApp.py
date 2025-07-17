"""
High-performance Flask service:
1. Extracts audio from video
2. Transcribes with Whisper (ru-large-v3)
3. Scores interviewee answers
"""

from __future__ import annotations
import librosa
import uuid
import logging
import tempfile
from pathlib import Path
from typing import List, Dict

from flask import Flask, request, jsonify
import torch, gc
torch.cuda.empty_cache()
gc.collect()
from transformers import BitsAndBytesConfig, WhisperProcessor, WhisperForConditionalGeneration
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
    SEMANTIC_MODEL,
    DEVICE,
    SAMPLE_RATE,
    HEADROOM_DB,
    MIN_SILENCE_MS,
    SILENCE_THRESH_DB,
)

# ------------------------------------------------------------------ #
# NLTK resources
# ------------------------------------------------------------------ #
for res in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{res}")
    except LookupError:
        nltk.download(res)

# ------------------------------------------------------------------ #
# Whisper long-form pipeline
# ------------------------------------------------------------------ #

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

whisper_model = WhisperForConditionalGeneration.from_pretrained(
    WHISPER_MODEL,
    quantization_config=bnb_config,
    device_map="auto",   
)

whisper_processor = WhisperProcessor.from_pretrained(
    WHISPER_MODEL,
    language="ru",
    task="transcribe"
)

semantic_model = SentenceTransformer(SEMANTIC_MODEL, device=DEVICE)

# ------------------------------------------------------------------ #
# Utility functions
# ------------------------------------------------------------------ #
def transcribe_long_form(audio_path: Path) -> str:
    """Return full transcript using Whisper’s own long-form algorithm."""
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    first_device = next(whisper_model.parameters()).device
    inputs = whisper_processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).input_features.to(DEVICE, dtype=torch.float16)

    predicted_ids = whisper_model.generate(
        inputs,
        language="ru",
        task="transcribe",
    )
    return whisper_processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0].strip()

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

        tick(25, "Extracting audio")
        audio_raw = extract_audio(video_tmp)
        temp_files.append(audio_raw)

        tick(45, "Optimizing audio")
        audio_clean = optimize_audio(audio_raw)
        temp_files.append(audio_clean)

        tick(60, "Transcribing full audio")
        full_text = transcribe_long_form(audio_clean)

        tick(75, "Splitting into sentences")
        sentences = sent_tokenize(full_text, language="russian")

        model_emb = semantic_model.encode(model_answer, convert_to_tensor=True) if model_answer else None

        transcript = []
        seg_weight = 25 / max(len(sentences), 1)  # remaining 25 %

        for idx, sent in enumerate(sentences, 1):
            entry = {
                "start": None,
                "end": None,
                "speaker": "interviewee",
                "text": sent,
            }
            if model_emb is not None:
                entry["score"] = score_answer(sent, keywords, model_emb)
            transcript.append(entry)
            tick(int(75 + seg_weight * idx), f"Scoring sentence {idx}/{len(sentences)}")

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
import asyncio
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import whisper

from .logger import log_metrics, logger
import logging

_model: Any = None
_executor = ThreadPoolExecutor(max_workers=1)

CHUNK_DURATION = 30
OVERLAP = 2


def load_model() -> Any:
    global _model
    model_name = os.environ.get("WHISPER_MODEL", "large-v3")
    device = os.environ.get("WHISPER_DEVICE", "auto")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "cpu"
        else:
            device = "cpu"
    log_metrics(logging.INFO, "model_load", f"Загрузка модели {model_name} (device={device}, compute={compute_type})...")
    t0 = time.perf_counter()
    _model = whisper.load_model(model_name, device=device)
    elapsed = time.perf_counter() - t0
    log_metrics(
        logging.INFO, "model_load",
        f"Модель {model_name} загружена",
        model=model_name, device=device, compute_type=compute_type,
        duration_sec=round(elapsed, 2),
    )
    return _model


def get_model() -> Any:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


async def load_model_async() -> Any:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, load_model)


def transcribe_chunk(chunk_path: str, offset: float) -> dict[str, Any]:
    model = get_model()
    chunk_idx = getattr(transcribe_chunk, "_idx", 0)
    total = getattr(transcribe_chunk, "_total", 1)
    log_metrics(
        logging.INFO, "transcribe_chunk",
        f"Транскрипция чанка {chunk_idx}/{total}...",
        chunk_idx=chunk_idx, total_chunks=total,
    )
    t0 = time.perf_counter()
    result = model.transcribe(
        chunk_path,
        language="ru",
        task="transcribe",
        fp16=False,
    )
    elapsed = time.perf_counter() - t0
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"] + offset,
            "end": seg["end"] + offset,
            "text": seg["text"].strip(),
        })

    word_count = sum(len(s["text"].split()) for s in segments)
    chunk_duration = CHUNK_DURATION if offset > 0 else 30
    speed = round(chunk_duration / elapsed, 1) if elapsed > 0 else 0

    log_metrics(
        logging.INFO, "transcribe_chunk",
        f"Чанк {chunk_idx}/{total} готов",
        chunk_idx=chunk_idx, total_chunks=total,
        duration_sec=round(elapsed, 2),
        segments=len(segments), words=word_count,
        speed=f"{speed}x",
    )
    return {"text": result["text"], "segments": segments}


async def split_audio(file_path: str) -> list[tuple[str, float]]:
    t0 = time.perf_counter()
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *probe_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    duration = float(stdout.decode().strip())

    log_metrics(
        logging.INFO, "split_audio",
        f"Длительность аудио: {duration:.1f}s",
        audio_duration=round(duration, 1),
    )

    if duration <= 30:
        elapsed = time.perf_counter() - t0
        log_metrics(
            logging.INFO, "split_audio",
            "Аудио короче 30s — чанкинг не требуется",
            duration_sec=round(elapsed, 2), chunks=1,
        )
        return [(file_path, 0.0)]

    chunks = []
    tmpdir = tempfile.mkdtemp(prefix="whisper_chunks_")
    offset = 0.0
    idx = 0

    while offset < duration:
        chunk_path = os.path.join(tmpdir, f"chunk_{idx:04d}.wav")
        cmd = [
            "ffmpeg", "-y", "-i", file_path,
            "-ss", str(offset),
            "-t", str(CHUNK_DURATION + OVERLAP),
            "-ar", "16000", "-ac", "1",
            "-f", "wav",
            chunk_path,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()

        if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
            chunks.append((chunk_path, offset))

        offset += CHUNK_DURATION
        idx += 1

    elapsed = time.perf_counter() - t0
    log_metrics(
        logging.INFO, "split_audio",
        f"Создано {len(chunks)} чанков",
        duration_sec=round(elapsed, 2), chunks=len(chunks),
    )
    return chunks


async def transcribe_audio(file_path: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    log_metrics(logging.INFO, "transcribe_audio", "Начало транскрипции")

    chunks = await split_audio(file_path)
    total_chunks = len(chunks)

    for i, (cp, off) in enumerate(chunks):
        transcribe_chunk._idx = i + 1
        transcribe_chunk._total = total_chunks

    tasks = [transcribe_chunk_async(cp, off) for cp, off in chunks]
    results = await asyncio.gather(*tasks)

    all_segments: list[dict[str, Any]] = []
    for r in results:
        all_segments.extend(r["segments"])

    all_segments.sort(key=lambda s: s["start"])

    deduped: list[dict[str, Any]] = []
    for seg in all_segments:
        if not deduped:
            deduped.append(seg)
            continue
        last = deduped[-1]
        if seg["start"] >= last["end"]:
            deduped.append(seg)
        elif len(seg["text"]) > len(last["text"]):
            deduped[-1] = seg

    full_text = " ".join(s["text"] for s in deduped)

    for cp, _ in chunks:
        if cp != file_path:
            try:
                os.remove(cp)
                os.rmdir(os.path.dirname(cp))
            except OSError:
                pass

    total_words = sum(len(s["text"].split()) for s in deduped)
    elapsed = time.perf_counter() - t0

    log_metrics(
        logging.INFO, "transcribe_audio",
        "Транскрипция завершена",
        duration_sec=round(elapsed, 2),
        segments=len(deduped), words=total_words,
        chunks=total_chunks,
    )

    return {"text": full_text, "segments": deduped}


async def transcribe_chunk_async(chunk_path: str, offset: float) -> dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, transcribe_chunk, chunk_path, offset)

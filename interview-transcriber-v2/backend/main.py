import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

import json

from .logger import log_metrics, logger
from .whisper_utils import load_model_async, transcribe_audio, transcribe_audio_stream
import logging

MAX_FILE_SIZE = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    log_metrics(logging.INFO, "startup", "Запуск приложения")
    await load_model_async()
    log_metrics(logging.INFO, "startup", "Приложение готово к работе")
    yield
    log_metrics(logging.INFO, "shutdown", "Остановка приложения")


app = FastAPI(title="Whisper Transcriber", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    t0 = time.perf_counter()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = tmp.name
        file_size = 0
        CHUNK_SIZE = 1024 * 1024
        while chunk := await file.read(CHUNK_SIZE):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                os.remove(tmp_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max: {MAX_FILE_SIZE} bytes",
                )
            tmp.write(chunk)

    file_size_mb = round(file_size / (1024 * 1024), 2)
    log_metrics(
        logging.INFO, "receive_file",
        f"Получен файл: {file.filename}",
        file_name=file.filename, file_ext=ext, file_size=f"{file_size_mb} MB",
    )

    try:
        result = await transcribe_audio(tmp_path)
        elapsed = time.perf_counter() - t0
        log_metrics(
            logging.INFO, "transcribe_complete",
            f"Обработка {file.filename} завершена",
            duration_sec=round(elapsed, 2),
            segments=len(result.get("segments", [])),
            words=len(result.get("text", "").split()),
        )
        return result
    except Exception as e:
        log_metrics(logging.ERROR, "transcribe_error", f"Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


async def _save_upload(file: UploadFile) -> tuple[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = tmp.name
        file_size = 0
        CHUNK_SIZE = 1024 * 1024
        while chunk := await file.read(CHUNK_SIZE):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                os.remove(tmp_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max: {MAX_FILE_SIZE} bytes",
                )
            tmp.write(chunk)
    return tmp_path, ext


@app.post("/transcribe/stream")
async def transcribe_stream(file: UploadFile = File(...)) -> StreamingResponse:
    tmp_path, ext = await _save_upload(file)

    log_metrics(
        logging.INFO, "receive_file",
        f"Получен файл: {file.filename}",
        file_name=file.filename, file_ext=ext,
    )

    async def event_generator():
        try:
            async for chunk_data in transcribe_audio_stream(tmp_path):
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    html_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return f.read()
    return "<h1>Whisper Transcriber API</h1><p>POST /transcribe to transcribe audio.</p>"

import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .logger import log_metrics, logger
from .whisper_utils import load_model_async, transcribe_audio
import logging

MAX_FILE_SIZE = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}


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
    allow_origins=["*"],
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

    content = await file.read()
    file_size_mb = round(len(content) / (1024 * 1024), 2)
    log_metrics(
        logging.INFO, "receive_file",
        f"Получен файл: {file.filename}",
        file_name=file.filename, file_ext=ext, file_size=f"{file_size_mb} MB",
    )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content)} bytes. Max: {MAX_FILE_SIZE} bytes",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

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


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    html_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return f.read()
    return "<h1>Whisper Transcriber API</h1><p>POST /transcribe to transcribe audio.</p>"

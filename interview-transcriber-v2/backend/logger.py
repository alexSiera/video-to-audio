import json
import logging
import sys
from datetime import datetime, timezone


class StructuredHandler(logging.Handler):
    def emit(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "step": getattr(record, "step", "general"),
            "message": record.getMessage(),
        }
        for key in [
            "model", "duration_sec", "file_size", "chunks", "segments",
            "words", "audio_duration", "speed", "chunk_idx", "total_chunks",
            "file_name", "file_ext",
        ]:
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        print(json.dumps(log_entry, ensure_ascii=False), file=sys.stdout)


class HumanReadableHandler(logging.Handler):
    def emit(self, record):
        ts = datetime.now().strftime("%H:%M:%S")
        msg = record.getMessage()
        extras = []
        for key, label in [
            ("duration_sec", "время"), ("speed", "скорость"),
            ("segments", "сегментов"), ("words", "слов"),
            ("audio_duration", "аудио"), ("chunks", "чанков"),
        ]:
            val = getattr(record, key, None)
            if val is not None:
                extras.append(f"{label}: {val}")
        if extras:
            msg += f" ({', '.join(extras)})"
        print(f"[{ts}] [{record.levelname}] {msg}", file=sys.stdout)


logger = logging.getLogger("whisper")
logger.setLevel(logging.DEBUG)
logger.addHandler(StructuredHandler())
logger.addHandler(HumanReadableHandler())


def log_metrics(level: int, step: str, message: str, **metrics):
    record = logger.makeRecord("whisper", level, "", 0, message, (), None)
    record.step = step
    for k, v in metrics.items():
        setattr(record, k, v)
    logger.handle(record)

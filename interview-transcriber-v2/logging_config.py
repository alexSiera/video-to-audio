"""
Модуль логирования для пайплайна транскрибации аудио (OpenAI Whisper).

Предоставляет:
- Централизованную настройку логирования с выводом в stdout и файл
- Специализированные логгеры для каждого этапа пайплайна
- Декоратор @log_duration для автоматического замера времени
- Функцию log_error с трейсбеком
"""

import logging
import os
import sys
import time
import functools
import traceback
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from typing import Optional, Callable, Any


# =============================================================================
# Константы
# =============================================================================

LOG_FILE = "transcription.log"
MAX_BYTES = 5 * 1024 * 1024  # 5 МБ
BACKUP_COUNT = 3


# =============================================================================
# Форматы сообщений
# =============================================================================

# Стандартный формат: временная метка, имя логгера, уровень, сообщение
DEFAULT_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"

# Расширенный формат для DEBUG: добавляется файл и строка
DEBUG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s"


def _get_log_level() -> int:
    """Получает уровень логирования из переменной окружения LOG_LEVEL."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def setup_logging() -> None:
    """
    Централизованная настройка логирования.

    - Уровень задаётся через LOG_LEVEL (по умолчанию INFO)
    - Вывод в stdout и файл с ротацией (5 МБ, 3 бэкапа)
    - В DEBUG-режиме добавляется файл и номер строки
    """
    level = _get_log_level()
    is_debug = level == logging.DEBUG

    # Выбор формата в зависимости от уровня
    fmt = DEBUG_FORMAT if is_debug else DEFAULT_FORMAT

    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Очистка существующих обработчиков (при повторном вызове)
    root_logger.handlers.clear()

    formatter = logging.Formatter(fmt)

    # Обработчик для stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # Обработчик для файла с ротацией
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


# =============================================================================
# Специализированные логгеры для этапов пайплайна
# =============================================================================

# Логгер загрузки модели
model_logger = logging.getLogger("model_loader")

# Логгер обработки аудио
audio_logger = logging.getLogger("audio_processor")

# Логгер транскрибации
transcribe_logger = logging.getLogger("transcriber")

# Логгер обработки результатов
result_logger = logging.getLogger("result_handler")

# Логгер оценки качества (WER/CER)
eval_logger = logging.getLogger("evaluator")


# =============================================================================
# Декоратор / контекстный менеджер @log_duration
# =============================================================================

@contextmanager
def log_duration(
    operation: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
):
    """
    Контекстный менеджер для автоматического замера и логирования времени выполнения.

    Использование:
        with log_duration("Загрузка модели", model_logger):
            model = load_model("base")
    """
    log = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.log(level, "%s завершено за %.2f сек", operation, elapsed)


def log_duration_decorator(
    operation: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
) -> Callable:
    """
    Декоратор для автоматического замера и логирования времени выполнения функции.

    Использование:
        @log_duration("Обработка аудио", audio_logger)
        def process_audio(path):
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or f"Выполнение {func.__name__}"
        log = logger or logging.getLogger(__name__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                log.log(level, "%s завершено за %.2f сек", op_name, elapsed)

        return wrapper
    return decorator


# =============================================================================
# Вспомогательная функция для логирования ошибок
# =============================================================================

def log_error(
    exc: Exception,
    message: str = "Произошла ошибка",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Логирует исключение с полным трейсбеком через logger.exception.

    Использование:
        try:
            transcribe(audio)
        except Exception as e:
            log_error(e, "Ошибка транскрибации", transcribe_logger)
    """
    log = logger or logging.getLogger(__name__)
    log.exception("%s: %s", message, exc)


# =============================================================================
# Пример интеграции в основной код транскрибации
# =============================================================================

if __name__ == "__main__":
    # Пример использования в главном скрипте транскрибации

    setup_logging()

    # Пример: загрузка модели
    def load_model(model_name: str) -> dict:
        """Загрузка модели Whisper."""
        model_logger.info("Начало загрузки модели: %s", model_name)
        with log_duration("Загрузка модели", model_logger):
            # Имитация загрузки модели
            time.sleep(0.5)
        model_info = {
            "name": model_name,
            "device": "cpu",
            "parameters": "39M",
        }
        model_logger.info("Модель загружена: %s (устройство: %s)", model_info["name"], model_info["device"])
        return model_info

    # Пример: обработка аудио
    @log_duration_decorator("Обработка аудиофайла", audio_logger)
    def process_audio(audio_path: str) -> dict:
        """Анализ входного аудиофайла."""
        audio_logger.info("Обработка аудио: %s", audio_path)
        audio_info = {
            "path": audio_path,
            "duration": 120.5,
            "sample_rate": 16000,
            "channels": 1,
            "file_size_mb": 12.3,
        }
        audio_logger.info(
            "Аудио: длительность=%.1fс, частота=%dГц, каналы=%d, размер=%.1fМБ",
            audio_info["duration"],
            audio_info["sample_rate"],
            audio_info["channels"],
            audio_info["file_size_mb"],
        )
        return audio_info

    # Пример: транскрибация
    def transcribe(audio_info: dict, model_info: dict, language: str = "ru") -> dict:
        """Выполнение транскрибации."""
        transcribe_logger.info("Параметры транскрибации: язык=%s, task=transcribe", language)
        transcribe_logger.info(
            "Параметры: temperature=0.0, beam_size=5, word_timestamps=True"
        )

        with log_duration("Транскрибация аудио", transcribe_logger):
            time.sleep(1)  # Имитация обработки

        result = {
            "text": "Пример распознанного текста...",
            "language": "ru",
            "segments": 15,
            "avg_confidence": 0.92,
        }
        return result

    # Пример: обработка результата
    def handle_result(result: dict) -> None:
        """Логирование результатов транскрибации."""
        result_logger.info(
            "Результат: сегментов=%d, язык=%s, средняя уверенность=%.2f",
            result["segments"],
            result["language"],
            result["avg_confidence"],
        )
        # В DEBUG-режиме выводим распознанный текст
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            result_logger.debug("Распознанный текст: %s", result["text"])

    # Пример: оценка качества
    def evaluate_quality(result: dict, reference: str) -> None:
        """Вычисление WER/CER при наличии эталонного текста."""
        eval_logger.info("Начало оценки качества транскрибации")
        with log_duration("Вычисление WER/CER", eval_logger):
            # Имитация вычислений
            wer = 0.08
            cer = 0.03
        eval_logger.info("WER=%.2f%%, CER=%.2f%%", wer * 100, cer * 100)

    # Пример: обработка ошибок
    def run_pipeline() -> None:
        """Основной пайплайн транскрибации с обработкой ошибок."""
        try:
            model_info = load_model("base")
            audio_info = process_audio("interview.mp3")
            result = transcribe(audio_info, model_info)
            handle_result(result)
            evaluate_quality(result, "Эталонный текст...")
        except Exception as e:
            log_error(e, "Ошибка в пайплайне транскрибации", transcribe_logger)
            raise

    run_pipeline()

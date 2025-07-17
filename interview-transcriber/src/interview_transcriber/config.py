import os
import torch

WHISPER_MODEL = "bond005/whisper-large-v3-ru-podlodka"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", os.cpu_count()))
CHUNK_LENGTH_S = 30
OVERLAP_MS = 1000
SAMPLE_RATE = 16000
HEADROOM_DB = 0.1
MIN_SILENCE_MS = 800
SILENCE_THRESH_DB = -40
# Whisper Transcriber

FastAPI backend for audio transcription using OpenAI Whisper.

## Setup

```bash
# Install uv if not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Or with requirements.txt
uv pip install -r backend/requirements.txt
```

## Running

```bash
uvicorn backend.main:app --reload
```

The API will be available at http://localhost:8000

## Endpoints

- `GET /health` - Health check
- `POST /transcribe` - Transcribe audio file (multipart/form-data)
- `GET /` - Web interface

## Environment Variables

- `WHISPER_MODEL` - Whisper model size (default: `large-v3`)

## Usage

### cURL

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.mp3"
```

### Python

```python
import requests

url = "http://localhost:8000/transcribe"
files = {"file": open("audio.mp3", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Response Format

```json
{
  "text": "Full transcribed text...",
  "segments": [
    {"start": 0.0, "end": 5.0, "text": "First segment"},
    {"start": 5.0, "end": 10.0, "text": "Second segment"}
  ]
}
```

## Supported Formats

mp3, wav, m4a, ogg, flac, webm, mp4

## Limits

- Max file size: 500 MB
- Files are split into 30-second chunks for processing

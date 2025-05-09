from flask import Flask, request, jsonify
from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence
from pydub.effects import normalize
import os
import tempfile
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    AutoFeatureExtractor,
    pipeline,
    WhisperFeatureExtractor
)
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

app = Flask(__name__)

# Load model components
model_name = "antony66/whisper-large-v3-russian"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
except Exception:
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    device=device,
    chunk_length_s=30,
    batch_size=16,
    model_kwargs={"language": "ru", "max_new_tokens": 128}
)

class AudioChunkDataset(Dataset):
    def __init__(self, chunk_paths):
        self.chunk_paths = chunk_paths

    def __len__(self):
        return len(self.chunk_paths)

    def __getitem__(self, idx):
        return self.chunk_paths[idx]


def extract_audio(video_path):
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        if not audio:
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio.write_audiofile(temp_audio.name, codec='pcm_s16le', verbose=False, logger=None)
            return temp_audio.name
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return None


def optimize_audio(audio_path):
    """Улучшенная предварительная обработка аудио"""
    try:
        audio = AudioSegment.from_file(audio_path)

        # Конвертируем в моно и 16 кГц
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Фильтрация шума
        audio = audio.low_pass_filter(8000).high_pass_filter(200)

        # Нормализация громкости
        normalized = normalize(audio, headroom=0.1)

        # Удаление тишины
        nonsilent = silence.detect_nonsilent(normalized, min_silence_len=800, silence_thresh=-40)
        cleaned = normalized if not nonsilent else normalized[nonsilent[0][0]:nonsilent[-1][1]]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            cleaned.export(temp.name, format="wav")
            return temp.name
    except Exception as e:
        print(f"Audio optimization failed: {e}")
        return None


def split_audio(audio_path):
    """Разделение аудио на фрагменты по 30 секунд"""
    try:
        audio = AudioSegment.from_file(audio_path)
        chunk_length_ms = 30 * 1000
        overlap = 1000
        temp_dir = tempfile.mkdtemp()
        chunk_paths = []

        for i, start in enumerate(range(0, len(audio), chunk_length_ms - overlap)):
            end = start + chunk_length_ms
            chunk = audio[start:end]
            chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)

        return chunk_paths
    except Exception as e:
        print(f"Audio splitting failed: {e}")
        return []


def transcribe_chunks(chunk_paths):
    """Обработка аудиочанков в батчах"""
    results = [""] * len(chunk_paths)
    dataset = AudioChunkDataset(chunk_paths)
    loader = DataLoader(dataset, batch_size=pipe.model.config.batch_size or 16, shuffle=False)

    with tqdm(total=len(chunk_paths), desc="Transcribing", ncols=100) as pbar:
        for batch_paths in loader:
            outputs = pipe(batch_paths)
            for i, path in enumerate(batch_paths):
                idx = chunk_paths.index(path)
                results[idx] = outputs[i]["text"]
            pbar.update(len(batch_paths))

    return " ".join(results)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        base_name = os.path.splitext(video_file.filename)[0]
        result_filename = f"{base_name}_transcript.txt"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            video_file.save(temp_video.name)
            video_path = temp_video.name

        audio_path = extract_audio(video_path)
        if not audio_path:
            return jsonify({"error": "Audio extraction failed"}), 500

        optimized_audio_path = optimize_audio(audio_path)
        if not optimized_audio_path:
            return jsonify({"error": "Audio optimization failed"}), 500

        chunk_paths = split_audio(optimized_audio_path)
        if not chunk_paths:
            return jsonify({"error": "Audio splitting failed"}), 500

        full_transcription = transcribe_chunks(chunk_paths)

        return jsonify({
            "filename": result_filename,
            "transcription": full_transcription
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        for path in [video_path, audio_path, optimized_audio_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    print(f"Cleanup failed for {path}: {e}")

        # Clean up all chunk files
        if 'chunk_paths' in locals():
            for path in chunk_paths:
                if os.path.exists(path):
                    os.unlink(path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
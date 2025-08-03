from flask import Flask, request, jsonify
from moviepy import VideoFileClip
from pydub import AudioSegment, silence
from pydub.effects import normalize
import os
import tempfile
import torch, torchvision, sys
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    AutoFeatureExtractor,
    pipeline
)
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import tempfile, os
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from optimum.onnxruntime import ORTConfig

if os.name == "nt":                       
    CUSTOM_TEMP = r"C:\Users\fa1nt\Documents\GitHubLearn\video-to-audio\temp"
else:                                     
    CUSTOM_TEMP = os.path.join(tempfile.gettempdir(), "whisper_tmp")

os.makedirs(CUSTOM_TEMP, exist_ok=True)

app = Flask(__name__)

os.environ["TMP"] = os.environ["TEMP"] = tempfile.mkdtemp(prefix="whisper_tmp_")

# Load model components
#model_name = "dvislobokov/whisper-large-v3-turbo-russian"
#model_name = "bond005/whisper-large-v3-ru-podlodka"
#model_name = "antony66/whisper-large-v3-russian"
#model_name = "openai/whisper-large-v3"
model_name = "dvislobokov/faster-whisper-large-v3-turbo-russian"

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    sys.exit("CUDA not available. Exiting.")

BATCH_SIZE = 64 if device == "cuda" else 4

try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
except Exception:
    from transformers import WhisperFeatureExtractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Замена загрузки модели на версию с поддержкой ONNX
if "ONNX_ENABLED" in os.environ:
    app.logger.info("Using ONNX Runtime for model acceleration")
    # Создайте оптимизированную ONNX версию (один раз)
    if not os.path.exists("onnx_model"):
        app.logger.info("Converting model to ONNX format (this may take a few minutes)...")
        ort_config = ORTConfig.from_pretrained(model_name)
        ORTModelForSpeechSeq2Seq.from_pretrained(model_name).save_pretrained("onnx_model", ort_config=ort_config)
        app.logger.info("ONNX model conversion completed")
    
    # Загрузите оптимизированную модель
    model = ORTModelForSpeechSeq2Seq.from_pretrained("onnx_model", device=device)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        batch_size=BATCH_SIZE,
        # ONNX-специфичные параметры
        provider="CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    )
else:
    app.logger.info("Using standard PyTorch model")
    # Стандартная загрузка
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
    model.resize_token_embeddings(len(tokenizer))
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=device,
        chunk_length_s=30,
        batch_size=BATCH_SIZE,
        model_kwargs={"language": "ru"}
    )

# Thread-safe pipeline lock
pipeline_lock = Lock()

def process_chunk(chunk):
    """Process individual audio chunk with thread safety"""
    try:
        # 1. Export to a real file that we close ourselves
        with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".wav",
                delete=False,          # we delete it ourselves
                dir=CUSTOM_TEMP) as f:
            chunk.export(f.name, format="wav")
            tmp_name = f.name        # keep the name before we leave the block

        # 2. File is now closed; Windows has no open handle → no PermissionError
        with pipeline_lock, torch.no_grad():
            text = pipe(tmp_name)["text"]

        # 3. Clean up
        os.unlink(tmp_name)
        return text

    except Exception as e:
        print(f"Chunk processing failed: {e}")
        return ""

def extract_audio(video_path):
    """Extract audio with progress indication [[1]]"""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        if not audio:
            return None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            with tqdm(total=100, desc="Extracting audio", ncols=100) as pbar:
                audio.write_audiofile(temp_audio.name, codec='pcm_s16le', logger=None)
                pbar.update(100)
            return temp_audio.name
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return None

# def optimize_audio(audio_path):
#     """Normalize audio with progress [[3]]"""
#     try:
#         audio = AudioSegment.from_file(audio_path)
#         with tqdm(total=100, desc="Optimizing audio", ncols=100) as pbar:
#             normalized_audio = normalize(audio)
#             pbar.update(100)
        
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_optimized:
#             normalized_audio.export(temp_optimized.name, format="wav")
#             return temp_optimized.name
#     except Exception as e:
#         print(f"Audio optimization failed: {e}")
#         return None

def optimize_audio(audio_path):
    """Enhanced audio preprocessing"""
    try:
        audio = AudioSegment.from_file(audio_path)
        
        # Apply processing chain
        with tqdm(total=100, desc="Optimizing audio", ncols=100) as pbar:
            # Convert to mono and 16kHz first
            audio = audio.set_frame_rate(16000).set_channels(1)
            pbar.update(20)
            
            # Noise reduction
            audio = audio.low_pass_filter(8000).high_pass_filter(200)
            pbar.update(20)
            
            # Normalization with dynamic compression
            normalized = normalize(audio, headroom=0.1)
            pbar.update(20)
            
            # Silence removal
            nonsilent = silence.detect_nonsilent(
                normalized, 
                # min_silence_len=500,
                min_silence_len=800,
                silence_thresh=-40
            )
            cleaned = normalized.split_to_mono()[0]
            if nonsilent:
                cleaned = normalized[nonsilent[0][0]:nonsilent[-1][1]]
            pbar.update(40)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            cleaned.export(temp.name, format="wav")
            return temp.name
    except Exception as e:
        print(f"Audio optimization failed: {e}")
        return None

def split_audio(audio_path):
    """Split audio into 30s chunks with progress [[5]]"""
    try:
        audio = AudioSegment.from_file(audio_path)
        chunk_length_ms = 30 * 1000
        chunks = []
        total_chunks = (len(audio) // chunk_length_ms) + 1
        
        with tqdm(total=total_chunks, desc="Preparing chunks", ncols=100) as pbar:
            overlap = 1000
            for i in range(0, len(audio), chunk_length_ms - overlap):
                chunk = audio[i:i + chunk_length_ms]
                chunks.append(chunk)
                pbar.update(1)
        return chunks
    except Exception as e:
        print(f"Audio splitting failed: {e}")
        return None

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

        chunks = split_audio(optimized_audio_path)
        if not chunks:
            return jsonify({"error": "Audio splitting failed"}), 500

        # Modified processing to maintain order
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit all chunks with their original indices
            future_to_index = {
                executor.submit(process_chunk, chunk): i
                for i, chunk in enumerate(chunks)
            }
            
            # Create list to hold results in original order
            results = [None] * len(chunks)
            
            with tqdm(total=len(chunks), desc="Transcribing", ncols=100) as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    pbar.update(1)

        # Combine results in original order
        ordered_transcription = " ".join([text for text in results if text])

        return jsonify({"transcription": ordered_transcription})

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        for path in [video_path, audio_path] + ([optimized_audio_path] if 'optimized_audio_path' in locals() else []):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    print(f"Cleanup failed for {path}: {e}")


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """
    Принимает аудио-файл, возвращает транскрипцию.
    Ключ поля формы: audio
    Поддерживаемые форматы: wav, mp3, ogg, m4a, flac, …
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "Empty file name"}), 400

    try:
        # 1. Сохраняем загруженный аудио-файл
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=CUSTOM_TEMP,
            suffix=os.path.splitext(audio_file.filename)[1] or '.wav'
        ) as tmp_audio:
            audio_file.save(tmp_audio.name)
            original_path = tmp_audio.name

        # 2. Оптимизируем аудио
        optimized_audio_path = optimize_audio(original_path)
        if not optimized_audio_path:
            return jsonify({"error": "Audio optimization failed"}), 500

        # 3. Разбиваем на чанки
        chunks = split_audio(optimized_audio_path)
        if not chunks:
            return jsonify({"error": "Audio splitting failed"}), 500

        # 4. Транскрибируем (порядок сохраняем)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_index = {
                executor.submit(process_chunk, chunk): i
                for i, chunk in enumerate(chunks)
            }
            results = [None] * len(chunks)

            with tqdm(total=len(chunks), desc="Transcribing", ncols=100) as pbar:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    results[idx] = future.result()
                    pbar.update(1)

        transcription = " ".join(t for t in results if t)

        return jsonify({"transcription": transcription})

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        # Удаляем все временные файлы
        for path in (original_path, optimized_audio_path) if 'optimized_audio_path' in locals() else (original_path,):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    print(f"Cleanup failed for {path}: {e}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# Entry-point wrapper
def main():
    app.run(host="0.0.0.0", port=5000)
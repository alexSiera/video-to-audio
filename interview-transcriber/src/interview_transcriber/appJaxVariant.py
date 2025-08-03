from flask import Flask, request, jsonify
from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence
from pydub.effects import normalize
import os
import tempfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np
from whisper_jax import FlaxWhisperPipline

# Configure temporary directory
if os.name == "nt":
    CUSTOM_TEMP = r"C:\Users\fa1nt\Documents\GitHubLearn\video-to-audio\temp"
else:
    CUSTOM_TEMP = os.path.join(tempfile.gettempdir(), "whisper_tmp")

os.makedirs(CUSTOM_TEMP, exist_ok=True)
os.environ["TMP"] = os.environ["TEMP"] = CUSTOM_TEMP

app = Flask(__name__)

# Initialize Whisper-JAX
model_name = "openai/whisper-large-v3"  # Works better than custom models for JAX
batch_size = 16  # Adjust based on your GPU memory
pipeline = FlaxWhisperPipline(model_name, dtype="float16", batch_size=batch_size)
pipeline_lock = Lock()

def extract_audio(video_path):
    """Extract audio from video with progress"""
    try:
        with VideoFileClip(video_path) as video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                video.audio.write_audiofile(temp_audio.name, codec='pcm_s16le', logger=None)
                return temp_audio.name
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return None

def optimize_audio(audio_path):
    """Enhanced audio preprocessing"""
    try:
        audio = AudioSegment.from_file(audio_path)
        
        # Processing chain
        audio = (
            audio.set_frame_rate(16000)
            .set_channels(1)
            .low_pass_filter(8000)
            .high_pass_filter(200)
        )
        
        # Normalization and silence removal
        normalized = normalize(audio, headroom=0.1)
        nonsilent = silence.detect_nonsilent(
            normalized, 
            min_silence_len=800,
            silence_thresh=-40
        )
        
        if nonsilent:
            cleaned = normalized[nonsilent[0][0]:nonsilent[-1][1]]
        else:
            cleaned = normalized

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            cleaned.export(temp.name, format="wav")
            return temp.name
    except Exception as e:
        print(f"Audio optimization failed: {e}")
        return None

def split_audio(audio_path, chunk_length=30):
    """Split audio into chunks with 1s overlap"""
    try:
        audio = AudioSegment.from_file(audio_path)
        chunk_ms = chunk_length * 1000
        return [
            audio[i:i + chunk_ms]
            for i in range(0, len(audio), chunk_ms - 1000)  # 1s overlap
        ]
    except Exception as e:
        print(f"Audio splitting failed: {e}")
        return None

def process_batch(chunks):
    """Process multiple chunks in a single batch"""
    temp_files = []
    try:
        # Export chunks to temporary files
        for chunk in chunks:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".wav",
                delete=False,
                dir=CUSTOM_TEMP
            ) as f:
                chunk.export(f.name, format="wav")
                temp_files.append(f.name)

        # Batch inference
        with pipeline_lock:
            outputs = pipeline(temp_files, task="transcribe", language="ru")
            return [output["text"] for output in outputs]
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return [""] * len(chunks)
    finally:
        for file in temp_files:
            try:
                os.unlink(file)
            except:
                pass

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Temporary file handling
    temp_files = []
    try:
        # Save video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            video_file.save(f.name)
            video_path = f.name
            temp_files.append(video_path)

        # Process audio
        audio_path = extract_audio(video_path)
        if not audio_path:
            return jsonify({"error": "Audio extraction failed"}), 500
        temp_files.append(audio_path)

        optimized_path = optimize_audio(audio_path)
        if not optimized_path:
            return jsonify({"error": "Audio optimization failed"}), 500
        temp_files.append(optimized_path)

        chunks = split_audio(optimized_path)
        if not chunks:
            return jsonify({"error": "Audio splitting failed"}), 500

        # Process in batches
        transcriptions = []
        for i in tqdm(range(0, len(chunks), batch_size), desc="Transcribing"):
            batch = chunks[i:i + batch_size]
            transcriptions.extend(process_batch(batch))

        return jsonify({
            "transcription": " ".join(transcriptions),
            "chunks": len(chunks),
            "batch_size": batch_size
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        for path in temp_files:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception as e:
                print(f"Cleanup failed for {path}: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
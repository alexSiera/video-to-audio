from flask import Flask, request, jsonify, send_file
import os
import torch
import librosa
import librosa.effects
import numpy as np
from moviepy.editor import VideoFileClip
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import scipy.signal as signal
import soundfile as sf
import time
import gc
import concurrent.futures
from tqdm import tqdm
import threading
import json
import warnings
import torchaudio
import traceback
import shutil
import re

app = Flask(__name__)
processing_status = {}
status_lock = threading.Lock()

def update_status(job_id, status, progress=None):
    with status_lock:
        if job_id not in processing_status:
            processing_status[job_id] = {"status": "initializing", "progress": 0}
        processing_status[job_id]["status"] = status
        if progress is not None:
            processing_status[job_id]["progress"] = progress

class RussianVideoTranscriber:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Initializing model on {self.device}")
        self.model_name = "dvislobokov/whisper-large-v3-turbo-russian"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.language = "ru"
            self.model.generation_config.task = "transcribe"

    def extract_audio(self, video_path, output_path="temp_audio.wav", job_id=None):
        try:
            update_status(job_id, "extracting_audio", 5)
            video = VideoFileClip(video_path)
            audio = video.audio
            if audio is None:
                raise ValueError("No audio track found")
            
            temp_audio_path = output_path + ".temp.wav"
            audio.write_audiofile(
                temp_audio_path,
                codec='pcm_s16le',
                ffmpeg_params=["-ac", "1", "-ar", "16000", "-af", "afftdn, loudnorm"],
                logger=None
            )
            audio.close()
            video.close()
            
            self.enhance_audio(temp_audio_path, output_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return output_path
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {str(e)}")

    def enhance_audio(self, input_path, output_path):
        try:
            y, sr = librosa.load(input_path, sr=16000)
            y = np.nan_to_num(y)
            y, _ = librosa.effects.trim(y, top_db=30)
            b, a = signal.butter(5, 80/(sr/2), 'highpass')
            y = signal.filtfilt(b, a, y)
            y = librosa.effects.preemphasis(y)
            sf.write(output_path, y, sr)
            return True
        except Exception as e:
            shutil.copyfile(input_path, output_path)
            return False

    def transcribe_audio(self, audio_path, job_id=None):
        try:
            update_status(job_id, "transcribing", 20)
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.mean(dim=0).unsqueeze(0).to(self.device)
            
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
                generated_ids = self.model.generate(
                    inputs.input_features.to(self.device),
                    max_new_tokens=448,
                    task="transcribe",
                    language="ru"
                )
            
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            update_status(job_id, "completed", 100)
            return transcription
        except Exception as e:
            update_status(job_id, "failed", 0)
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def process_video(self, video_path, job_id=None):
        try:
            audio_path = f"temp_audio_{int(time.time())}.wav"
            self.extract_audio(video_path, audio_path, job_id)
            return self.transcribe_audio(audio_path, job_id)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

transcriber = RussianVideoTranscriber()

@app.route('/transcribe', methods=['POST'])
def transcribe_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    job_id = f"job_{int(time.time())}"
    video_path = f"temp_{job_id}.mp4"
    video_file.save(video_path)

    try:
        update_status(job_id, "starting", 0)
        transcription = transcriber.process_video(video_path, job_id)
        return jsonify({
            "transcription": transcription,
            "job_id": job_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    with status_lock:
        return jsonify(processing_status.get(job_id, {"status": "not_found"}))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
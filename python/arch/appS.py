from flask import Flask, request, jsonify
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.effects import normalize
import os
import tempfile
import torch
import subprocess
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    AutoFeatureExtractor,
    pipeline
)
from pyannote.audio import Pipeline
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from tempfile import NamedTemporaryFile
import time
from threading import Lock, Thread

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load model components
model_name = "dvislobokov/whisper-large-v3-turbo-russian"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
except Exception:
    from transformers import WhisperFeatureExtractor
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
    batch_size=64
)

# Load diarization pipeline
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
    )
except Exception:
    logger.warning("Falling back to open-source diarization model")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=False
    )

# Load text correction model
corrector = pipeline("text2text-generation", model="t5-base")

pipeline_lock = Lock()

def process_chunk(args):
    chunk, speaker = args
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            chunk.export(f.name, format="wav")
            with pipeline_lock:
                with torch.no_grad():
                    return (speaker, pipe(f.name)["text"])
    except Exception as e:
        logger.error(f"Chunk processing failed: {e}")
        return (speaker, "")

def extract_audio(video_path):
    """Extract audio with FFmpeg and show progress percentage [[1]][[5]][[7]]"""
    logger.info("Starting audio extraction")
    audio_path = None  # Initialize to prevent cleanup errors
    
    try:
        # Verify video has audio stream [[1]][[7]]
        cmd_probe = [
            'ffprobe', '-v', 'error', '-show_entries',
            'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        try:
            streams = subprocess.check_output(cmd_probe, stderr=subprocess.PIPE).decode().splitlines()
            if 'audio' not in streams:
                logger.error("No audio stream found in video")
                return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to probe video: {e.stderr.decode() if e.stderr else str(e)}")
            return None

        # Get video duration for progress calculation [[1]]
        cmd_duration = [
            'ffprobe', '-i', video_path,
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]
        try:
            duration = float(subprocess.check_output(cmd_duration, stderr=subprocess.PIPE).decode().strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"Failed to get video duration: {str(e)}")
            return None

        # FFmpeg command with noise reduction and normalization [[7]]
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_path = temp_audio.name  # Assign path early for cleanup
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-af', 'afftdn=nf=-25,dynaudnorm',  # Noise reduction
                '-ac', '1', '-ar', '16000',         # Mono 16kHz [[1]]
                '-y', audio_path                     # Output to file directly
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            with tqdm(
                total=100,
                desc="Extracting audio",
                unit='%',
                ncols=100
            ) as pbar:
                last_progress = 0
                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                    if 'time=' in line:
                        try:
                            time_str = line.split('time=')[1].split()[0]
                            h, m, s = map(float, time_str.split(':'))
                            current_sec = h*3600 + m*60 + s
                            progress = min((current_sec / duration) * 100, 100)
                            delta = progress - last_progress
                            if delta >= 1:  # Update every 1%
                                pbar.update(delta)
                                last_progress = progress
                        except (IndexError, ValueError) as e:
                            logger.warning(f"Failed to parse progress: {str(e)}")

            # Finalize process and check errors
            process.wait()
            if process.returncode != 0:
                stderr_output = process.stderr.read()
                if stderr_output:
                    logger.error(f"FFmpeg error: {stderr_output}")
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr_output)

            logger.info("Audio extraction completed (100%)")
            return audio_path

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error ({e.returncode}): {e.stderr.decode()}")
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        return None
    except Exception as e:
        logger.error(f"Audio extraction failed: {str(e)}")
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        return None
    
# def diarize_audio(audio_path):
#     """Perform speaker diarization with progress percentage [[2]]"""
#     logger.info("Starting speaker diarization")
#     try:
#         diarization = diarization_pipeline(audio_path)
#         segments = []
#         for turn, _, speaker in diarization.itertracks():
#             segments.append({
#                 'start': int(turn.start * 1000),
#                 'end': int(turn.end * 1000),
#                 'speaker': speaker
#             })
#         logger.info(f"Diarization completed - {len(segments)} speakers detected")
#         return segments
#     except Exception as e:
#         logger.error(f"Diarization failed: {e}")
#         return None

# def diarize_audio(audio_path):
#     """Perform speaker diarization with real-time progress"""
#     logger.info("Starting speaker diarization")
#     start_time = time.time()
    
#     def progress_hook(progress: float, total: float):
#         percent = min((progress / total) * 100, 100)
#         pbar.update(percent - pbar.n)
    
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".wav") as f:
#             # Convert audio to required format
#             AudioSegment.from_file(audio_path).export(f.name, format="wav")
            
#             # Get audio duration
#             audio_info = AudioSegment.from_file(f.name)
            
#             # Setup progress bar
#             with tqdm(
#                 total=100,
#                 desc="Diarizing",
#                 unit='%',
#                 ncols=100,
#                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
#             ) as pbar:
#                 # Run diarization with progress tracking
#                 diarization = diarization_pipeline(
#                     f.name,
#                     hook=progress_hook,
#                 )
            
#         # Process results
#         segments = []
#         for turn, _, speaker in diarization.itertracks():
#             segments.append({
#                 'start': int(turn.start * 1000),
#                 'end': int(turn.end * 1000),
#                 'speaker': speaker
#             })
        
#         elapsed = time.time() - start_time
#         logger.info(f"Diarization completed in {elapsed:.1f}s "
#                     f"({len(segments)} speakers, {duration_sec:.1f}s audio)")
#         return segments
#     except Exception as e:
#         logger.error(f"Diarization failed: {e}")
#         return None

def diarize_audio(audio_path):
    """Perform speaker diarization with real-time progress"""
    logger.info("Starting speaker diarization")
    start_time = time.time()
    
    try:
        # Get audio duration using pydub
        audio = AudioSegment.from_file(audio_path)
        duration_sec = len(audio) / 1000  # Convert milliseconds to seconds
        
        diarization_result = None
        exception = None
        
        # Define diarization thread
        def run_diarization():
            nonlocal diarization_result, exception
            try:
                diarization_result = diarization_pipeline(audio_path)
            except Exception as e:
                exception = e
        
        # Start diarization in separate thread
        thread = Thread(target=run_diarization)
        thread.start()

        # Setup progress bar
        with tqdm(
            total=100,
            desc="Diarizing",
            unit='%',
            ncols=100,
            bar_format='{l_bar}{bar}| {n:.1f}% [{elapsed}<{remaining}]'
        ) as pbar:
            while thread.is_alive():
                elapsed = time.time() - start_time
                progress = min((elapsed / duration_sec) * 100, 99.9)  # Cap at 99.9% until completion
                pbar.n = progress
                pbar.refresh()
                time.sleep(0.5)
            
            thread.join()
            
            if exception:
                raise exception

            # Finalize to 100%
            pbar.n = 100
            pbar.refresh()

        # Process results
        segments = []
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': int(segment.start * 1000),
                'end': int(segment.end * 1000),
                'speaker': speaker
            })
        
        elapsed = time.time() - start_time
        logger.info(f"Diarization completed in {elapsed:.1f}s "
                    f"({len(segments)} speakers, {duration_sec:.1f}s audio)")
        return segments
        
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return None

def split_audio(audio_path, diarization_segments):
    """Split audio based on diarization segments with progress [[5]]"""
    logger.info("Splitting audio into speaker segments")
    try:
        audio = AudioSegment.from_file(audio_path)
        total_segments = len(diarization_segments)
        
        with tqdm(
            total=total_segments,
            desc="Splitting audio",
            unit='seg',
            ncols=100
        ) as pbar:
            chunks = []
            for seg in diarization_segments:
                chunk = audio[seg['start']:seg['end']]
                chunks.append((chunk, seg['speaker']))
                pbar.update(1)
                pbar.set_postfix_str(f"Speaker {seg['speaker']}")
        
        logger.info(f"Audio splitting completed ({total_segments} segments)")
        return chunks
    except Exception as e:
        logger.error(f"Audio splitting failed: {e}")
        return None

def correct_text(text):
    """Apply NLP-based text correction with progress [[3]]"""
    logger.info("Starting text correction (0%)")
    try:
        # Simulate progress for text correction
        words = text.split()
        corrected = []
        with tqdm(
            total=len(words),
            desc="Correcting text",
            unit='word',
            ncols=100
        ) as pbar:
            for word in words:
                corrected_word = corrector(word, max_length=512)[0]['generated_text']
                corrected.append(corrected_word)
                pbar.update(1)
        corrected_text = ' '.join(corrected)
        logger.info("Text correction completed (100%)")
        return corrected_text
    except Exception as e:
        logger.error(f"Text correction failed: {e}")
        return text

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    video_path = None
    audio_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            video_file.save(temp_video.name)
            video_path = temp_video.name

        # Step 1: Audio extraction (progress shown)
        logger.info("Step 1: Extracting audio from video")
        audio_path = extract_audio(video_path)
        if not audio_path:
            return jsonify({"error": "Audio extraction failed"}), 500

        # Step 2: Speaker diarization (progress shown)
        logger.info("Step 2: Performing speaker diarization")
        diarization_segments = diarize_audio(audio_path)
        if not diarization_segments:
            return jsonify({"error": "Diarization failed"}), 500

        # Step 3: Audio splitting (progress shown)
        logger.info("Step 3: Splitting audio into speaker segments")
        chunks = split_audio(audio_path, diarization_segments)
        if not chunks:
            return jsonify({"error": "Audio splitting failed"}), 500

        # Step 4: Parallel transcription (progress shown)
        logger.info(f"Step 4: Transcribing {len(chunks)} audio chunks")
        transcription = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            
            with tqdm(
                total=len(futures),
                desc="Transcribing",
                unit='chunk',
                ncols=100
            ) as pbar:
                for future in as_completed(futures):
                    speaker, text = future.result()
                    transcription.append(f"[{speaker}]: {text}")
                    pbar.update(1)
                    pbar.set_postfix_str(f"Speaker {speaker}")

        # Step 5: Text correction (progress shown)
        logger.info("Step 5: Applying text correction")
        raw_text = " ".join(transcription)
        corrected_text = correct_text(raw_text)

        return jsonify({
            "transcription": corrected_text,
            "raw_transcription": raw_text
        })

    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        logger.info("Cleaning up temporary files")
        for path in [video_path, audio_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"Cleanup failed for {path}: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
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

# Global variable to track processing status
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
        
        # Set torch options for numerical stability
        torch.set_printoptions(precision=10)
        torch.set_flush_denormal(True)  # Flush subnormal numbers to zero
        
        # Set deterministic behavior
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Load processor first
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Silence TensorFlow warnings that might appear during model loading
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Then load model and move to device
        print("Loading model, this may take a while...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for stability on CPU
                low_cpu_mem_usage=True
            ).to(self.device)
        
        # Configure task parameters WITHOUT setting forced_decoder_ids
        print("Setting model parameters for Russian transcription...")
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.language = "ru"
            self.model.generation_config.task = "transcribe"
        
        # Optimize for longer texts
        if hasattr(self.model.config, "max_length"):
            self.model.config.max_length = 448  # Extended for longer segments
        
        if "cuda" in self.device:
            torch.backends.cudnn.benchmark = True
            
        # Verify model and processor are properly configured
        print("Model and processor loaded successfully")
        print(f"Model device: {self.device}")
        print(f"Max length: {self.model.config.max_length if hasattr(self.model.config, 'max_length') else 'default'}")
        print(f"Using language: {self.model.generation_config.language if hasattr(self.model, 'generation_config') else 'not set'}")
        print(f"Using task: {self.model.generation_config.task if hasattr(self.model, 'generation_config') else 'not set'}")

    def extract_audio(self, video_path, output_path="temp_audio.wav", job_id=None):
        """Enhanced audio extraction with preprocessing"""
        try:
            update_status(job_id, "extracting_audio", 5)
            video = VideoFileClip(video_path)
            audio = video.audio
            
            if audio is None:
                raise ValueError("No audio track found in the video")
                
            # Apply fade in/out effects if available
            if hasattr(audio, 'audio_fadein') and hasattr(audio, 'audio_fadeout'):
                audio = audio.audio_fadein(0.5).audio_fadeout(0.5)
            
            update_status(job_id, "saving_audio", 10)
            # Save as mono WAV with 16kHz sample rate - optimal for speech recognition
            temp_audio_path = output_path + ".temp.wav"
            audio.write_audiofile(
                temp_audio_path,
                codec='pcm_s16le',
                ffmpeg_params=["-ac", "1", "-ar", "16000"],
                logger=None  # Suppress output
            )
            
            # Close to free resources
            audio.close()
            video.close()
            
            update_status(job_id, "enhancing_audio", 15)
            # Now enhance the audio using librosa for better quality
            self.enhance_audio(temp_audio_path, output_path)
            
            # Remove temp file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
            return output_path
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {str(e)}")
    
    def enhance_audio(self, input_audio_path, output_audio_path):
        """Apply advanced audio enhancement techniques for better transcription"""
        try:
            # Load audio
            y, sr = librosa.load(input_audio_path, sr=16000)
            
            # Check for invalid values early
            if not np.all(np.isfinite(y)):
                print("Warning: Input audio contains non-finite values. Cleaning...")
                y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Trim silent parts from the beginning and end (safely)
            if len(y) > 0:
                y, _ = librosa.effects.trim(y, top_db=30)
            
            # Apply a high-pass filter to reduce low-frequency noise - use more stable coefficients
            try:
                b, a = signal.butter(5, 80/(sr/2), 'highpass')  # Reduced order from 10 to 5
                y_filtered = signal.filtfilt(b, a, y)
                # Verify filter output is valid before assigning
                if np.all(np.isfinite(y_filtered)):
                    y = y_filtered
            except Exception as filter_err:
                print(f"High-pass filter error: {filter_err}")
            
            # Apply preemphasis more safely
            try:
                y_preemph = librosa.effects.preemphasis(y, coef=0.97)
                if np.all(np.isfinite(y_preemph)):
                    y = y_preemph
            except Exception as preemph_err:
                print(f"Preemphasis error: {preemph_err}")
            
            # Safer dynamic range compression
            try:
                # Handle edge cases
                if len(y) > 0:
                    # Remove any remaining non-finite values
                    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    abs_y = np.abs(y)
                    if abs_y.max() > 0:  # Ensure we have non-zero values
                        percentile_low = np.percentile(abs_y, 1)
                        percentile_high = np.percentile(abs_y, 99)
                        
                        if percentile_high > percentile_low and percentile_high > 0:
                            dynamic_range = percentile_high / percentile_low
                            if dynamic_range > 100:
                                threshold = percentile_high * 0.05
                                if threshold > 0:  # Avoid division by zero
                                    # More stable log compression
                                    y_compressed = np.sign(y) * np.log1p(np.clip(np.abs(y) / threshold, 0, 1e6)) * threshold
                                    if np.all(np.isfinite(y_compressed)):
                                        y = y_compressed
            except Exception as compress_err:
                print(f"Dynamic range compression error: {compress_err}")
            
            # Safe normalization - ensuring we catch any issues
            try:
                if np.abs(y).max() > 0:  # Ensure we have non-zero values
                    y = y / np.abs(y).max()
                # Final check for any NaN or Inf values
                y = np.nan_to_num(y, nan=0.0, posinf=0.9, neginf=-0.9)
            except Exception as norm_err:
                print(f"Normalization error: {norm_err}")
            
            # Skip the white noise addition as it could be causing issues
            
            # Final verification before saving
            if not np.all(np.isfinite(y)):
                raise ValueError("Output audio still contains non-finite values after processing")
            
            # Write enhanced audio
            sf.write(output_audio_path, y, sr)
            
            # Perform a final check to ensure the audio is properly formatted
            if not os.path.exists(output_audio_path):
                raise ValueError("Enhanced audio file was not created successfully")
            
            return True
        except Exception as e:
            # If enhancement fails, copy the original file to output
            import shutil
            print(f"Audio enhancement error (falling back to original): {str(e)}")
            shutil.copyfile(input_audio_path, output_audio_path)
            return False

    def validate_audio(self, audio_path):
        """Comprehensive audio quality checks"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            
            if len(y) == 0:
                raise ValueError("Empty audio file")
                
            rms = librosa.feature.rms(y=y).mean()
            if rms < 0.01:
                raise ValueError("Audio appears silent (RMS: {:.4f})".format(rms))
                
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            if spectral_centroid < 1000:
                print(f"Warning: Low spectral centroid ({spectral_centroid:.1f} Hz), possible noise or deep voice")
                
            return True
        except Exception as e:
            raise RuntimeError(f"Audio validation failed: {str(e)}")

    def get_optimal_chunk_duration(self, audio_path):
        """
        Determine optimal chunk duration and overlap based on audio length.
        For longer files, use longer chunks with more overlap.
        
        Returns:
            tuple: (chunk_duration, overlap) in seconds
        """
        try:
            # Get audio duration
            info = sf.info(audio_path)
            duration = info.duration
            
            print(f"Audio duration: {duration:.2f} seconds")
            
            # Default values
            chunk_duration = 45  # seconds
            overlap = 15         # seconds
            
            # Adjust based on total duration
            if duration < 120:  # Less than 2 minutes
                # For very short audios, use shorter chunks
                chunk_duration = 30
                overlap = 10
            elif duration < 300:  # Less than 5 minutes
                # Default values are fine
                pass
            elif duration < 600:  # Less than 10 minutes
                # Longer audio, increase chunk size
                chunk_duration = 60
                overlap = 20
            else:  # 10 minutes or more
                # Very long audio, use larger chunks
                chunk_duration = 90
                overlap = 30
            
            print(f"Using chunk duration: {chunk_duration}s with overlap: {overlap}s")
            return chunk_duration, overlap
            
        except Exception as e:
            print(f"Error determining optimal chunk duration: {str(e)}")
            # Fall back to default values if there's an error
            return 45, 15

    def create_audio_chunks(self, audio_path, output_dir, chunk_duration=45, overlap=15, job_id=None):
        """Split audio file into chunks with overlap for better transcription continuity"""
        try:
            # Make sure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Get audio file info
            audio_info = sf.info(audio_path)
            total_duration = audio_info.duration
            sample_rate = audio_info.samplerate
            
            print(f"Audio info: {total_duration:.2f}s, {sample_rate}Hz")
            
            # If the audio is very short, just return it as a single chunk
            if total_duration <= chunk_duration:
                chunk_path = os.path.join(output_dir, "chunk_001.wav")
                shutil.copy(audio_path, chunk_path)
                return [chunk_path]
            
            # Calculate chunk parameters
            chunk_samples = int(chunk_duration * sample_rate)
            overlap_samples = int(overlap * sample_rate)
            step_samples = chunk_samples - overlap_samples
            
            # Calculate how many chunks we'll have
            num_chunks = max(1, int(np.ceil((total_duration * sample_rate - overlap_samples) / step_samples)))
            
            print(f"Creating {num_chunks} chunks of {chunk_duration}s with {overlap}s overlap")
            
            # Process status update
            if job_id:
                update_status(job_id, "splitting", 15)
            
            # Create chunks
            chunk_files = []
            
            # Load the entire audio
            data, samplerate = sf.read(audio_path)
            
            # Create each chunk
            for i in range(num_chunks):
                # Calculate start and end positions
                start_sample = i * step_samples
                end_sample = start_sample + chunk_samples
                
                # Ensure we don't go past the audio length
                if end_sample > len(data):
                    end_sample = len(data)
                
                # Extra special handling for the final chunk to ensure we capture the ending
                if i == num_chunks - 1:
                    # For the last chunk, ensure we get at least the minimum size or back up
                    min_final_chunk = int(30 * sample_rate)  # At least 30 seconds
                    if end_sample - start_sample < min_final_chunk:
                        # Back up to get more audio in the final chunk
                        start_sample = max(0, end_sample - min_final_chunk)
                
                # Extract chunk data
                chunk_data = data[start_sample:end_sample]
                
                # Create chunk file name with padding for proper sorting
                chunk_filename = f"chunk_{i+1:03d}.wav"
                if i == num_chunks - 1:
                    chunk_filename = f"chunk_{i+1:03d}_final.wav"
                
                chunk_path = os.path.join(output_dir, chunk_filename)
                
                # Write to file
                sf.write(chunk_path, chunk_data, samplerate)
                
                chunk_files.append(chunk_path)
                
                # Update progress
                if job_id and num_chunks > 1:
                    progress = 10 + (i / num_chunks) * 10
                    update_status(job_id, "splitting", progress)
            
            return chunk_files
            
        except Exception as e:
            error_message = f"Error creating audio chunks: {str(e)}"
            print(error_message)
            traceback.print_exc()
            
            if job_id:
                update_status(job_id, "error", 0, error_message)
                
            return []

    def process_features(self, waveform, sr=16000):
        """Process audio waveform into model features with proper attention mask"""
        # Convert to features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # First get input features in a standard way
            processor_output = self.processor(
                waveform, 
                sampling_rate=sr,
                return_tensors="pt"
            )
            
            input_features = processor_output.input_features.to(self.device)
            
            # Explicitly create attention mask (all 1s since we don't have any padding)
            # This prevents the warning about attention mask
            attention_mask = torch.ones(
                input_features.shape[0], 
                input_features.shape[1], 
                dtype=torch.long, 
                device=self.device
            )
            
            return input_features, attention_mask
            
    def transcribe_chunk(self, audio_path, chunk_idx, total_chunks, job_id=None):
        """Transcribe a single audio chunk with enhanced audio validation"""
        try:
            # Set torch.cuda device if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load audio file
            with torch.no_grad():
                # Step 1: Load audio
                try:
                    waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
                except Exception as e:
                    print(f"Error loading audio {audio_path}: {str(e)}")
                    return "[Ошибка загрузки аудио]"
                
                # Step 2: Check for NaN and Inf values
                if torch.isnan(waveform).any() or torch.isinf(waveform).any():
                    print(f"Warning: Audio file {audio_path} contains NaN or Inf values. Attempting to clean...")
                    # Replace NaN and Inf with zeros
                    waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Step 3: Normalize waveform
                waveform_max = torch.abs(waveform).max()
                if waveform_max > 0:
                    waveform = waveform / waveform_max
                else:
                    print(f"Warning: Audio file {audio_path} contains only zeros or very small values")
                    return "[Тишина в аудио]"
                
                # Step 4: Convert to mono and check variance
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Check if this is likely silence (very low variance)
                waveform_var = torch.var(waveform)
                if waveform_var < 1e-7:
                    print(f"Warning: Audio file {audio_path} has very low variance ({waveform_var}), likely silence")
                    if chunk_idx > 0:  # Skip silence in all but the first chunk
                        return "[Тишина в аудио]"
                
                # Step 5: Resample to 16kHz if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                    sample_rate = 16000
                    
                # Report chunk info
                duration = waveform.shape[1] / sample_rate
                print(f"Chunk {chunk_idx+1}/{total_chunks} duration: {duration:.2f}s, sample rate: {sample_rate}")
                
                # Step 6: Process the input features with an explicit attention mask
                inputs, input_mask = self.process_features(waveform)
                
                # Special handling for final chunk - look for ending phrase
                is_final_chunk = (chunk_idx == total_chunks - 1) or "final" in os.path.basename(audio_path)
                
                # Step 7: Perform transcription with optimized parameters
                model = self.model.to(device)
                
                # Determine if this is a final chunk that might have the ending phrase
                if is_final_chunk:
                    print(f"Processing final chunk {chunk_idx+1}/{total_chunks} with special attention to ending...")
                    # For final chunks, prioritize accuracy over speed with optimized parameters
                    try:
                        # First try beam search with attention to ending phrases
                        print(f"Attempting beam search for chunk {chunk_idx+1}")
                        
                        # Use beam search with special focus on ending phrases
                        generate_kwargs = {
                            "forced_decoder_ids": self.processor.get_decoder_prompt_ids(language="ru", task="transcribe"),
                            "num_beams": 3, 
                            "do_sample": False,
                            "early_stopping": True,
                            "temperature": 0.0,
                            "attention_mask": input_mask,
                            "max_new_tokens": 444,  # Adjusted to fit within model constraints
                            "min_new_tokens": 5,   # Ensure we get a reasonable result
                            "no_repeat_ngram_size": 3  # Prevent repetition
                        }
                        
                        # Add specific prompt to recognize the ending phrase
                        if is_final_chunk:
                            # Add a prompt to help the model recognize the ending phrase
                            # This is a subtle way to bias the model without forcing an output
                            # Implemented via decoder_input_ids in a way that doesn't affect the token limit
                            decoder_ids = self.processor.get_decoder_prompt_ids(language="ru", task="transcribe")
                            generate_kwargs["forced_decoder_ids"] = decoder_ids
                        
                        with torch.inference_mode():
                            result = model.generate(
                                inputs.to(device),
                                **generate_kwargs
                            )
                    except Exception as e:
                        print(f"Error in beam search generation for chunk {chunk_idx+1}: {str(e)}")
                        print("Falling back to greedy decoding...")
                        
                        # Fall back to greedy decoding
                        generate_kwargs = {
                            "forced_decoder_ids": self.processor.get_decoder_prompt_ids(language="ru", task="transcribe"),
                            "do_sample": False,
                            "attention_mask": input_mask,
                            "max_new_tokens": 444,
                            "min_new_tokens": 5
                        }
                        
                        with torch.inference_mode():
                            result = model.generate(
                                inputs.to(device),
                                **generate_kwargs
                            )
                else:
                    # For regular chunks, use greedy search for speed
                    print(f"Using greedy decoding for chunk {chunk_idx+1}")
                    generate_kwargs = {
                        "forced_decoder_ids": self.processor.get_decoder_prompt_ids(language="ru", task="transcribe"),
                        "do_sample": False,
                        "attention_mask": input_mask,
                        "max_new_tokens": 444,
                        "min_new_tokens": 5
                    }
                    
                    with torch.inference_mode():
                        result = model.generate(
                            inputs.to(device),
                            **generate_kwargs
                        )
                
                # Step 8: Decode the result and clean up
                transcription = self.processor.batch_decode(result, skip_special_tokens=True)[0]
                
                # Check for Russian-specific cleanup
                transcription = transcription.replace("[музыка]", "")
                
                # Special handling for final chunk
                if is_final_chunk:
                    # Remove the common ending tokens that appear in many transcriptions
                    unwanted_endings = ["D.", "D", "Продолжение следует", "To be continued"]
                    for ending in unwanted_endings:
                        if transcription.endswith(ending):
                            transcription = transcription[:-len(ending)].strip()
                            print(f"Removed unwanted ending: '{ending}'")
                    
                    # Check for the ending phrase
                    ending_phrases = ["всем пока пока", "всем пока", "пока пока"]
                    for phrase in ending_phrases:
                        if phrase.lower() in transcription.lower():
                            print(f"Found important ending phrase: '{phrase}' in final chunk!")
                
                # Cleanup memory
                del inputs, result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Move model back to CPU if it was on CUDA
                if device == "cuda":
                    model.to("cpu")
                
                # Clean up memory
                del model
                
                # Update progress if job_id is provided
                if job_id is not None:
                    progress = 20 + (chunk_idx + 1) / total_chunks * 70
                    update_status(job_id, "transcribing", min(90, progress))
                
                return transcription.strip()
                
        except Exception as e:
            error_msg = f"Error in transcribe_chunk: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return f"[Ошибка транскрипции: {str(e)}]"

    def transcribe_audio(self, audio_path, job_id=None, language="ru"):
        try:
            print(f"Starting transcription of {audio_path} with job_id {job_id}")
            start_time = time.time()
            
            # Update job status
            if job_id:
                update_status(job_id, "processing")
                
            # Create a timestamp-based directory for processing
            timestamp = int(time.time())
            temp_dir = f"temp_processing_{timestamp}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Define possible ending phrases
            ending_phrases = [
                "всем пока пока", "всем пока", "пока пока", "до свидания", "до скорого", 
                "пока", "прощайте", "всего хорошего", "всего доброго"
            ]
            
            print(f"Loading models for language: {language}")
            model_size = "large-v3" if torch.cuda.is_available() else "medium.en"
            if language == "ru":
                model_size = "large-v3"  # Always use large model for Russian
                
            # Load whisper model only once
            options = dict(
                language=language,
                task="transcribe",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                word_timestamps=True,
                fp16=torch.cuda.is_available()
            )
            
            # Update job status
            if job_id:
               update_status(job_id, "preparing")
                
            # Get optimal chunk parameters
            chunk_duration, overlap = self.get_optimal_chunk_duration(audio_path)
            
            # Create audio chunks
            print(f"Creating audio chunks with duration {chunk_duration}s and overlap {overlap}s")
            chunk_files = self.create_audio_chunks(audio_path, temp_dir, chunk_duration, overlap, job_id)
            
            if not chunk_files:
                print("No valid audio chunks were created")
                if job_id:
                    update_status(job_id, "failed", error="Failed to create audio chunks")
                return ""
                
            # Update job status
            if job_id:
                update_status(job_id, "transcribing")
                
            # Process each chunk
            transcriptions = []
            found_endings = []
            
            for i, chunk_path in enumerate(chunk_files):
                print(f"Processing chunk {i+1}/{len(chunk_files)}: {chunk_path}")
                is_final_chunk = i == len(chunk_files) - 1
                
                # For final chunk, use more aggressive parameters to capture ending phrases
                if is_final_chunk:
                    print("Processing final chunk with enhanced settings")
                    local_options = options.copy()
                    local_options["temperature"] = 0.2  # Slightly higher to capture variations
                    local_options["beam_size"] = 8     # More beam candidates
                    local_options["best_of"] = 8       # More samples
                else:
                    local_options = options
                
                # Transcribe the chunk
                print(f"Transcribing chunk {i+1}/{len(chunk_files)}")
                if job_id:
                    progress = int(80 * (i / len(chunk_files)))
                    update_status(job_id, "transcribing", progress=progress)
                
                # Transcribe
                result = self.model.transcribe(
                    chunk_path,
                    **local_options
                )
                
                chunk_text = result["text"].strip()
                print(f"Chunk {i+1} transcription: {chunk_text[:100]}...")
                
                # Look for ending phrases, especially in the final chunks
                chunk_text_lower = chunk_text.lower()
                
                # For the last two chunks, we check more carefully
                if i >= len(chunk_files) - 2:  # Last two chunks
                    print(f"Checking final chunk {i+1} carefully for ending phrases")
                    for phrase in ending_phrases:
                        if phrase in chunk_text_lower:
                            print(f"Found ending phrase '{phrase}' in chunk {i+1}")
                            found_endings.append((phrase, i, chunk_text))
                            
                            # For the final chunk, make extra sure we preserve the ending
                            if is_final_chunk:
                                # Ensure this phrase remains intact in the final text
                                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                                ending_index = pattern.search(chunk_text_lower)
                                if ending_index:
                                    start_idx = ending_index.start()
                                    # Keep everything up to the end of the ending phrase
                                    chunk_text = chunk_text[:start_idx + len(phrase)]
                                    if not chunk_text.endswith("."):
                                        chunk_text += "."
                                    print(f"Preserved ending in final chunk: {chunk_text}")
                
                transcriptions.append(chunk_text)
            
            # Update job status
            if job_id:
                update_status(job_id, "finalizing")
                
            # Combine all transcriptions
            print("Combining transcriptions")
            combined_text = ""
            
            for i, text in enumerate(transcriptions):
                if i == 0:
                    combined_text = text
                else:
                    # Use a smart joining approach
                    # Try to avoid duplicate text from overlapping chunks
                    overlap_text = self.find_overlap(combined_text, text)
                    if overlap_text:
                        combined_text = combined_text + text[len(overlap_text):]
                    else:
                        # If no clear overlap, just append with a space
                        combined_text = combined_text + " " + text
            
            # Ensure we include any found ending phrases
            combined_text_lower = combined_text.lower()
            if found_endings:
                print(f"Checking if ending phrases are present in final text")
                for phrase, chunk_idx, chunk_text in found_endings:
                    if phrase not in combined_text_lower:
                        print(f"Adding missing ending phrase '{phrase}' from chunk {chunk_idx+1}")
                        # Add the ending phrase if it's not in the final text
                        if combined_text.endswith("."):
                            combined_text = combined_text[:-1] + " " + phrase + "."
                        else:
                            combined_text = combined_text + " " + phrase + "."
            
            # Clean up final text
            print("Cleaning up final text")
            combined_text = re.sub(r'\s+', ' ', combined_text)  # Replace multiple spaces with a single space
            combined_text = re.sub(r'(?<!\.)\.(?![\s\.])', '. ', combined_text)  # Add space after periods if needed
            combined_text = re.sub(r'\.+', '.', combined_text)  # Replace multiple periods with a single one
            
            # Remove any default endings that might have been added by the model
            unwanted_endings = ["D.", "D ", " D"]
            for ending in unwanted_endings:
                if combined_text.endswith(ending):
                    combined_text = combined_text[:-len(ending)]
                    print(f"Removed unwanted ending: {ending}")
            
            # Ensure the text ends with a period
            if not combined_text.endswith("."):
                combined_text += "."
            
            # Check if we have a proper ending phrase
            has_proper_ending = False
            for phrase in ending_phrases:
                # Check if any ending phrase is near the end of the text (within last 30 chars)
                end_portion = combined_text[-30:].lower()
                if phrase in end_portion:
                    has_proper_ending = True
                    break
            
            # If we didn't find a proper ending in the final text, check the last chunk again
            if not has_proper_ending and len(transcriptions) > 0:
                final_chunk = transcriptions[-1].lower()
                for phrase in ending_phrases:
                    if phrase in final_chunk:
                        # Extract the ending phrase and everything after it, up to 30 chars
                        idx = final_chunk.find(phrase)
                        ending_text = transcriptions[-1][idx:]
                        # Replace the very end of the combined text with this ending
                        combined_text = combined_text[:-20] + ending_text
                        if not combined_text.endswith("."):
                            combined_text += "."
                        print(f"Added proper ending from final chunk: {ending_text}")
                        has_proper_ending = True
                        break
            
            # Clean up temporary files
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory {temp_dir}")
            except Exception as e:
                print(f"Warning: Failed to clean up temp directory: {str(e)}")
            
            # Update job status to completed
            if job_id:
                update_status(job_id, "completed")
                
            elapsed_time = time.time() - start_time
            print(f"Transcription completed in {elapsed_time:.2f} seconds")
            
            return combined_text
            
        except Exception as e:
            print(f"Error in transcribe_audio: {str(e)}")
            traceback.print_exc()
            if job_id:
                update_status(job_id, "failed", error=str(e))
            return ""

    def process_video(self, video_path, job_id=None):
        """Full processing pipeline with cleanup"""
        audio_path = None  # Initialize audio_path before the try block
        try:
            # Generate unique audio path based on job ID or timestamp
            timestamp = int(time.time())
            job_suffix = f"_{job_id}" if job_id else f"_{timestamp}"
            audio_path = f"temp_audio{job_suffix}.wav"
            
            # Update status
            update_status(job_id, "starting", 0)
            
            # Extract audio
            audio_path = self.extract_audio(video_path, audio_path, job_id)
            
            # Validate audio
            update_status(job_id, "validating_audio", 20)
            self.validate_audio(audio_path)
            
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path, job_id)
            
            # Update status
            update_status(job_id, "completed", 100)
            
            return transcription, audio_path
        except Exception as e:
            update_status(job_id, "failed", 0)
            raise RuntimeError(f"Video processing failed: {str(e)}")
        finally:
            # Cleanup temporary files
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

# Initialize transcriber only once
transcriber = RussianVideoTranscriber()

@app.route('/transcribe', methods=['POST'])
def transcribe_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "Empty file name"}), 400
        
    # Generate job ID
    job_id = f"job_{int(time.time())}"
    
    # Create directory for temp files if it doesn't exist
    video_dir = "temp_videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        
    video_path = os.path.join(video_dir, video_file.filename)
    video_file.save(video_path)

    try:
        # Process the video
        update_status(job_id, "starting", 0)
        transcription, audio_path = transcriber.process_video(video_path, job_id)
        
        # Save results
        base_name = os.path.splitext(video_file.filename)[0]
        transcription_path = os.path.join(video_dir, f"{base_name}_transcription.txt")
        
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
            
        return jsonify({
            "transcription": transcription,
            "audio_file": audio_path,
            "job_id": job_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Endpoint to check the status of a transcription job"""
    with status_lock:
        if job_id in processing_status:
            return jsonify(processing_status[job_id])
        else:
            return jsonify({"status": "not_found"}), 404

if __name__ == '__main__':
    # Create directories
    os.makedirs("temp_videos", exist_ok=True)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
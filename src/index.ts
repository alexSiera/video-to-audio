import { VideoProcessor } from "./utils/VideoProcessor.js";
import { TranscriptionService } from "./services/TranscriptionService.js";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Define output paths upfront
const OUTPUT_DIR = join(__dirname, "..", "output");
const DEFAULT_AUDIO_PATH = join(OUTPUT_DIR, "audio.mp3");
const DEFAULT_TRANSCRIPTION_PATH = join(OUTPUT_DIR, "transcription.txt");

async function processVideo(
  videoPath: string,
  audioPath: string = DEFAULT_AUDIO_PATH,
  transcriptionPath: string = DEFAULT_TRANSCRIPTION_PATH
): Promise<void> {
  try {
    // Ensure output directory exists
    if (!existsSync(OUTPUT_DIR)) {
      await mkdir(OUTPUT_DIR, { recursive: true });
    }

    // Extract audio from video
    console.log("Extracting audio from video...");
    await VideoProcessor.extractAudio(videoPath, audioPath);
    console.log("Audio extraction completed successfully!");

    // Transcribe audio
    console.log("Starting transcription...");
    const transcription = await TranscriptionService.transcribeAudio(audioPath);
    console.log("Transcription completed successfully!");

    // Save transcription to file
    await writeFile(transcriptionPath, transcription, "utf-8");
    console.log(`Transcription saved to: ${transcriptionPath}`);
    console.log("\nTranscription result:");
    console.log(transcription);
  } catch (error) {
    console.error("Error processing video:", error);
    process.exit(1);
  }
}

// Check if video path is provided
const videoPath = process.argv[2];
if (!videoPath) {
  console.error("Please provide the path to the video file as an argument");
  console.error("Usage: npm start path/to/your/video.mp4");
  process.exit(1);
}

processVideo(videoPath);

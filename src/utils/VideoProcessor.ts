import { exec } from "child_process";
import { promisify } from "util";
import ffmpeg from "ffmpeg-static";
import fs from "fs";
import path from "path";

const execAsync = promisify(exec);

export class VideoProcessor {
  // Handle the FFmpeg path correctly
  private static readonly ffmpegPath: string =
    typeof ffmpeg === "string" ? ffmpeg : "";

  public static async extractAudio(
    videoPath: string,
    outputPath: string
  ): Promise<void> {
    try {
      if (!this.ffmpegPath) {
        throw new Error(
          "FFmpeg path not found. Please ensure ffmpeg-static is properly installed."
        );
      }

      // Ensure the output directory exists
      const outputDir = path.dirname(outputPath);
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }

      console.log(`Extracting audio from: ${videoPath}`);
      console.log(`Saving audio to: ${outputPath}`);

      const command = `${this.ffmpegPath} -i "${videoPath}" -vn -acodec libmp3lame "${outputPath}"`;
      const { stderr } = await execAsync(command);

      if (stderr && !stderr.includes("video:0kB")) {
        // ffmpeg outputs info to stderr
        console.warn("FFmpeg warnings:", stderr);
      }
    } catch (error) {
      throw new Error(
        `Failed to extract audio: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }
}

import { pipeline } from "@xenova/transformers";
import fs from "fs";

export class TranscriptionService {
  private static readonly MODEL_NAME = "Xenova/whisper-large-v3-turbo-russian";
  private static pipelineInstance: any = null;

  private static async getPipeline() {
    if (!this.pipelineInstance) {
      console.log(`Loading Whisper model: ${this.MODEL_NAME}`);
      this.pipelineInstance = await pipeline(
        "automatic-speech-recognition",
        this.MODEL_NAME
      );
    }
    return this.pipelineInstance;
  }

  public static async transcribeAudio(audioPath: string): Promise<string> {
    try {
      // Check if audio file exists
      if (!fs.existsSync(audioPath)) {
        throw new Error(`Audio file not found at path: ${audioPath}`);
      }

      console.log(`Starting transcription of: ${audioPath}`);
      const transcriber = await this.getPipeline();

      // Define options before using them
      const options = {
        chunk_length_s: 30,
        stride_length_s: 5,
        language: "russian",
        task: "transcribe",
      };

      console.log("Transcription in progress...");
      const result = await transcriber(audioPath, options);

      console.log("Transcription completed");
      return result.text;
    } catch (error) {
      console.error("Transcription error:", error);
      throw new Error(
        `Failed to transcribe audio: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }
}

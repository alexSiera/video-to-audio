// @ts-nocheck

import express, { Request, Response, NextFunction } from 'express';
import multer from 'multer';
import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs';
import { nodewhisper } from 'nodejs-whisper';
import { exec } from 'child_process';
import { promisify } from 'util';
import nlp from 'ru-compromise';

const execPromise = promisify(exec);

// Configuration
const app = express();
const port = process.env.PORT || 3000;
const MODEL_NAME = 'large-v3-turbo';
const UPLOAD_DIR = path.resolve(__dirname, '../uploads');

// Type definitions
interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  isQuestion?: boolean;
}

interface ProcessingResult {
  transcription: TranscriptSegment[];
  audioClips: string[];
  model: string;
}

// Shared utilities
const audioProcessor = {
  normalizeAudio: async (inputPath: string, outputPath: string) => {
    await execPromise(
      `sox ${inputPath} -r 16k ${outputPath} norm -1.5 compand 0.3,1 -80,-80,-60,-60,-50,-20,0,0 -6 -1 0.2`,
    );
  },

  convertToWav: async (inputPath: string, outputPath: string) => {
    await new Promise<void>((resolve, reject) => {
      ffmpeg(inputPath)
        .audioChannels(1)
        .audioFrequency(16000)
        .audioCodec('pcm_s16le')
        //@ts-ignore
        .on('end', resolve)
        .on('error', reject)
        .save(outputPath);
    });
  },

  detectQuestions: (segments: TranscriptSegment[]): TranscriptSegment[] => {
    return segments.map((segment) => {
      const doc = nlp(segment.text);
      return {
        ...segment,
        isQuestion:
          doc.questions().length > 0 ||
          segment.text.includes('?') ||
          /^(кто|что|как|почему|когда|где)/i.test(segment.text),
      };
    });
  },

  createAudioClips: async (
    sourcePath: string,
    segments: TranscriptSegment[],
  ): Promise<string[]> => {
    const clipPaths: string[] = [];

    for (const [index, segment] of segments.entries()) {
      const clipPath = path.join(UPLOAD_DIR, `clip_${index}_${uuidv4()}.wav`);

      await new Promise<void>((resolve, reject) => {
        ffmpeg(sourcePath)
          .setStartTime(segment.start)
          .setDuration(segment.end - segment.start)
          .on('end', () => {
            clipPaths.push(clipPath);
            resolve();
          })
          .on('error', reject)
          .save(clipPath);
      });
    }

    return clipPaths;
  },
};

// Shared processing pipeline
const processMediaFile = async (
  inputPath: string,
): Promise<ProcessingResult> => {
  const normalizedPath = path.join(UPLOAD_DIR, `normalized_${uuidv4()}.wav`);

  // Convert and normalize audio
  await audioProcessor.convertToWav(inputPath, normalizedPath);
  await audioProcessor.normalizeAudio(normalizedPath, normalizedPath);

  // Transcribe with Whisper
  const segments = (await nodewhisper(normalizedPath, {
    modelName: MODEL_NAME,
    autoDownloadModelName: MODEL_NAME,
    removeWavFileAfterTranscription: true,
    whisperOptions: {
      language: 'ru',
      outputInText: false,
    },
  })) as unknown as TranscriptSegment[];

  // Detect questions and create clips
  const processedSegments = audioProcessor.detectQuestions(segments);
  const audioClips = await audioProcessor.createAudioClips(
    inputPath,
    processedSegments,
  );

  return { transcription: processedSegments, audioClips, model: MODEL_NAME };
};

// File upload configuration
const storage = multer.diskStorage({
  destination: UPLOAD_DIR,
  filename: (req, file, cb) => {
    cb(null, `${uuidv4()}${path.extname(file.originalname)}`);
  },
});

const fileUpload = (allowedMimeTypes: string[]) =>
  multer({
    storage,
    fileFilter: (req, file, cb) => {
      allowedMimeTypes.includes(file.mimetype)
        ? cb(null, true)
        : cb(new Error('Неподдерживаемый формат файла'));
    },
  });

// Routes
app.post(
  '/process-video',
  fileUpload(['video/mp4', 'video/quicktime']).single('video'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      if (!req.file) {
        res.status(400).json({ error: 'Не загружен видеофайл' });
        return;
      }

      // Extract audio from video
      const audioPath = path.join(UPLOAD_DIR, `${uuidv4()}.wav`);
      await new Promise<void>((resolve, reject) => {
        ffmpeg(req?.file?.path)
          .audioChannels(1)
          .audioFrequency(16000)
          .outputOptions('-acodec pcm_s16le')
          //@ts-ignore
          .on('end', resolve)
          .on('error', reject)
          .save(audioPath);
      });

      const result = await processMediaFile(audioPath);
      res.json(result);
    } catch (error) {
      next(error);
    } finally {
      if (req.file?.path) fs.unlinkSync(req.file.path);
    }
  },
);

app.post(
  '/transcribe-audio',
  fileUpload(['audio/mp4', 'audio/wav', 'audio/x-m4a']).single('audio'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      if (!req.file) {
        res.status(400).json({ error: 'Не загружен аудиофайл' });
        return;
      }

      const result = await processMediaFile(req.file.path);
      res.json(result);
    } catch (error) {
      next(error);
    } finally {
      if (req.file?.path) fs.unlinkSync(req.file.path);
    }
  },
);

// Error handling
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error('Ошибка:', err);
  res.status(500).json({ error: err.message || 'Внутренняя ошибка сервера' });
});

app.listen(port, () => {
  console.log(`Сервер запущен на http://localhost:${port}`);
});

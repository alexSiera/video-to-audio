import express, { Request, Response, NextFunction } from 'express';
import multer from 'multer';
import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs';
import { nodewhisper } from 'nodejs-whisper';
import { exec } from 'child_process';
import { promisify } from 'util';

const execPromise = promisify(exec);

const app = express();
const port = process.env.PORT || 3000;

const MODEL_NAME = 'large-v3-turbo';

const UPLOAD_DIR = path.resolve(__dirname, '../uploads');

const storage = multer.diskStorage({
  destination: UPLOAD_DIR,
  filename: (req, file, cb) => {
    cb(null, `${uuidv4()}${path.extname(file.originalname)}`);
  },
});

// Добавляем конфигурацию Multer для аудио
const audioUpload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['audio/mp4', 'audio/wav', 'audio/x-m4a'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Неподдерживаемый формат аудио'));
    }
  },
});

const upload = multer({ storage });

app.post(
  '/process-video',
  upload.single('video'),
  async (req: Request, res: Response, next: NextFunction) => {
    let videoPath = '';
    let audioPath = '';
    let normalizedAudioPath = '';

    try {
      if (!req.file) {
        res.status(400).json({ error: 'Не загружен видеофайл' });
        return;
      }

      videoPath = req.file.path;
      audioPath = path.join(UPLOAD_DIR, `${uuidv4()}.wav`);
      normalizedAudioPath = path.join(UPLOAD_DIR, `normalized_${uuidv4()}.wav`);

      await new Promise<void>((resolve, reject) => {
        ffmpeg(videoPath)
          .audioChannels(1)
          .audioFrequency(16000)
          .outputOptions('-acodec pcm_s16le')
          .on('end', () => resolve())
          .on('error', reject)
          .save(audioPath);
      });

      await execPromise(
        `sox ${audioPath} -r 16k ${normalizedAudioPath} norm -0.5 compand 0.3,1 -90,-90,-70,-70,-60,-20,0,0 -5 0 0.2`,
      );
      console.log(MODEL_NAME, 'MODEL_CONFIG.NAME');
      const result = await nodewhisper(normalizedAudioPath, {
        modelName: MODEL_NAME,
        autoDownloadModelName: MODEL_NAME,
        removeWavFileAfterTranscription: true,
        whisperOptions: {
          language: 'ru',
          outputInText: true,
        },
      });

      res.json({ transcription: result, model: MODEL_NAME });
    } catch (error) {
      next(error);
    } finally {
      [videoPath, audioPath, normalizedAudioPath].forEach((p) => {
        if (p && fs.existsSync(p)) fs.unlinkSync(p);
      });
    }
  },
);

app.post(
  '/transcribe-audio',
  audioUpload.single('audio'),
  async (req: Request, res: Response, next: NextFunction) => {
    let audioPath = '';
    let normalizedAudioPath = '';

    try {
      if (!req.file) {
        res.status(400).json({ error: 'Не загружен аудиофайл' });
        return;
      }

      audioPath = req.file.path;
      normalizedAudioPath = path.join(UPLOAD_DIR, `normalized_${uuidv4()}.wav`);

      // Конвертация
      await new Promise<void>((resolve, reject) => {
        ffmpeg(audioPath)
          .audioChannels(1)
          .audioFrequency(16000)
          .audioCodec('pcm_s16le')
          //@ts-ignore
          .on('end', resolve)
          .on('error', reject)
          .save(normalizedAudioPath);
      });

      // Нормализация
      const { stdout, stderr } = await execPromise(
        `sox ${normalizedAudioPath} -r 16k ${normalizedAudioPath}_temp.wav norm -1.5 compand 0.3,1 -80,-80,-60,-60,-50,-20,0,0 -6 -1 0.2`,
      );

      if (stderr && stderr.includes('error')) {
        throw new Error(`SOX error: ${stderr}`);
      }

      fs.renameSync(`${normalizedAudioPath}_temp.wav`, normalizedAudioPath);

      // Транскрипция
      const result = await nodewhisper(normalizedAudioPath, {
        modelName: MODEL_NAME,
        autoDownloadModelName: MODEL_NAME,
        removeWavFileAfterTranscription: true,
        whisperOptions: {
          language: 'ru',
          outputInText: true,
        },
      });

      res.json({ transcription: result, model: MODEL_NAME });
    } catch (error) {
      next(error);
    } finally {
      [audioPath, normalizedAudioPath].forEach((p) => {
        if (p && fs.existsSync(p)) fs.unlinkSync(p);
      });
    }
  },
);

app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error('Global error handler:', err);
  res.status(500).json({
    error: err.message || 'Internal Server Error',
  });
});

app.listen(port, () => {
  console.log(`Сервер запущен на http://localhost:${port}`);
});

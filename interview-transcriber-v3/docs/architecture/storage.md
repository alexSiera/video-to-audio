# Object Storage Design

> Этап 2. Хранение файлов и артефактов.
> Зависит от: `system-design.md` (upload pipeline, pipeline artifacts), `database.md` (storage_key).
> Статус: **на утверждение**.

---

## 1. Выбор backend

| Среда | Backend | Примечание |
|---|---|---|
| Dev | **MinIO** (в docker-compose) | S3-совместимый, одна команда, веб-консоль на :9001 |
| Prod | **Любой S3-совместимый** (AWS S3, Cloudflare R2, Yandex Object Storage, Selectel) | Меняется через env, код не правится |

Доступ через **boto3** (или `aioboto3` для async) с абстракцией `ObjectStorage` (интерфейс в `packages/shared`-стиле внутри `apps/api`/`apps/worker`). Это даёт:
- прозрачную замену MinIO ↔ S3;
- лёгкое мокирование в тестах.

### 1.1 Почему не локальный диск / не БД
- Файлы до 10 часов = гигабайты; БД и диск не масштабируются под это.
- Object storage даёт: presigned URLs (браузер льёт напрямую), multipart upload, lifecycle/retention, версионирование.
- Единый путь dev↔prod.

---

## 2. Структура бакетов и ключей

Один физический бакет в dev; в prod — разделить по назначению (см. §2.4). Ключи — детерминированные, читаемые.

### 2.1 Бакет `media` — пользовательские исходники
```
u/{user_id}/uploads/{upload_id}/source/{filename}
u/{user_id}/uploads/{upload_id}/audio.wav          # извлечённое аудио (из браузера или сервера)
u/{user_id}/uploads/{upload_id}/video.mp4          # (опц.) отдельно загруженное видео для плеера
```

### 2.2 Бакет `artifacts` — промежуточные/итоговые артефакты пайплайна
```
u/{user_id}/jobs/{job_id}/vad.json
u/{user_id}/jobs/{job_id}/chunks/*.json
u/{user_id}/jobs/{job_id}/raw_transcript.json      # вывод STT до постобработки
u/{user_id}/jobs/{job_id}/aligned.json
u/{user_id}/jobs/{job_id}/diarization.json
u/{user_id}/jobs/{job_id}/merged.json
u/{user_id}/jobs/{job_id}/final.json               # финальный артефакт = источник правок
```
> Ключи **детерминированы по job_id и стадии** — это основа resumable pipeline (`system-design.md` §7.3).

### 2.3 Бакет `exports` — готовые экспорты
```
u/{user_id}/transcripts/{transcript_id}/exports/{export_id}.{ext}
```

### 2.4 Окружение в одном MinIO
В dev все три бакета создаются автоматически при старте (init-контейнер). В prod — отдельные бакеты, разные lifecycle-политики.

---

## 3. Поток данных

```mermaid
sequenceDiagram
    participant B as Browser
    participant API as apps/api
    participant OS as Object Storage
    participant WK as apps/worker

    Note over B,API: 1. Init
    B->>API: POST /uploads/init (size, mime, duration)
    API->>API: validate, create Upload(pending)
    API->>OS: create multipart upload
    API-->>B: {upload_id, presigned PUT parts}

    Note over B,OS: 2. Upload (browser → storage напрямую)
    B->>OS: PUT part 1..N (presigned, chunked)
    B->>API: POST /uploads/{id}/complete (parts)
    API->>OS: complete multipart
    API->>API: ffprobe audio, set Upload(uploaded)

    Note over WK,OS: 3. Pipeline
    WK->>OS: GET audio
    WK->>OS: PUT artifacts per stage (vad/chunks/.../final)
    WK->>API: status updates via DB

    Note over B,OS: 4. Export
    B->>API: POST /transcripts/{id}/export?format=docx
    API->>(WK): enqueue export_job
    WK->>OS: PUT exports/{export_id}.docx
    WK->>API: done
    API-->>B: presigned GET (TTL)
```

---

## 4. Presigned URLs — контракты

| Операция | Метод | TTL | Кто использует |
|---|---|---|---|
| Загрузка части файла | PUT | 15 мин | Browser → Storage (multipart) |
| Скачивание экспорта | GET | 1 час (configurable) | Browser |
| Просмотр видео в редакторе | GET | 1 час | Browser ( HLS/progressive ) |
| Доступ воркера к артефактам | — | без presigned; серверный SDK | Worker → Storage |

Правила:
- Presigned URL выдаёт **только API** после проверки прав пользователя на ресурс.
- В URL не должно быть ничего кроме `object_key` и подписи; метаданные авторизации — только для логирования.
- Логи — без подписанных частей (truncate).

---

## 5. Multipart / resumable upload

- Браузер дробит файл на части (env: `UPLOAD_PART_BYTES`, по умолчанию 8 МБ, max 5 ГБ/часть по лимиту S3).
- Идёмпотентность: SHA-256 каждой части; повтор только не подтверждённых.
- Прогресс загрузки — на клиенте (по отправленным байтам), не в БД (избегаем write-амплитуды).
- При `POST /complete` сервер проверяет количество частей и финальный размер против `client_size_bytes`.

---

## 6. Retention и lifecycle

| Тип данных | Политика | Триггер |
|---|---|---|
| `media/audio.wav` | удалять после `final.json` + grace 24ч (для отладки) | job completed |
| `media/source` | TTL 30 дней от загрузки, если не привязан к транскрипту | lifecycle |
| `media/video.mp4` | TTL 30 дней (редко нужен после транскрибации) | lifecycle |
| `artifacts/*` (промежуточные) | TTL 7 дней после job completed | lifecycle |
| `artifacts/{job}/final.json` | хранить, пока жив транскрипт | cleanup при удалении транскрипта |
| `exports/*` | TTL = `expires_at` (default 24 ч) | lifecycle |

Cleanup-задача:
- Внешний cron (или Dramatiq-periodic) раз в сутки сканирует истёкшие ключи и удаляет.
- Удаление транскрипта/пользователя → каскадное удаление по префиксу `u/{user_id}/...`.

---

## 7. Шифрование

- **In transit**: TLS (MinIO с self-signed cert в dev, Let's Encrypt/prod cert manager в prod).
- **At rest**: SSE-S3 (или SSE-KMS в prod). Ключом управляет backend; код приложения не трогает.
- Политика доступа: бакеты приватные (никакого public-read). Доступ — только presigned.

---

## 8. Резервное копирование

- **Dev**: не критично; MinIO volume можно сбросить.
- **Prod**:
  - `media` и `artifacts` — versioning + cross-region replication (если поддерживает backend) на критичных данных (`final.json`, исходное аудио 7 дней).
  - Регулярный `mc mirror` в холодное хранилище (S3 Glacier/аналог) раз в неделю — опционально.
- БД — отдельная стратегия (см. `deployment.md`).

---

## 9. Безопасность (кратко, подробнее в `security.md`)

- Все presigned URL выдаются **только после** проверки ACL (`user_id` владеет ресурсом).
- Имена ключей детерминированы, но пользователь не контролирует их напрямую (предотвращает path traversal).
- Загружаемые файлы — проверка MIME и `ffprobe` на сервере; подозрительные → `Upload(corrupted)` + quarantine prefix.
- Антивирусная проверка (ClamAV) — опционально в post-v1.

---

## 10. Конфигурация (env)

| Переменная | Назначение | Dev default |
|---|---|---|
| `S3_ENDPOINT` | URL (MinIO/S3) | `http://minio:9000` |
| `S3_REGION` | регион | `us-east-1` |
| `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` | креды | dev-пары MinIO |
| `S3_BUCKET_MEDIA` / `S3_BUCKET_ARTIFACTS` / `S3_BUCKET_EXPORTS` | бакеты | `media`/`artifacts`/`exports` |
| `S3_PRESIGNED_TTL_SECONDS` | TTL presigned GET | `3600` |
| `UPLOAD_PART_BYTES` | размер части multipart | `8388608` |
| `MAX_UPLOAD_BYTES` | лимит размера | `2147483648` (2 ГБ audio) |
| `MAX_AUDIO_DURATION_HOURS` | лимит длительности | `10` |

---

## 11. Открытые вопросы

1. **Хранить ли извлечённое аудио после транскрибации?** Полезно для повторной транскрибации другой моделью. Предложение: хранить 7 дней, затем переход в cold storage.
2. **Direct browser → storage upload или через API proxy?** Direct экономит CPU/трафик API, но требует CORS-настройки MinIO. Предлагаю direct (см. §3). Альтернатива — proxy для совместимости с фронтендами без CORS-поддержки.
3. **Bucket-per-tenant vs shared-with-prefix.** В v1 — shared с префиксом `u/{user_id}/` (проще). На пост-v1, при необходимости изоляции — бакеты на tenant.

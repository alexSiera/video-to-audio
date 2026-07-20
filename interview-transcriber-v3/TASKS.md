# TASKS.md

Главный roadmap проекта `interview-transcriber-v3`. Согласован с `docs/architecture/`.
После выполнения каждой задачи отмечать пункт `[x]` и (опционально) ставить дату и PR-ссылку.

> Принципы работы с этим файлом:
> - Один чекбокс = одна атомарная задача (влезает в один PR, до ~800 строк диффа).
> - Порядок этапов — рекомендуемый, не обязательный; зависимости — внутри этапов.
> - Если задача заблокирована другой — отметить ` (blocked: <task>)`.
> - Если задача устарела — `(~)` с пояснением.

---

## Этап 1 — Исследование (skipped)

> Пропущен по решению (сессия 2026-07-20). При необходимости вернуться и заполнить `docs/research/`.

- [x] ~~Этап пропущен~~

---

## Этап 2 — Проектирование системы

- [x] `docs/architecture/system-design.md` — компоненты, lifecycle, ADR
- [x] `docs/architecture/database.md` — схема PostgreSQL
- [x] `docs/architecture/storage.md` — object storage
- [x] `docs/architecture/queues.md` — Dramatiq + Redis
- [x] `docs/architecture/deployment.md` — docker-compose dev/prod
- [x] `docs/architecture/security.md` — auth, ACL, retention
- [x] `docs/architecture/README.md` — навигация
- [x] `CLAUDE.md` — постоянные правила
- [x] `TASKS.md` — roadmap

---

## Этап 3 — Структура монорепозитория

- [ ] Корневой `package.json` / workspace-конфиг (pnpm workspaces)
- [ ] `apps/web` — скелет Next.js 14 (App Router, TS strict, Tailwind, shadcn/ui init)
- [ ] `apps/api` — скелет FastAPI (uv, pyproject.toml, `/health`)
- [ ] `apps/worker` — скелет Dramatiq-воркера
- [ ] `apps/stt-service` — каркас пакета + интерфейс `SpeechModel`
- [ ] `packages/ui` — пустой пакет + экспорт одной демо-компоненты
- [ ] `packages/shared` — Python-пакет (path-dep) + TS-аналог
- [ ] `packages/config` — общие конфиги (eslint, prettier, tsconfig, ruff, pyproject)
- [ ] `packages/types` — генерация типов из OpenAPI (placeholder pipeline)
- [ ] `scripts/` — bootstrap, dev-скрипты
- [ ] Корневой `Makefile` (`up`, `down`, `lint`, `test`, `migrate`)
- [ ] `.env.example`, `.gitignore`, `.editorconfig`
- [ ] `README.md` корневой (overview + one-command start)

---

## Этап 4 — Frontend (apps/web)

- [ ] Layout: sidebar, topbar, тема (RU/EN i18n init)
- [ ] Страница **Dashboard** (последние проекты/транскрипты)
- [ ] Страница **Projects** (CRUD)
- [ ] Страница **Upload** (drag & drop, множественная загрузка, прогресс)
- [ ] Страница **Transcript Editor** (каркас: плеер + текст + спикеры)
- [ ] Страница **Settings** (язык, дефолтная модель)
- [ ] TanStack Query: клиент + auth-интерсептор
- [ ] Zustand-стор: сессия, настройки UI
- [ ] React Hook Form + Zod: формы (логин, загрузка, настройки)
- [ ] Framer Motion: базовые переходы/микро-анимации
- [ ] WebSocket-клиент: прогресс джоба, обновления редактора
- [ ] Toast/уведомления (shadcn/ui sonner)
- [ ] Error boundaries + skeletons + empty-states

---

## Этап 5 — Upload Pipeline (frontend)

- [ ] Интеграция `@ffmpeg/ffmpeg` (FFmpeg.wasm)
- [ ] Probe кодека/длительности в браузере
- [ ] Извлечение аудио → 16kHz mono WAV/Opus
- [ ] Стриминг-чанкинг в wasm (для длинных файлов)
- [ ] Расчёт SHA-256 частей (для resumable)
- [ ] Multipart upload через presigned URL (browser → storage напрямую)
- [ ] Прогресс загрузки (по отправленным байтам)
- [ ] Resumable: повтор только незавершённых частей
- [ ] Опциональная отдельная загрузка видео (для плеера)
- [ ] Fallback на серверное извлечение при размере > порога
- [ ] Валидация на клиенте (тип, длительность, размер)

---

## Этап 6 — Backend (apps/api)

- [ ] FastAPI app + роутер `/api/v1`
- [ ] Pydantic v2 schemas (DTO)
- [ ] SQLAlchemy 2.x async engine + session
- [ ] Alembic init + миграция `0001_init_users_auth`
- [ ] Слои: `presentation / application / domain / infrastructure`
- [ ] Dependency Injection (FastAPI deps)
- [ ] Repository pattern (base + per-entity)
- [ ] `/health` (DB + Redis ping)
- [ ] OpenAPI экспорт + автогенерация типов в `packages/types`
- [ ] structlog JSON-логирование
- [ ] Sentry SDK init
- [ ] OpenTelemetry instrumentation (FastAPI, SQLAlchemy, Redis)
- [ ] Глобальный error-handler middleware
- [ ] CORS + security headers

---

## Этап 7 — Очередь задач (apps/worker)

- [ ] Dramatiq setup + Redis broker
- [ ] Очереди: `default`, `stt-gpu`, `export`
- [ ] Actor `run_pipeline` (оркестратор)
- [ ] Middleware: logging, retries, timeout, Sentry
- [ ] Graceful shutdown (SIGTERM handling)
- [ ] Resumability: чтение `jobs.progress_stage` + существующих артефактов
- [ ] Pub/sub прогресса в Redis (`jobs:{id}`)
- [ ] DLQ consumer (`dead` → `jobs.status=failed`)
- [ ] Periodic cleanup (истёкшие exports, lifecycle) — Dramatiq periodic или cron

---

## Этап 8 — STT Service (apps/stt-service)

- [ ] Интерфейс `SpeechModel` (Protocol) + типы (`AudioSegment`, `TranscribeOptions`, `TranscribeResult`)
- [ ] Registry моделей (по env `STT_MODEL`)
- [ ] Адаптер **Faster-Whisper** (`large-v3`, `large-v3-turbo`)
- [ ] Lazy/eager init (env-controlled)
- [ ] `estimate_time` / `estimate_cost` (формулы + тесты)
- [ ] Адаптер NVIDIA NeMo (Parakeet/Canary) — экспериментально
- [ ] CPU-fallback (`STT_DEVICE=cpu`, int8)
- [ ] Hotwords / initial prompt для RU
- [ ] Unit-тесты адаптеров на коротких аудио

---

## Этап 9 — Pipeline обработки

- [ ] `pipeline.normalize` (FFmpeg: 16kHz mono, peak-normalize)
- [ ] `pipeline.vad` (Silero VAD v5, интерфейс `Vad`)
- [ ] `pipeline.chunking` (Sliding/Adaptive + overlap)
- [ ] `pipeline.alignment` (WhisperX / wav2vec2 RU+EN)
- [ ] `pipeline.diarization` (pyannote 3.x; резерв NeMo)
- [ ] `pipeline.merge` (разрешение оверлапов, склейка)
- [ ] `pipeline.postfix` (RU пунктуация/регистр: правила + опц. LLM)
- [ ] `pipeline.artifact` (детерминированные ключи, идемпотентность)
- [ ] Интеграция всех этапов в `run_pipeline`
- [ ] Тесты: boundary-чанков (нет потерь слов), идемпотентность рестарта

---

## Этап 10 — Chunking (отдельный модуль)

- [ ] Sliding Window стратегия
- [ ] Adaptive Window стратегия
- [ ] Overlap + merge по пересечению
- [ ] Streaming Chunking (для будущего real-time)
- [ ] Chunk Merge (защита от потерь слов на границах)
- [ ] Тесты: длинная пауза, склейка слов через границу, очень короткий чанк

---

## Этап 11 — Диаризация

- [ ] Единый интерфейс `Diarizer`
- [ ] Реализация pyannote.audio 3.x (проверить лицензию!)
- [ ] Реализация-резерв: NeMo diarizer / Sortformer
- [ ] Маппинг speaker labels → `transcript_speakers`
- [ ] Обработка overlap (несколько спикеров одновременно)
- [ ] Тесты на known-multi-speer аудио

---

## Этап 12 — Transcript Editor

- [ ] Пагинированная загрузка сегментов (по `ordinal`)
- [ ] Клик по слову → jump в плеере (по `words[].t0`)
- [ ] Редактирование текста (inline)
- [ ] Autosave (debounced PATCH, optimistic concurrency `version`)
- [ ] Undo/redo (минимальный журнал `transcript_edits`)
- [ ] Подсветка и переименование спикеров
- [ ] Поиск (pg_trgm) + навигация по совпадениям
- [ ] Горячие клавиши (play/pause, jump, search)
- [ ] Выделение фрагментов (selection → экспорт части)
- [ ] Синхронизация с видео (если загружено)

---

## Этап 13 — Экспорт

- [ ] Worker-actor `export_transcript`
- [ ] TXT
- [ ] DOCX (python-docx)
- [ ] Markdown
- [ ] JSON (полный артефакт)
- [ ] CSV
- [ ] SRT
- [ ] VTT
- [ ] PDF (WeasyPrint или reportlab)
- [ ] Опции: speakers on/off, timecodes on/off, диапазон сегментов
- [ ] Presigned GET с TTL

---

## Этап 14 — API (REST + WebSocket)

- [ ] `/auth/*` (login, register, refresh, logout, oauth callbacks)
- [ ] `/projects` CRUD
- [ ] `/uploads/init`, `/uploads/{id}/parts`, `/uploads/{id}/complete`
- [ ] `/jobs/{id}` (status, cancel, retry)
- [ ] `/transcripts/{id}` (read, patch segment, speakers rename)
- [ ] `/transcripts/{id}/export`
- [ ] `/transcripts/{id}/search`
- [ ] WebSocket `/ws/jobs/{id}` (прогресс)
- [ ] WebSocket `/ws/transcripts/{id}` (обновления редактора)
- [ ] Rate limiting (slowapi)
- [ ] Pagination на всех list-эндпоинтах
- [ ] OpenAPI finalized → регенер `packages/types`

---

## Этап 15 — Авторизация

- [ ] Argon2id password hashing
- [ ] JWT access token (HS256, 15 мин)
- [ ] Refresh token rotating + family-revoke на reuse
- [ ] Logout (Redis denylist `jti`)
- [ ] Google OAuth (Authorization Code + PKCE)
- [ ] GitHub OAuth
- [ ] Email verification (пароль-регистрация)
- [ ] «Активные сессии» UI (отзыв refresh-токенов)
- [ ] Тесты: reuse detection, expired refresh, ACL

---

## Этап 16 — Docker

- [ ] `apps/web/Dockerfile` (multi-stage)
- [ ] `apps/api/Dockerfile`
- [ ] `apps/worker/Dockerfile` (CPU: default/export)
- [ ] `apps/worker/Dockerfile.gpu` (CUDA-база)
- [ ] `apps/stt-service/Dockerfile` (если нужен standalone; иначе как layer)
- [ ] `docker-compose.dev.yml` (с hot-reload, MinIO, profiles: gpu/cpu)
- [ ] `docker-compose.prod.yml` (Caddy, healthchecks, restart policies)
- [ ] `docker-compose.ci.yml` (ephemeral БД/Redis)
- [ ] init-контейнеры: `migrate` (alembic), `minio-init` (buckets)
- [ ] `.dockerignore` на каждый сервис
- [ ] Makefile: `up`, `gpu-up`, `down`, `migrate`, `seed`, `test`, `lint`, `build`

---

## Этап 17 — Логирование и observability

- [ ] structlog config (redact-процессор для PII)
- [ ] pino config (Next.js)
- [ ] Sentry (Python + JS SDK, PII redaction)
- [ ] OpenTelemetry traces (API → Worker → STT, `traceparent` в metadata джоба)
- [ ] Логирование security events (login, refresh-reuse, ACL-denied)
- [ ] Healthchecks для всех сервисов
- [ ] (post-v1) Prometheus `/metrics`

---

## Этап 18 — Тестирование

- [ ] Unit-тесты: domain-логика, pipeline-этапы, STT-адаптеры
- [ ] Integration-тесты: API эндпоинты (testcontainers PG/Redis/MinIO)
- [ ] Integration-тесты: end-to-end pipeline на коротком аудио
- [ ] Тесты ACL (cross-user denied)
- [ ] Тесты resumability (рестарт воркера посередине)
- [ ] Тесты chunking на граничных случаях
- [ ] Покрытие ≥ 70% на core-модулях
- [ ] CI: `lint`, `typecheck`, `unit`, `integration`, `trivy`, `alembic upgrade/downgrade`

---

## Пост-v1 (backlog, не в этом roadmap)

- [ ] Биллинг / подписки (Stripe)
- [ ] Команды и шеринг проектов (RBAC: owner/editor/viewer)
- [ ] Совместное редактирование в реальном времени (CRDT)
- [ ] Мобильные приложения
- [ ] Автоперевод / саммаризация
- [ ] Meilisearch/Typesense для поиска
- [ ] Kubernetes / multi-region
- [ ] 2FA
- [ ] API keys для внешних интеграций
- [ ] Webhooks

---

## Статус по этапам (сводка)

| Этап | Статус | Прогресс |
|---|---|---|
| 1. Исследование | skipped | — |
| 2. Проектирование | ✅ done | 9/9 |
| 3. Структура монорепо | ⏳ next | 0/13 |
| 4. Frontend | pending | 0/14 |
| 5. Upload Pipeline | pending | 0/11 |
| 6. Backend | pending | 0/14 |
| 7. Очередь задач | pending | 0/8 |
| 8. STT Service | pending | 0/9 |
| 9. Pipeline | pending | 0/11 |
| 10. Chunking | pending | 0/6 |
| 11. Диаризация | pending | 0/6 |
| 12. Transcript Editor | pending | 0/10 |
| 13. Экспорт | pending | 0/11 |
| 14. API | pending | 0/12 |
| 15. Авторизация | pending | 0/9 |
| 16. Docker | pending | 0/12 |
| 17. Observability | pending | 0/7 |
| 18. Тестирование | pending | 0/8 |

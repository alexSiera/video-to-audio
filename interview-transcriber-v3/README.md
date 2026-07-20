# interview-transcriber-v3

SaaS-сервис транскрибации длинных интервью и разговоров (до 10 часов), с фокусом на **русский язык** (EN — вторично),
и удобным редактором транскрипта.

> Статус: **скелет монорепозитория (Этап 3)**. Бизнес-логика и пайплайн добавляются на последующих этапах.

---

## Документация

- [`docs/architecture/`](./docs/architecture/) — проектирование системы (Этап 2): system-design, database, storage, queues, deployment, security.
- [`CLAUDE.md`](./CLAUDE.md) — постоянные правила разработки (Clean Architecture, SOLID, ограничения кода).
- [`TASKS.md`](./TASKS.md) — roadmap по всем этапам.

---

## Стек

| Слой | Технологии |
|---|---|
| Frontend | Next.js 14, React, TypeScript (strict), TailwindCSS, shadcn/ui, TanStack Query, Zustand, React Hook Form, Framer Motion |
| Backend | Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.x (async), Alembic |
| Worker | Python 3.12, Dramatiq + Redis |
| STT | Faster-Whisper (CTranslate2, GPU); интерфейс `SpeechModel` для замены |
| Infra | PostgreSQL 16, Redis 7, MinIO (dev) / S3-совместимый (prod), Docker Compose |
| Observability | structlog, pino, Sentry, OpenTelemetry |

---

## Структура монорепозитория

```
apps/
  web/           Next.js frontend
  api/           FastAPI (REST + WebSocket)
  worker/        Dramatiq workers (pipeline execution)
  stt-service/   SpeechModel interface + адаптеры (GPU)
packages/
  ui/            Переиспользуемые UI-компоненты (shadcn-based)
  shared/        Общий TS-код (форматирование, константы)
  shared-py/     Общий Python-код (исключения, утилиты)
  config/        Общие конфиги (eslint, tsconfig, prettier)
  types/         Контракты API ↔ web (генерируются из OpenAPI)
docs/            Архитектура, runbooks
scripts/         Утилиты (bootstrap, dev-скрипты)
```

Каждый сервис в `apps/` — независимый процесс со своим `Dockerfile` и набором зависимостей.

---

## Быстрый старт

### Требования
- Node.js 20+, pnpm 10+
- Python 3.12+ (рекомендуется через [uv](https://docs.astral.sh/uv/))
- Docker + Docker Compose v2
- (Опционально) NVIDIA GPU + драйверы для GPU-режима

### One-command start (dev)

```bash
make bootstrap      # install deps + поднять сервисы + применить миграции
# или по шагам:
make install        # uv sync + pnpm install
make up             # поднять dev-окружение (CPU)
make migrate        # применить миграции БД
```

С GPU:
```bash
make gpu-up         # включает GPU-воркер (--gpus all)
```

Сервисы:
- Web: http://localhost:3000
- API: http://localhost:8000 (OpenAPI: `/docs`)
- MinIO console: http://localhost:9001 (minioadmin / minioadmin)

### Частые команды

```bash
make logs           # логи всех сервисов
make ps             # статус контейнеров
make lint           # линт Python + TS
make typecheck      # mypy + tsc
make test           # unit-тесты
make test-integration
make migrate-new MSG="init users"   # создать миграцию
make down           # остановить
make reset          # ОСТОРОЖНО: удалить volumes и пересоздать
```

---

## Разработка

Перед любой нетривиальной задачей — смотри `CLAUDE.md` (правила) и `docs/architecture/` (контракт).
Архитектура этапа должна быть согласована до реализации.

Коммит-сообщения — Conventional Commits (`feat:`, `fix:`, `docs:`, ...). Ветка `main` — стабильно собирается, тесты проходят.

---

## Лицензия

(Определяется отдельно. Скелет не содержит кода под специфической лицензией.)

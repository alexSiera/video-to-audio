# Architecture Documentation

> Этап 2. Проектирование системы.
> Все документы в этом каталоге согласуются между собой; изменения в одном — требуют ревью зависимых.

---

## Карта документов

| Документ | О чём |
|---|---|
| **[system-design.md](./system-design.md)** | Высокоуровневая архитектура: компоненты, lifecycle, ADR-сводка, риски. Читать первым. |
| **[database.md](./database.md)** | Схема PostgreSQL, ER-диаграмма, индексы, миграции. |
| **[storage.md](./storage.md)** | Object storage (MinIO/S3): бакеты, ключи, presigned URL, retention. |
| **[queues.md](./queues.md)** | Dramatiq + Redis: топология очередей, retry/DLQ, прогресс для UI. |
| **[deployment.md](./deployment.md)** | Docker compose (dev/prod), GPU, one-command start, CI/CD, backup. |
| **[security.md](./security.md)** | Auth, ACL, безопасность файлов, GDPR/retention, секреты. |

---

## Зафиксированные решения (session 2026-07-20)

| Решение | Выбор |
|---|---|
| STT-стратегия | Self-hosted GPU, swappable `SpeechModel` |
| Масштаб v1 | Dev-first, single `docker-compose`; GPU в отдельном воркере |
| Языки v1 | RU (primary) + EN |
| Сессия | Документы Этапа 2; код не пишется до утверждения |

---

## Открытые вопросы (требуют решения до/во время реализации)

Сводка из всех документов — каждый пункт раскрыт в конце соответствующего файла.

### Архитектура
1. Browser-side audio extraction как дефолт? — `system-design.md` §5
2. `stt-service` как библиотека, а не микросервис? — ADR-003
3. Faster-Whisper large-v3 как стартовый дефолт? — ADR-007
4. Resumable pipeline с чекпойнтами vs рестарт с нуля в v1? — `system-design.md` §7.3
5. Диаризация через pyannote (резерв NeMo по лицензии)?

### Данные
6. `words` как jsonb vs отдельная таблица? — `database.md` §6
7. Хранить `original_text` для diff/аудита? — `database.md` §6
8. Включать ли Row-Level Security как второй слой ACL? — `database.md` §6

### Storage
9. Хранить извлечённое аудио после транскрибации (7 д. + cold)?
10. Browser → storage напрямую vs через API proxy?
11. Bucket-per-tenant vs shared-with-prefix?

### Queues
12. Оркестратор vs event-driven этапы?
13. Совмещать `default` и `export` очереди в одном CPU-воркере?
14. Dramatiq vs RQ?

### Deployment
15. Один VPS с GPU vs отдельный GPU-хост?
16. Caddy vs Nginx?
17. Observability-стек в v1 (минимум Sentry + логи)?

### Security
18. Обязательная верификация email при парольной регистрации?
19. 2FA в v1? (предлагаем отложить)
20. Token storage: access в памяти + refresh в httponly cookie?

---

## Ключевые принципы (из спецификации)

- **Без утверждённой архитектуры код не пишется.**
- Clean Architecture + SOLID + DRY + KISS.
- Один файл ≤ 300–500 строк; функция ≤ 40–60 строк; класс — одна ответственность.
- Feature-first организация; общее — в `packages/`.
- TypeScript strict; Python — обязательные type hints.
- Каждый публичный модуль — с документацией; каждый новый модуль — с тестами.
- Перед завершением этапа проект собирается и тесты проходят.

Эти правила фиксируются в `CLAUDE.md` (создаётся на этапе инициализации кода).

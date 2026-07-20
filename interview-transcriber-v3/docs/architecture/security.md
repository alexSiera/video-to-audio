# Security

> Этап 2. Безопасность сервиса.
> Зависит от: `system-design.md` (auth, upload), `storage.md` (presigned), `database.md` (tokens, soft-delete).
> Статус: **на утверждение**.

---

## 1. Threat model (STRIDE, кратко)

| Угроза | Сценарий | Контроль |
|---|---|---|
| Spoofing | Чужой токен | JWT verify + rotating refresh + family detection |
| Tampering | Подмена правки/файла | Optimistic concurrency, server-side validation, SHA-256 |
| Repudiation | «Я не загружал» | Аудит (`job_events`, app logs с user_id) |
| Info disclosure | Доступ к чужому транскрипту | ACL на каждом ресурсе, presigned URLs с проверкой владения |
| DoS | Заливка файлов / тяжёлые запросы | Rate limits, квоты на пользователя, очереди |
| Elevation of privilege | Подделка user_id | Все команды берут user_id из токена, не из тела |
| Malicious media | FFmpeg-bomb / эксплойт кодека | server-side `ffprobe` на метаданные, размер/длительность limits, sandbox обработки |

---

## 2. Аутентификация

### 2.1 Пароли
- `argon2id` через `passlib` (рекомендация OWASP).
- Минимум 12 символов; проверка на top-пароли (HIBP-style, опционально post-v1).
- Учётная запись может быть OAuth-only (`password_hash IS NULL`).

### 2.2 JWT
- **Access token**: HS256 (или RS256 при нескольких инстансах API; в v1 HS256), TTL 15 мин.
- Claims: `sub` (user_id), `iat`, `exp`, `jti`.
- Верификация на каждый запрос в FastAPI dependency; `Authorization: Bearer <token>`.
- Отзыв «досрочно» — через короткий TTL + Redis denylist `jti` для logout.

### 2.3 Refresh token
- Случайный 256-битный токен, хранится только **хэш** (`token_hash`) в БД.
- TTL 30 дней, rotating: каждый refresh выдаёт новую пару, инвалидизирует старый токен в том же `family_id`.
- **Reuse detection**: при использовании отозванного токена внутри семьи → отзывать всю семью (compromise signal), требовать повторный логин.
- `user_agent`/`ip` логируются для UI «Активные сессии».

### 2.4 OAuth (Google, GitHub)
- Flow: Authorization Code + **PKCE**.
- Callback: `/auth/oauth/{provider}/callback`.
- На новый `provider_user_id` — создаётся user (OAuth-only) или линкуется к существующему по совпадению email (только если email верифицирован у провайдера).
- `state` + `code_verifier` в сессии (cookie, httponly, samesite=strict).

---

## 3. Авторизация (RBAC-lite)

В v1 роли нет, есть только **владелец**. Ресурс (`project`, `upload`, `transcript`) доступен только его `user_id`.
- Все репозитории фильтруют по `user_id` (base repository).
- Серверные проверки прав **обязательны** перед выдачей presigned URL или изменением ресурса.
- Post-v1: шеринг проекта (role: `owner`, `editor`, `viewer`) — отдельная таблица `project_members`.

---

## 4. API hardening

- **CORS**: whitelist origin (env `WEB_ORIGIN`). Никакого `*`.
- **CSRF**: для cookie-based auth — `SameSite=Strict` + double-submit token. Для Bearer-token auth CSRF не применим (токен не в cookie).
- **Rate limiting**: `slowapi` (или `redis-rate`) на:
  - `/auth/login` — 5/мин/IP.
  - `/auth/register` — 3/час/IP.
  - `/uploads/init` — 10/мин/user.
  - остальные — 100/мин/user (configurable).
- **Payload limits**: тело ≤ 1 МБ для JSON-эндпоинтов; файлы только через presigned → storage.
- **Validation**: Pydantic strict, `max_length`, allow-list enum-полей.
- **Headers**: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Content-Security-Policy` на web.

---

## 5. Безопасность файлов

### 5.1 На upload
- Сервер валидирует `client_mime` против allow-list: `audio/*`, `video/*`.
- После завершения загрузки — `ffprobe` для проверки реального контейнера/кодека и длительности. Если `client_*` не совпадает с реальным — `Upload(corrupted)`, quarantine prefix `quarantine/...`, без обработки.
- Размер и длительность — против лимитов (`MAX_UPLOAD_BYTES`, `MAX_AUDIO_DURATION_HOURS`).
- **SHA-256** каждого файла хранится; используется для дедупликации (в рамках одного пользователя).

### 5.2 На обработку
- Worker работает с файлом в **scratch-директории** контейнера; после этапа — удаление.
- `ffmpeg`/`ffprobe` вызываются с `-nostdin`, явными аргументами (без shell=True), таймаутом.
- Антивирус (ClamAV) — опционально post-v1; для v1 — server-side probe + лимиты достаточно.

### 5.3 На download / presigned
- Presigned URL выдаётся только после проверки ACL.
- TTL короткий (1 ч для просмотра, 15 мин для upload parts).
- Имена ключей — детерминированы сервером, не контролируются клиентом (анти path traversal).

---

## 6. Данные и приватность

### 6.1 GDPR / 152-ФЗ
- Возможность **экспорта** всех данных пользователя (JSON).
- Возможность **удаления** аккаунта → каскадное удаление БД-записей + объекты в storage по префиксу `u/{user_id}/`.
- Политика хранения (retention) публична в `/legal/retention`.

### 6.2 Retention (по умолчанию, configurable)
| Данные | Срок |
|---|---|
| Исходное аудио | 7 дней после `final.json` (или по выбору пользователя) |
| Видео | 7 дней |
| Промежуточные артефакты | 7 дней |
| Финальный транскрипт | пока не удалён пользователем |
| Логи | 30 дней |
| Refresh-токены | до истечения/отзыва |
| Экспорты | 24 часа |

### 6.3 Шифрование
- **In transit**: TLS везде (Caddy в prod, MinIO с TLS при необходимости).
- **At rest**: SSE-S3 (managed) по умолчанию; SSE-KMS опционально в prod. БД — disk encryption на уровне хостинга.

### 6.4 PII в логах
- Логи — `user_id` (не email), `job_id`, `trace_id`.
- Никогда не логировать: содержимое транскрипта, имена файлов в полном виде (truncate), токены, секреты.
- structlog-процессор «redact» для известных полей.

---

## 7. Секреты

- В образе — ноль секретов.
- В prod — через Docker secrets / env, инжектируемые orchestration.
- Ротация: `JWT_SECRET` rotation с graceful-окном (два допустимых секрета). `S3_*` — по необходимости.
- `HUGGINGFACE_TOKEN` для gated-моделей — только на GPU-воркере, не в API.

---

## 8. Зависимости

- Запрет на пакеты с известными CVE (`pip-audit`, `npm audit`, `trivy` в CI).
- Lock-файлы обязательны (`uv.lock`, `pnpm-lock.yaml`).
- Dependabot/Renovate на репозитории.

---

## 9. Observability для security

- Sentry на ошибки (PII redaction включён).
- Security-event log: логины, refresh-reuse, подозрительные загрузки, ACL-denied — отдельный поток (`security` topic).
- Алерты: всплеск `401/403`, refresh-reuse, массовые запросы к `/uploads/init`.

---

## 10. Открытые вопросы

1. **Email verification обязательна?** Предлагаю: пароль-регистрация требует верификации email перед первым логином; OAuth — считается верифицированным, если провайдер это гарантирует.
2. **2FA в v1?** Предлагаю отложить (post-v1). Подготовить схему `user_factors` заранее — нет.
3. **Cookie vs LocalStorage для токена.** Предлагаю access-token в памяти + refresh в `httponly, samesite=strict` cookie. Согласовать.
4. **Path-traversal:** детерминированные ключи и `user_id`-префикс гарантируют изоляцию. Подтвердить подход.
5. **Diarization models и коммерческое использование (лицензия pyannote).** Решаем на этапе 11; резерв — NeMo/Sortformer.

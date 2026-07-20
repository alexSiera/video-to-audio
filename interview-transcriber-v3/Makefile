# interview-transcriber-v3 — корневой Makefile.
# Единая точка входа для dev-команд. См. docs/architecture/deployment.md.

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

COMPOSE_DEV := docker compose -f docker-compose.dev.yml
COMPOSE_PROD := docker compose -f docker-compose.prod.yml
PROFILES :=

# ============================================================
# Help
# ============================================================

.PHONY: help
help: ## Показать список доступных целей
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ============================================================
# Dev lifecycle
# ============================================================

.PHONY: up
up: ## Поднять dev-окружение (CPU режим)
	$(COMPOSE_DEV) up -d

.PHONY: gpu-up
gpu-up: ## Поднять dev-окружение с GPU-воркером
	$(COMPOSE_DEV) --profile gpu up -d

.PHONY: down
down: ## Остановить dev-окружение
	$(COMPOSE_DEV) down

.PHONY: logs
logs: ## Логи всех сервисов (tail)
	$(COMPOSE_DEV) logs -f --tail=200

.PHONY: ps
ps: ## Статус контейнеров
	$(COMPOSE_DEV) ps

.PHONY: rebuild
rebuild: ## Пересобрать образы и поднять
	$(COMPOSE_DEV) up -d --build

.PHONY: reset
reset: ## ОСТОРОЖНО: снести контейнеры, тома и пересоздать
	$(COMPOSE_DEV) down -v
	$(COMPOSE_DEV) up -d

# ============================================================
# DB / migrations
# ============================================================

.PHONY: migrate
migrate: ## Применить миграции (alembic upgrade head)
	$(COMPOSE_DEV) run --rm migrate alembic upgrade head

.PHONY: migrate-new
migrate-new: ## Создать новую миграцию: make migrate-new MSG=name
	$(COMPOSE_DEV) run --rm migrate alembic revision --autogenerate -m "$(MSG)"

.PHONY: migrate-down
migrate-down: ## Откатить последнюю миграцию
	$(COMPOSE_DEV) run --rm migrate alembic downgrade -1

.PHONY: seed
seed: ## Заполнить БД демо-данными
	$(COMPOSE_DEV) run --rm api python -m app.seed

# ============================================================
# Lint / format / typecheck
# ============================================================

.PHONY: lint
lint: ## Линт (Python + TS)
	@echo "--- Python lint (ruff) ---"
	uv run ruff check .
	@echo "--- TypeScript lint (eslint) ---"
	pnpm -r run lint

.PHONY: fmt
fmt: ## Форматирование (ruff format + prettier)
	uv run ruff format .
	uv run ruff check --fix .
	pnpm -r run format || true

.PHONY: typecheck
typecheck: ## Проверка типов (mypy + tsc)
	uv run mypy apps packages || true
	pnpm -r run typecheck

# ============================================================
# Tests
# ============================================================

.PHONY: test
test: ## Unit-тесты (Python + TS)
	@echo "--- Python tests ---"
	uv run pytest -q
	@echo "--- TypeScript tests ---"
	pnpm -r run test || true

.PHONY: test-integration
test-integration: ## Интеграционные тесты (нужны сервисы БД/Redis/MinIO)
	$(COMPOSE_DEV) -f docker-compose.ci.yml up -d postgres redis minio
	uv run pytest -q -m integration
	$(COMPOSE_DEV) -f docker-compose.ci.yml down

# ============================================================
# Install / bootstrap
# ============================================================

.PHONY: install
install: ## Установить все зависимости (uv sync + pnpm install)
	uv sync
	pnpm install

.PHONY: bootstrap
bootstrap: ## Полная инициализация: install + up + migrate
	uv sync
	pnpm install
	$(COMPOSE_DEV) up -d
	$(MAKE) migrate

.PHONY: clean
clean: ## Удалить артефакты сборки
	@find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \
		-o -name .mypy_cache -o -name .turbo -o -name node_modules -o -name dist \
		-o -name .next -o -name "*.egg-info" \) -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "clean done"

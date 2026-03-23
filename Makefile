SHELL := /bin/bash

DEV_COMPOSE := ./scripts/dev-compose.sh
SERVICE ?=

.PHONY: dev-cache dev-build dev-up dev-down dev-restart dev-logs dev-ps dev-audio-up dev-ingest

dev-cache:
	mkdir -p .cache/huggingface

dev-build: dev-cache
	$(DEV_COMPOSE) build $(SERVICE)

dev-build-up: dev-cache
	$(DEV_COMPOSE) up -d --build $(SERVICE)

dev-up: dev-cache
	$(DEV_COMPOSE) up -d

dev-down:
	$(DEV_COMPOSE) down

dev-restart:
	$(DEV_COMPOSE) restart $(SERVICE)

dev-logs:
	$(DEV_COMPOSE) logs -f $(SERVICE)

dev-ps:
	$(DEV_COMPOSE) ps

dev-audio-up: dev-cache
	$(DEV_COMPOSE) --profile audio up -d

dev-ingest: dev-cache
	$(DEV_COMPOSE) --profile ingest up ingest

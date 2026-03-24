SHELL := /bin/bash

DEV_COMPOSE := docker compose -f docker-compose.dev.yaml
VAST_COMPOSE := docker compose -f docker-compose.vast.yaml
SERVICE ?=
DOCKERHUB_NAMESPACE ?= minhquan1906
IMAGE_TAG ?= latest

IMAGE_EMBEDDING := $(DOCKERHUB_NAMESPACE)/eco-rag-embedding:$(IMAGE_TAG)
IMAGE_RERANKER := $(DOCKERHUB_NAMESPACE)/eco-rag-reranker:$(IMAGE_TAG)
IMAGE_GUARD := $(DOCKERHUB_NAMESPACE)/eco-rag-guard:$(IMAGE_TAG)
IMAGE_ASR := $(DOCKERHUB_NAMESPACE)/eco-rag-asr:$(IMAGE_TAG)
IMAGE_TTS := $(DOCKERHUB_NAMESPACE)/eco-rag-tts:$(IMAGE_TAG)
IMAGE_ORCHESTRATOR := $(DOCKERHUB_NAMESPACE)/eco-rag-orchestrator:$(IMAGE_TAG)
IMAGE_INGEST := $(DOCKERHUB_NAMESPACE)/eco-rag-ingest:$(IMAGE_TAG)

.PHONY: test test-integration dev-cache dev-build dev-up dev-down dev-restart dev-logs dev-ps dev-audio-up dev-ingest vast-pull vast-up vast-down vast-logs vast-ps images-build images-push images-build-push

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

# Dev env
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

# VAST 
vast-pull:
	$(VAST_COMPOSE) pull

vast-up:
	$(VAST_COMPOSE) up -d

vast-down:
	$(VAST_COMPOSE) down

vast-logs:
	$(VAST_COMPOSE) logs -f $(SERVICE)

vast-ps:
	$(VAST_COMPOSE) ps

# Build and push images to Hub
images-build:
	test -n "$(DOCKERHUB_NAMESPACE)"
	docker build -t $(IMAGE_EMBEDDING) services/embedding
	docker build -t $(IMAGE_RERANKER) services/reranker
	docker build -t $(IMAGE_GUARD) services/guard
	docker build -t $(IMAGE_ASR) services/asr
	docker build -t $(IMAGE_TTS) services/tts
	docker build -t $(IMAGE_ORCHESTRATOR) api
	docker build -t $(IMAGE_INGEST) scripts

images-push:
	test -n "$(DOCKERHUB_NAMESPACE)"
	docker push $(IMAGE_EMBEDDING)
	docker push $(IMAGE_RERANKER)
	docker push $(IMAGE_GUARD)
	docker push $(IMAGE_ASR)
	docker push $(IMAGE_TTS)
	docker push $(IMAGE_ORCHESTRATOR)
	docker push $(IMAGE_INGEST)

images-build-push: images-build images-push

SHELL := /bin/bash

DEV_COMPOSE := docker compose -f docker-compose.dev.yaml
COMPOSE := docker compose -f docker-compose.yaml
SERVICE ?=
DOCKERHUB_NAMESPACE ?= minhquan1906
IMAGE_TAG ?= latest

IMAGE_INFERENCE := $(DOCKERHUB_NAMESPACE)/eco-rag-inference:$(IMAGE_TAG)
IMAGE_ORCHESTRATOR := $(DOCKERHUB_NAMESPACE)/eco-rag-orchestrator:$(IMAGE_TAG)
IMAGE_INGEST := $(DOCKERHUB_NAMESPACE)/eco-rag-ingest:$(IMAGE_TAG)

.PHONY: test test-integration dev-cache dev-build dev-build-up dev-up dev-down dev-restart dev-logs dev-ps dev-ingest build build-up up down restart logs ps ingest images-build images-push images-build-push

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

dev-ingest: dev-cache
	$(DEV_COMPOSE) --profile ingest up ingest

# Main
build:
	$(COMPOSE) build $(SERVICE)

build-up:
	$(COMPOSE) up -d --build $(SERVICE)

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart $(SERVICE)

logs:
	$(COMPOSE) logs -f $(SERVICE)

ps:
	$(COMPOSE) ps

ingest: 
	$(COMPOSE) --profile ingest up ingest

# Build and push images to Hub
images-build:
	test -n "$(DOCKERHUB_NAMESPACE)"
	docker build -t $(IMAGE_INFERENCE) -f infra/docker/app.gpu.Dockerfile .
	docker build -t $(IMAGE_ORCHESTRATOR) -f infra/docker/app.cpu.Dockerfile .
	docker build -t $(IMAGE_INGEST) -f infra/docker/app.cpu.Dockerfile .

images-push:
	test -n "$(DOCKERHUB_NAMESPACE)"
	docker push $(IMAGE_INFERENCE)
	docker push $(IMAGE_ORCHESTRATOR)
	docker push $(IMAGE_INGEST)

images-build-push: images-build images-push

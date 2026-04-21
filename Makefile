SHELL := /bin/bash

COMPOSE     := docker compose -f docker-compose.yaml
COMPOSE_DEV := docker compose -f docker-compose.dev.yaml

DOCKERHUB_NAMESPACE ?= minhquan1906
IMAGE_TAG           ?= latest
SERVICE             ?=

IMAGE_INFERENCE    := $(DOCKERHUB_NAMESPACE)/eco-rag-inference:$(IMAGE_TAG)
IMAGE_ORCHESTRATOR := $(DOCKERHUB_NAMESPACE)/eco-rag-orchestrator:$(IMAGE_TAG)

QDRANT_URL ?= http://localhost:6333

# ─── Colours ──────────────────────────────────────────────────────────────────
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[0;32m
CYAN  := \033[0;36m
GRAY  := \033[0;90m

.PHONY: help \
        start stop \
        setup pull build push \
        bootstrap bootstrap-docker snapshot-restore \
        up down restart logs ps \
        dev dev-stop dev-build dev-logs dev-ps \
        test test-integration \
        quantize-setup quantize-vlm quantize-vlm-fp8 \
        _require-env _dev-cache _qdrant-wait _snapshot-restore-curl

# ─── Default ──────────────────────────────────────────────────────────────────
help:
	@printf "$(BOLD)Multimodal Economic RAG Assistant$(RESET)\n\n"
	@printf "$(CYAN)E2E$(RESET)\n"
	@printf "  make start                Pull images → data → all services\n"
	@printf "  make stop                 Stop and remove all containers\n\n"
	@printf "$(CYAN)First-time setup$(RESET)\n"
	@printf "  make setup                Create .env from .env.example\n"
	@printf "  make pull                 Pull pre-built images from Docker Hub\n"
	@printf "  make build                Build images locally\n"
	@printf "  make push                 Push images to Docker Hub\n\n"
	@printf "$(CYAN)Data$(RESET)\n"
	@printf "  make snapshot-restore     Upload data/*.snapshot → Qdrant  (needs uv)\n"
	@printf "  make bootstrap            Create collection + indexes locally (needs uv)\n"
	@printf "  make bootstrap-docker     Same, but runs inside the orchestrator container\n\n"
	@printf "$(CYAN)Runtime$(RESET)\n"
	@printf "  make up      [SERVICE]    Start services\n"
	@printf "  make down    [SERVICE]    Stop services\n"
	@printf "  make restart [SERVICE]    Restart\n"
	@printf "  make logs    [SERVICE]    Follow logs\n"
	@printf "  make ps                   Container status\n\n"
	@printf "$(CYAN)Dev$(RESET)\n"
	@printf "  make dev                  Start dev stack (hot-reload, bind mounts)\n"
	@printf "  make dev-stop             Stop dev stack\n"
	@printf "  make dev-build [SERVICE]  Build dev images\n"
	@printf "  make dev-logs  [SERVICE]  Follow dev logs\n\n"
	@printf "$(CYAN)Tests$(RESET)\n"
	@printf "  make test                 Unit tests\n"
	@printf "  make test-integration     Integration tests\n\n"
	@printf "$(CYAN)Quantization$(RESET)\n"
	@printf "  make quantize-setup       Create .venv-quantize (run once on vast.ai)\n"
	@printf "  make quantize-vlm [ARGS]  Quantize Qwen3.5-4B with W4A16 GPTQ (llm-compressor)\n"
	@printf "  make quantize-vlm-fp8     Quantize with FP8-dynamic (data-free, safer fallback)\n\n"
	@printf "$(GRAY)Variables: DOCKERHUB_NAMESPACE=$(DOCKERHUB_NAMESPACE)  IMAGE_TAG=$(IMAGE_TAG)  QDRANT_URL=$(QDRANT_URL)$(RESET)\n"

# ─── E2E (Docker-only, no uv required) ───────────────────────────────────────

## Cloud instance quickstart: make setup → fill .env → make start
start: _require-env pull
	@printf "$(GREEN)►$(RESET) Starting Qdrant...\n"
	@$(COMPOSE) up -d qdrant
	@$(MAKE) _qdrant-wait
	@if ls data/*.snapshot 1>/dev/null 2>&1; then \
		printf "$(GREEN)►$(RESET) Snapshot found — restoring...\n"; \
		$(MAKE) _snapshot-restore-curl; \
	fi
	@printf "$(GREEN)►$(RESET) Bootstrapping collection indexes...\n"
	@$(MAKE) bootstrap-docker
	@printf "$(GREEN)►$(RESET) Starting all services...\n"
	@$(COMPOSE) up -d
	@printf "$(BOLD)$(GREEN)✓ Stack is up$(RESET) — open http://localhost:8080\n"
	@$(MAKE) ps

stop:
	$(COMPOSE) down

# ─── Setup ────────────────────────────────────────────────────────────────────

setup:
	@[ -f .env ] \
		&& printf ".env already exists\n" \
		|| (cp .env.example .env && printf "$(GREEN)✓$(RESET) .env created — fill in your secrets then run $(BOLD)make start$(RESET)\n")

pull:
	docker pull $(IMAGE_INFERENCE)
	docker pull $(IMAGE_ORCHESTRATOR)

build:
	docker build -t $(IMAGE_INFERENCE)    -f infra/docker/app.gpu.Dockerfile .
	docker build -t $(IMAGE_ORCHESTRATOR) -f infra/docker/app.cpu.Dockerfile .

push:
	@test -n "$(DOCKERHUB_NAMESPACE)" || (printf "DOCKERHUB_NAMESPACE is not set\n"; exit 1)
	docker push $(IMAGE_INFERENCE)
	docker push $(IMAGE_ORCHESTRATOR)

# ─── Data ─────────────────────────────────────────────────────────────────────

## Requires uv — use on dev machines or CI
snapshot-restore:
	SERVICES__QDRANT_URL=$(QDRANT_URL) uv run python scripts/qdrant_snapshot_restore.py

bootstrap:
	SERVICES__QDRANT_URL=$(QDRANT_URL) uv run python scripts/qdrant_bootstrap.py

## Runs inside orchestrator container — no local Python needed
bootstrap-docker:
	$(COMPOSE) --profile tools run --rm bootstrap

# ─── Runtime ──────────────────────────────────────────────────────────────────

up:
	$(COMPOSE) up -d $(SERVICE)

down:
	$(COMPOSE) down $(SERVICE)

restart:
	$(COMPOSE) restart $(SERVICE)

logs:
	$(COMPOSE) logs -f $(SERVICE)

ps:
	$(COMPOSE) ps

# ─── Dev ──────────────────────────────────────────────────────────────────────

dev: _dev-cache
	$(COMPOSE_DEV) up -d

dev-stop:
	$(COMPOSE_DEV) down

dev-build: _dev-cache
	$(COMPOSE_DEV) build $(SERVICE)

dev-logs:
	$(COMPOSE_DEV) logs -f $(SERVICE)

dev-ps:
	$(COMPOSE_DEV) ps

# ─── Tests ────────────────────────────────────────────────────────────────────

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

# ─── Quantization (runs on vast.ai RTX 3090 or local GPU) ────────────────────

## W4A16 GPTQ (primary). Runs in .venv-quantize (separate from main env due to dep conflicts).
## Setup once: python3 -m venv .venv-quantize && .venv-quantize/bin/pip install -r scripts/requirements-quantize.txt
## Usage: make quantize-vlm ARGS="--push-to-hub --hub-id <namespace>/Qwen3.5-4B-W4A16"
quantize-vlm:
	.venv-quantize/bin/python scripts/quantize_llmcompressor.py --scheme w4a16 $(ARGS)

## FP8 dynamic (data-free, always works on Qwen MoE).
quantize-vlm-fp8:
	.venv-quantize/bin/python scripts/quantize_llmcompressor.py --scheme fp8-dynamic $(ARGS)

## Bootstrap quantize venv (run once on vast.ai before quantize-vlm).
quantize-setup:
	python3 -m venv .venv-quantize
	.venv-quantize/bin/pip install --upgrade pip
	.venv-quantize/bin/pip install -r scripts/requirements-quantize.txt
	@printf "$(GREEN)✓$(RESET) quantize venv ready — run $(BOLD)make quantize-vlm$(RESET)\n"

# ─── Internal ─────────────────────────────────────────────────────────────────

_require-env:
	@[ -f .env ] || (printf "$(BOLD).env not found$(RESET) — run $(BOLD)make setup$(RESET) first\n"; exit 1)

_dev-cache:
	@mkdir -p .cache/huggingface

_qdrant-wait:
	@printf "$(GRAY)  waiting for Qdrant...$(RESET)\n"
	@until curl -sf $(QDRANT_URL)/healthz > /dev/null 2>&1; do sleep 2; done
	@printf "$(GREEN)  Qdrant ready$(RESET)\n"

# Inline snapshot upload via curl — no Python or uv required.
# Reads SERVICES__QDRANT_COLLECTION from .env automatically.
_snapshot-restore-curl:
	$(eval _SNAPSHOT := $(shell ls data/*.snapshot 2>/dev/null | sort | tail -1))
	$(eval _COLLECTION := $(shell grep -m1 '^SERVICES__QDRANT_COLLECTION=' .env | cut -d= -f2))
	@[ -n "$(_SNAPSHOT)" ]   || (printf "no snapshot file found in data/\n"; exit 1)
	@[ -n "$(_COLLECTION)" ] || (printf "SERVICES__QDRANT_COLLECTION not set in .env\n"; exit 1)
	@printf "  uploading $(_SNAPSHOT) → $(_COLLECTION)\n"
	@curl -sf -X POST \
		"$(QDRANT_URL)/collections/$(_COLLECTION)/snapshots/upload?priority=snapshot" \
		-F "snapshot=@$(_SNAPSHOT)" \
		| python3 -c "import sys,json; r=json.load(sys.stdin); print('  done:', r.get('status','?'))"

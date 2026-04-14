# Multimodal Economic RAG Assistant

An OpenAI-compatible multimodal RAG system for Vietnamese economic and financial queries.

This repository combines:

- a FastAPI orchestration layer in `api/`
- retrieval, ranking, safety, ASR, and TTS microservices in `services/`
- a dataset ingestion pipeline in `scripts/`
- Docker Compose environments for development and image-based deployment

The project is Docker-first for runtime, but local Python tooling is still useful for tests, API work, and ingestion debugging.

---

## 1. What This Project Does

The stack is built around a retrieval-augmented generation workflow for Vietnamese economic content.

At a high level:

1. the user sends a chat request to the orchestrator
2. the orchestrator rewrites and classifies the query
3. the query is checked by the safety service
4. embeddings and sparse signals are used for retrieval
5. retrieved passages are reranked
6. the system optionally falls back to web search when internal support is weak
7. the LLM generates an answer grounded in the assembled context
8. the answer is checked again by the safety layer
9. structured citations are appended for text responses

The repository also supports optional speech input/output with ASR and TTS services, plus a standalone ingestion workflow for building the Qdrant collection.

---

## 2. Repository Structure

The main code layout is:

```text
.
├── Makefile
├── README.md
├── api
│   ├── Dockerfile
│   ├── orchestrator
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── pipeline
│   │   │   ├── rag.py
│   │   │   ├── rag_context.py
│   │   │   ├── rag_guard.py
│   │   │   ├── rag_policy.py
│   │   │   └── rag_prompts.py
│   │   ├── routers
│   │   │   ├── audio.py
│   │   │   └── chat.py
│   │   ├── services
│   │   │   ├── asr.py
│   │   │   ├── conversation.py
│   │   │   ├── embedder.py
│   │   │   ├── guard.py
│   │   │   ├── llm.py
│   │   │   ├── reranker.py
│   │   │   ├── retriever.py
│   │   │   ├── sparse_encoder.py
│   │   │   ├── tts.py
│   │   │   └── web_search.py
│   │   └── tracing.py
│   └── requirements.txt
├── docker-compose.dev.yaml
├── docker-compose.yaml
├── docs
│   └── diagrams
├── pyproject.toml
├── scripts
│   ├── Dockerfile
│   ├── chunker.py
│   ├── ingest.py
│   └── requirements.txt
├── services
│   ├── asr
│   │   ├── Dockerfile
│   │   ├── asr_app.py
│   │   └── requirements.txt
│   ├── embedding
│   │   ├── Dockerfile
│   │   ├── embedding_app.py
│   │   └── requirements.txt
│   ├── guard
│   │   ├── Dockerfile
│   │   ├── guard_app.py
│   │   └── requirements.txt
│   ├── reranker
│   │   ├── Dockerfile
│   │   ├── reranker_app.py
│   │   └── requirements.txt
│   └── tts
│       ├── Dockerfile
│       ├── abbreviations.py
│       ├── text_preprocessor.py
│       ├── tts_app.py
│       └── requirements.txt
└── tests
    ├── orchestrator
    ├── scripts
    └── services
```

### Folder-by-folder overview

#### `api/`

Contains the orchestration API.

- `main.py`: FastAPI app startup, service wiring, router registration
- `config.py`: runtime configuration and prompt templates
- `routers/chat.py`: OpenAI-compatible chat endpoint
- `routers/audio.py`: speech-oriented endpoints
- `services/`: clients for embedding, reranking, guard, ASR, TTS, retrieval, and web search
- `pipeline/rag.py`: LangGraph workflow assembly
- `pipeline/rag_prompts.py`: prompt-building logic for RAG responses
- `pipeline/rag_policy.py`: web-fallback heuristics
- `pipeline/rag_guard.py`: safety denial and retry helpers
- `pipeline/rag_context.py`: context assembly and citation finalization

#### `services/`

Contains standalone inference services, each packaged as its own container.

- `embedding/`: dense embedding service
- `reranker/`: reranker service
- `guard/`: safety classification and moderation service
- `asr/`: speech-to-text service
- `tts/`: text-to-speech service

Each service has its own `Dockerfile` and `requirements.txt` because they are built and deployed independently.

#### `scripts/`

Contains the ingestion pipeline.

- `ingest.py`: dataset loading, vector generation, and Qdrant upsert flow
- `chunker.py`: chunking logic and chunk metadata conventions

#### `tests/`

Split by responsibility:

- `tests/orchestrator/`: API, pipeline, config, client, and regression tests
- `tests/services/`: service-container and service-behavior tests
- `tests/scripts/`: ingestion and chunking tests

#### `docs/`

Project documentation and Mermaid diagrams for architecture and workflows.

---

## 3. Architecture Summary

### Core runtime components

- `orchestrator`: user-facing API and workflow coordinator
- `qdrant`: vector database
- `embedding`: dense query embedding service
- `reranker`: passage reranking service
- `guard`: safety classification for prompts and outputs
- `webui`: Open WebUI configured against the orchestrator
- `asr` and `tts`: optional audio services

### Query flow

1. The orchestrator receives a chat request.
2. Conversation utilities normalize messages, resolve the latest user intent, and classify the route.
3. Unsafe inputs are rejected early by the guard service.
4. Dense embeddings and sparse signals are used for hybrid retrieval.
5. Retrieved passages are reranked.
6. If internal support is weak or the question is time-sensitive, web fallback may run.
7. The orchestrator builds a grounded prompt from the final context.
8. The LLM generates an answer.
9. The output is safety-checked.
10. Text responses get a structured citation footer.

### Ingestion flow

1. Load source documents
2. Chunk documents into retrieval units
3. Generate vectors and metadata
4. Create or update the Qdrant collection
5. Upsert chunks into Qdrant

---

## 4. Run Modes

This repository has two main Docker entrypoints:

- `docker-compose.dev.yaml`
  - development stack
  - builds containers from the local source tree
  - intended for active development
  - uses Ollama as the LLM backend in the current setup

- `docker-compose.yaml`
  - image-based deployment stack
  - intended for machines where images are already available or prebuilt
  - expects an NVIDIA GPU host
  - uses the deployment-oriented service images and a vLLM-style OpenAI-compatible LLM endpoint

Use `docker-compose.dev.yaml` for most local work. Use `docker-compose.yaml` when you want a deployment-like stack on a Linux GPU machine.

---

## 5. Platform Support

### Linux

Linux is the primary supported platform for the full stack.

- recommended for full local runtime
- required for realistic NVIDIA GPU deployment
- best option for development plus service execution on one host

### Windows

Windows is workable through Docker Desktop and WSL2.

- recommended setup: Windows 11 + Docker Desktop + WSL2 Ubuntu
- GPU-backed services require WSL2 GPU support
- if GPU passthrough is unavailable, use Windows for development and tests, and point the orchestrator to remote services

### macOS

macOS is appropriate for code work, tests, and documentation, but not for the full GPU-backed inference stack in this repository.

- local development and tests are fine
- full model-serving compose workflows are not the target path
- use a remote Linux GPU host if you need the full runtime

---

## 6. Prerequisites

Install the following first:

- `git`
- Docker with the `docker compose` plugin
- `uv`
- Python 3.12 for local non-Docker workflows

Useful references:

- Docker Desktop: <https://docs.docker.com/desktop/>
- Docker Engine: <https://docs.docker.com/engine/install/>
- Docker Compose plugin: <https://docs.docker.com/compose/install/>
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
- `uv`: <https://docs.astral.sh/uv/getting-started/installation/>

### Install `uv`

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 7. Clone And Configure

Clone the repository and create a local `.env` file:

macOS / Linux:

```bash
git clone https://github.com/minhquana1906/Multimodal-Economic-RAG-Assistant.git
cd Multimodal-Economic-RAG-Assistant
cp .env.example .env
```

Windows PowerShell:

```powershell
git clone https://github.com/minhquana1906/Multimodal-Economic-RAG-Assistant.git
cd Multimodal-Economic-RAG-Assistant
Copy-Item .env.example .env
```

### Important environment variables

Review these first in `.env`:

- `HF_TOKEN`
  - required for Hugging Face model downloads

- `LLM__URL`
  - for the dev compose file, this should match the Ollama service
  - for the deployment compose file, this should point to the OpenAI-compatible LLM endpoint

- `LLM__MODEL`
  - the active model name used by the orchestrator

- `SERVICES__QDRANT_COLLECTION`
  - default retrieval collection name

- `OBSERVABILITY__LANGSMITH_API_KEY`
  - optional, enables LangSmith tracing

- `OBSERVABILITY__TAVILY_API_KEY`
  - optional, enables web search support where configured

- `CLOUDFLARE_TUNNEL_TOKEN`
  - only needed if you use the tunnel service

### Additional variables for `docker-compose.yaml`

The image-based deployment stack expects deployment-oriented LLM variables. A typical setup looks like:

```env
LLM__URL=http://llm:8004/v1
LLM__MODEL=Qwen/Qwen3-4B-Instruct-2507
VLLM_LLM_GPU_MEMORY_UTILIZATION=0.92
VLLM_LLM_MAX_MODEL_LEN=8192
DOCKERHUB_NAMESPACE=your-dockerhub-namespace
IMAGE_TAG=latest
```

If those deployment variables are missing, `docker-compose.yaml` will not behave like the intended production-like environment.

---

## 8. Local Python Environment

Use local Python dependencies when you want to:

- run tests
- work on the API without starting the whole stack
- debug ingestion scripts
- inspect imports and configuration locally

Install the root development environment from the repository root:

```bash
uv sync --dev
```

This uses the dependencies declared in [`pyproject.toml`](pyproject.toml).

### Service-specific dependencies

Some components keep separate `requirements.txt` files because they are packaged as isolated Docker services:

- [`api/requirements.txt`](api/requirements.txt)
- [`scripts/requirements.txt`](scripts/requirements.txt)
- [`services/asr/requirements.txt`](services/asr/requirements.txt)
- [`services/embedding/requirements.txt`](services/embedding/requirements.txt)
- [`services/guard/requirements.txt`](services/guard/requirements.txt)
- [`services/reranker/requirements.txt`](services/reranker/requirements.txt)
- [`services/tts/requirements.txt`](services/tts/requirements.txt)

If you want to run a component outside Docker, install that component's requirements directly.

Example for the orchestrator:

```bash
cd api
uv venv
uv pip install -r requirements.txt
uv run uvicorn orchestrator.main:app --reload --host 0.0.0.0 --port 8000
```

Example for the ingestion pipeline:

```bash
cd scripts
uv venv
uv pip install -r requirements.txt
uv run ingest.py
```

---

## 9. Development Stack

Use `docker-compose.dev.yaml` when you want Docker to build directly from your checked-out source tree.

### Start the main development stack

```bash
docker compose -f docker-compose.dev.yaml up -d --build
```

Equivalent Make targets:

```bash
make dev-build-up
make dev-ps
make dev-logs
```

### Main services and ports

- Orchestrator API: `http://localhost:8000`
- Open WebUI: `http://localhost:8080`
- Qdrant HTTP: `http://localhost:6333`
- Qdrant gRPC: `localhost:6334`
- Embedding service: `http://localhost:8001`
- Reranker service: `http://localhost:8002`
- Guard service: `http://localhost:8003`
- Ollama service: `http://localhost:11434`

### Pull the Ollama model after first startup

The dev stack uses Ollama as the LLM backend. After the container is running, pull the model declared in `.env`.

```bash
docker compose -f docker-compose.dev.yaml exec ollama ollama pull qwen3:0.6b
```

If you changed `LLM__MODEL`, pull that model name instead.

### Health checks

```bash
curl http://localhost:8000/health
curl http://localhost:6333/collections
curl http://localhost:11434/api/tags
```

### Stop the development stack

```bash
docker compose -f docker-compose.dev.yaml down
```

Or:

```bash
make dev-down
```

---

## 10. Optional Audio Services

ASR and TTS are behind the `audio` profile in the development compose file.

```bash
docker compose -f docker-compose.dev.yaml --profile audio up -d --build
```

Or:

```bash
make dev-audio-up
```

Audio service ports:

- ASR: `http://localhost:8005`
- TTS: `http://localhost:8006`

---

## 11. Ingestion Pipeline

The ingestion workflow is responsible for building the Qdrant collection consumed by the orchestrator.

The pipeline typically:

- loads the source dataset
- chunks source documents
- generates vectors and metadata
- creates or updates the target Qdrant collection
- upserts chunk records into Qdrant

### Run ingestion in Docker

```bash
docker compose -f docker-compose.dev.yaml --profile ingest up ingest
```

Or:

```bash
make dev-ingest
```

### Important ingestion settings

Common variables from `.env`:

- `SERVICES__QDRANT_URL=http://qdrant:6333`
- `SERVICES__QDRANT_COLLECTION=econ_vn_news`
- `INGEST__BATCH_SIZE=256`
- `INGEST__FORCE_RECREATE=false`

### When to use `INGEST__FORCE_RECREATE=true`

Set `INGEST__FORCE_RECREATE=true` when you intentionally want to destroy and rebuild the target collection, for example after a chunking schema change.

Example:

```bash
INGEST__FORCE_RECREATE=true docker compose -f docker-compose.dev.yaml --profile ingest up ingest
```

Do not use that casually on valuable data.

### Verify ingestion results

List collections:

```bash
curl http://localhost:6333/collections
```

Inspect collection metadata:

```bash
curl http://localhost:6333/collections/econ_vn_news
```

---

## 12. Image-Based Deployment Stack

Use `docker-compose.yaml` when images already exist in Docker Hub or were built and tagged in advance.

This mode is intended for a Linux NVIDIA GPU host.

### Build and push images

If you publish your own service images:

```bash
make images-build
make images-push
```

Or:

```bash
make images-build-push
```

### Start the deployment stack

```bash
docker compose -f docker-compose.yaml up -d
```

Optional audio profile:

```bash
docker compose -f docker-compose.yaml --profile audio up -d
```

Optional ingestion profile:

```bash
docker compose -f docker-compose.yaml --profile ingest up ingest
```

### Why direct `docker compose` is preferred here

The current production-like compose file is `docker-compose.yaml`, and explicit compose commands make the active runtime target clear.

---

## 13. Qdrant Snapshot Workflow

The local Qdrant service in this repository exposes its HTTP API on `http://localhost:6333`.

The examples below assume:

- collection name: `econ_vn_news`
- no Qdrant API key

If you enable a Qdrant API key, add the appropriate header to your requests.

### Create a snapshot

```bash
curl -X POST "http://localhost:6333/collections/econ_vn_news/snapshots"
```

### List snapshots

```bash
curl "http://localhost:6333/collections/econ_vn_news/snapshots"
```

### Download a snapshot

```bash
curl -L "http://localhost:6333/collections/econ_vn_news/snapshots/<snapshot-name>" -o "./<snapshot-name>"
```

### Upload and recover from a snapshot file

macOS / Linux:

```bash
curl -X POST \
  "http://localhost:6333/collections/econ_vn_news/snapshots/upload?wait=true&priority=snapshot" \
  -F "snapshot=@./econ_vn_news.snapshot"
```

Windows PowerShell:

```powershell
curl.exe -X POST "http://localhost:6333/collections/econ_vn_news/snapshots/upload?wait=true&priority=snapshot" `
  -F "snapshot=@C:\path\to\econ_vn_news.snapshot"
```

### Recover from a snapshot URL or file URI

```bash
curl -X PUT "http://localhost:6333/collections/econ_vn_news/snapshots/recover?wait=true" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "file:///qdrant/snapshots/econ_vn_news.snapshot",
    "priority": "snapshot"
  }'
```

Or from an HTTP URL:

```bash
curl -X PUT "http://localhost:6333/collections/econ_vn_news/snapshots/recover?wait=true" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "https://example.com/econ_vn_news.snapshot",
    "priority": "snapshot"
  }'
```

### Snapshot safety notes

- recovery overwrites collection data
- restore into the correct collection name
- use a separate collection name first when testing a new dataset version
- verify chunking compatibility before mixing old and new ingestion runs

---

## 14. Testing

The canonical test entrypoint for this repository is:

```bash
uv run pytest
```

Matching Make targets:

```bash
make test
make test-integration
```

Integration tests are opt-in:

```bash
uv run pytest -m integration
```

### What the test layout covers

- `tests/orchestrator/`
  - config, routers, pipeline logic, service clients, tracing, regression checks

- `tests/services/`
  - container-level expectations, service behavior, compose assumptions

- `tests/scripts/`
  - ingestion logic and chunking behavior

### Regression coverage

The orchestrator regression matrix includes cases such as:

- single-turn factual chat
- follow-up questions that depend on earlier turns
- sparse-sensitive keyword or entity-heavy retrieval
- no-context responses
- web-fallback behavior
- citation-sensitive multiline answers where streaming must preserve inline markdown

---

## 15. Useful Commands

### Development Make targets

```bash
make test
make test-integration
make dev-build
make dev-build-up
make dev-up
make dev-down
make dev-restart
make dev-logs
make dev-ps
make dev-audio-up
make dev-ingest
```

### Image publishing Make targets

```bash
make images-build
make images-push
make images-build-push
```

---

## 16. Troubleshooting

### Model services never become healthy

Common causes:

- `HF_TOKEN` is missing or invalid
- the selected model is too large for available GPU memory
- NVIDIA Container Toolkit is not configured correctly
- Windows GPU support through WSL2 is not enabled

### macOS cannot run the full stack

That is expected for the current GPU-oriented compose setup. Use macOS for development and tests, or point the orchestrator to remote Linux GPU services.

### The orchestrator starts but answers fail

Check:

- `LLM__URL`
- `LLM__MODEL`
- whether the Ollama model was pulled in the dev stack
- whether Qdrant contains data in `SERVICES__QDRANT_COLLECTION`

### Re-ingestion fails because of chunking or version mismatch

That usually means the ingestion pipeline is protecting you from mixing incompatible chunk layouts. Either:

- ingest into a new collection name, or
- rerun with `INGEST__FORCE_RECREATE=true` if you explicitly want to rebuild the collection

---

## 17. Notes For Contributors

- Prefer Docker-based runtime testing for end-to-end behavior.
- Use `uv run pytest` as the baseline local verification command.
- Keep README examples aligned with the active compose files and Make targets.
- When changing the retrieval pipeline, also review prompt contracts, web fallback logic, and citation behavior together.

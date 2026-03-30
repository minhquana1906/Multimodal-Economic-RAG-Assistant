# Multimodal Economic RAG Assistant

OpenAI-compatible multimodal RAG stack for Vietnamese economic news. The repository contains:

- an orchestration API in [`api/`](api),
- retrieval and safety services in [`services/`](services),
- a data ingestion pipeline in [`scripts/`](scripts),
- Docker Compose manifests for local development and image-based deployment.

The practical way to run this project is Docker-first. Local Python dependencies are still useful for tests, quick API work, and ingestion debugging.

## 1. Run Modes

This repo currently has 2 main Docker entrypoints:

- `docker-compose.dev.yaml`: local development stack that builds images from the source tree.
- `docker-compose.yaml`: image-based deployment stack that pulls or uses prebuilt images from Docker Hub and expects an NVIDIA GPU host.

There is also `docker-compose.yml.legacy`, which is only an archived older compose file.

## 2. Platform Support

### Linux (Ubuntu/Debian)

This is the primary supported platform for the full stack.

- Recommended for full local runtime.
- Required if you want to run the GPU-backed inference services directly on the same machine.
- Install Docker Engine, Docker Compose plugin, and NVIDIA Container Toolkit if you have an NVIDIA GPU.

### Windows

Use Windows 11 + Docker Desktop + WSL2.

- Recommended shell: PowerShell for setup, WSL2 Ubuntu for daily development commands.
- Full GPU-backed Docker runtime requires an NVIDIA GPU with WSL2 GPU support.
- If you do not have NVIDIA GPU passthrough available, you can still run tests locally and connect the API to remote services.

### macOS

macOS is fine for source code work, unit tests, and documentation updates, but not for the full GPU-backed inference stack in this repo.

- Docker Desktop works for basic containers and tooling.
- The current compose files expect NVIDIA GPU-backed services for embedding, reranking, guard, ASR, TTS, and model serving.
- On macOS, the realistic workflow is:
  - run unit tests locally,
  - optionally run only lightweight services,
  - or point the orchestrator to remote Linux GPU services.

## 3. Prerequisites

Install these first:

- `git`
- Docker with `docker compose`
- `uv`
- Python 3.12 for local non-Docker workflows

Useful official install pages:

- Docker Desktop: <https://docs.docker.com/desktop/>
- Docker Engine (Linux): <https://docs.docker.com/engine/install/>
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

## 4. Clone And Configure Environment

Clone the repository and create `.env` from the template:

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

Minimum variables you should review in `.env`:

- `HF_TOKEN`: required for model downloads from Hugging Face.
- `LLM__URL`: default in `.env.example` points to `ollama`, which matches `docker-compose.dev.yaml`.
- `LLM__MODEL`: default is `qwen3:0.6b` for the local Ollama-based dev stack.
- `SERVICES__QDRANT_COLLECTION`: default collection is `econ_vn_news`.
- `OBSERVABILITY__LANGSMITH_API_KEY`: optional.
- `OBSERVABILITY__TAVILY_API_KEY`: optional.
- `CLOUDFLARE_TUNNEL_TOKEN`: optional unless you use the `tunnel` service.

### Important: `docker-compose.yaml` needs extra variables

If you run the image-based compose file (`docker-compose.yaml`) instead of the dev compose file, update `.env` accordingly because it expects a vLLM-style OpenAI-compatible endpoint:

```env
LLM__URL=http://llm:8004/v1
LLM__MODEL=Qwen/Qwen3-4B-Instruct-2507
VLLM_LLM_GPU_MEMORY_UTILIZATION=0.92
VLLM_LLM_MAX_MODEL_LEN=8192
DOCKERHUB_NAMESPACE=your-dockerhub-namespace
IMAGE_TAG=latest
```

If you do not add the `VLLM_*` variables, `docker-compose.yaml` will not start cleanly.

## 5. Install Local Python Dependencies

Use this when you want to run tests, inspect code, or work on the API without starting the whole Docker stack.

From the repository root:

```bash
uv sync --dev
```

This installs the root project dependencies declared in [`pyproject.toml`](pyproject.toml).

### Service-specific dependencies

Some services also keep their own `requirements.txt` because they run in isolated Docker images:

- [`api/requirements.txt`](api/requirements.txt)
- [`scripts/requirements.txt`](scripts/requirements.txt)
- [`services/asr/requirements.txt`](services/asr/requirements.txt)
- [`services/embedding/requirements.txt`](services/embedding/requirements.txt)
- [`services/reranker/requirements.txt`](services/reranker/requirements.txt)
- [`services/guard/requirements.txt`](services/guard/requirements.txt)
- [`services/tts/requirements.txt`](services/tts/requirements.txt)

If you want to run one of those components outside Docker, install that folder's requirements explicitly in that folder.

Example for the orchestrator API:

```bash
cd api
uv venv
uv pip install -r requirements.txt
uv run uvicorn orchestrator.main:app --reload --host 0.0.0.0 --port 8000
```

Example for the ingestion script:

```bash
cd scripts
uv venv
uv pip install -r requirements.txt
uv run ingest.py
```

## 6. Run The Local Development Stack

Use `docker-compose.dev.yaml` when you want Docker to build from the checked-out source code.

### Start the main stack

```bash
docker compose -f docker-compose.dev.yaml up -d --build
```

Equivalent Make targets:

```bash
make dev-build-up
make dev-ps
make dev-logs
```

Main services and ports:

- Orchestrator API: `http://localhost:8000`
- OpenWebUI: `http://localhost:8080`
- Qdrant HTTP: `http://localhost:6333`
- Qdrant gRPC: `localhost:6334`
- Embedding service: `http://localhost:8001`
- Reranker service: `http://localhost:8002`
- Guard service: `http://localhost:8003`
- Ollama service: `http://localhost:11434`

### Pull the Ollama model after first startup

The dev stack uses Ollama as the LLM backend. After the container is up, pull the model declared in `.env`.

```bash
docker compose -f docker-compose.dev.yaml exec ollama ollama pull qwen3:0.6b
```

If you changed `LLM__MODEL`, pull that model instead.

### Health checks

```bash
curl http://localhost:8000/health
curl http://localhost:6333/collections
curl http://localhost:11434/api/tags
```

### Stop the dev stack

```bash
docker compose -f docker-compose.dev.yaml down
```

Or:

```bash
make dev-down
```

## 7. Run Optional Audio Services

ASR and TTS are behind the `audio` profile.

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

## 8. Run The Ingestion Pipeline

The ingestion pipeline:

- loads the `khoalnd/EconVNNews` dataset,
- chunks articles,
- generates dense and sparse vectors,
- upserts them into Qdrant.

### Run ingestion in Docker

```bash
docker compose -f docker-compose.dev.yaml --profile ingest up ingest
```

Or:

```bash
make dev-ingest
```

### Important ingestion settings

From `.env`:

- `SERVICES__QDRANT_URL=http://qdrant:6333`
- `SERVICES__QDRANT_COLLECTION=econ_vn_news`
- `INGEST__BATCH_SIZE=256`
- `INGEST__FORCE_RECREATE=false`

### When to use `INGEST__FORCE_RECREATE=true`

Set `INGEST__FORCE_RECREATE=true` when you intentionally want to destroy and rebuild the target Qdrant collection, for example after a chunking schema change.

Example:

```bash
INGEST__FORCE_RECREATE=true docker compose -f docker-compose.dev.yaml --profile ingest up ingest
```

This will recreate the Qdrant collection before re-ingesting. Do not use it casually on valuable data.

### Verify ingestion result

Check the collection list:

```bash
curl http://localhost:6333/collections
```

Inspect collection metadata:

```bash
curl http://localhost:6333/collections/econ_vn_news
```

## 9. Run The Image-Based Deployment Stack

Use `docker-compose.yaml` when images already exist in Docker Hub or were built and tagged in advance.

This mode is intended for a Linux NVIDIA GPU host.

### Build and push images first

If you are publishing your own images:

```bash
make images-build
make images-push
```

Or in one step:

```bash
make images-build-push
```

### Start the image-based stack

Use direct `docker compose` commands:

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

### Why direct `docker compose` is recommended here

The current production-like compose file is `docker-compose.yaml`. The repo also keeps `docker-compose.yml.legacy` as an archived older file, so using explicit compose commands is clearer than depending on older helper targets.

## 10. Qdrant Snapshot Workflow

The local Qdrant service in this repo exposes its HTTP API on `http://localhost:6333`.

The commands below assume:

- collection name: `econ_vn_news`
- no Qdrant API key is configured

If you enable an API key in your own deployment, add `-H "api-key: <your-key>"` to the `curl` requests below.

### Create a snapshot

```bash
curl -X POST "http://localhost:6333/collections/econ_vn_news/snapshots"
```

### List available snapshots

```bash
curl "http://localhost:6333/collections/econ_vn_news/snapshots"
```

### Download a snapshot

Replace `<snapshot-name>` with the value returned by the list or create call.

```bash
curl -L "http://localhost:6333/collections/econ_vn_news/snapshots/<snapshot-name>" -o "./<snapshot-name>"
```

### Upload and recover from a local snapshot file

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

This overwrites the current collection contents with the uploaded snapshot. If the collection does not exist, Qdrant creates it during recovery.

### Recover from a snapshot URL or file URI

If the snapshot file is already reachable by URL, or mounted where Qdrant can read it, use the recover endpoint:

```bash
curl -X PUT "http://localhost:6333/collections/econ_vn_news/snapshots/recover?wait=true" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "file:///qdrant/snapshots/econ_vn_news.snapshot",
    "priority": "snapshot"
  }'
```

You can also use an HTTP URL:

```bash
curl -X PUT "http://localhost:6333/collections/econ_vn_news/snapshots/recover?wait=true" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "https://example.com/econ_vn_news.snapshot",
    "priority": "snapshot"
  }'
```

### Snapshot safety notes

- Recovery overwrites collection data.
- Restore into the correct collection name.
- If you are testing a new dataset version, use a separate collection name first.
- If the snapshot was created from a different chunking strategy, verify compatibility before mixing old and new ingestion runs.

## 11. Test Workflow

The canonical local unit-test entrypoint for this repository is:

```bash
uv run pytest
```

Matching shortcuts:

```bash
make test
make test-integration
```

`integration` tests are opt-in and skipped by default in local runs:

```bash
uv run pytest -m integration
```

## 12. What The Regression Matrix Covers

`tests/orchestrator/test_regression_matrix.py` defines the baseline regression matrix for:

- single-turn factual chat,
- follow-up questions that depend on earlier turns,
- sparse-sensitive keyword and entity-heavy retrieval cases,
- no-context responses,
- web-fallback queries,
- citation-sensitive multiline answers where streaming must preserve inline markdown links.

## 13. Troubleshooting

### Docker starts but model services never become healthy

Most likely causes:

- `HF_TOKEN` is missing or invalid.
- the model is too large for your GPU memory budget,
- NVIDIA Container Toolkit is not installed correctly on Linux,
- Docker Desktop on Windows does not have WSL2 GPU support enabled.

### macOS cannot run the full stack

That is expected with the current compose files because they assume NVIDIA GPU containers. Use a remote Linux GPU host for the full stack.

### Orchestrator is healthy but responses fail

Check:

- `LLM__URL`
- `LLM__MODEL`
- whether the Ollama model was pulled in the dev stack
- whether Qdrant contains data in `SERVICES__QDRANT_COLLECTION`

### Re-ingestion fails with chunking/version mismatch

That is usually intentional protection from mixing old and new chunking layouts. Either:

- ingest into a new collection name, or
- rerun with `INGEST__FORCE_RECREATE=true` if you explicitly want to rebuild the collection.

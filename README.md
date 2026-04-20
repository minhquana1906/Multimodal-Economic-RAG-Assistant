# Multimodal Economic RAG Assistant

An OpenAI-compatible RAG system for Vietnamese economic and financial queries.

The stack combines a FastAPI orchestration layer, a consolidated inference service (embedding + reranking), a vector database, an LLM backend, and an Open WebUI frontend.

---

## 1. What This Project Does

The system answers economic and financial questions in Vietnamese using retrieval-augmented generation.

At a high level:

1. The user sends a chat request to the orchestrator.
2. LangGraph routes the query (RAG path or direct LLM path).
3. Hybrid retrieval runs against the Qdrant collection.
4. Retrieved passages are reranked.
5. When internal corpus support is weak or the question is time-sensitive, web search fallback runs automatically. **Web search fallback is enabled by default because the internal corpus is still sparse.**
6. The LLM generates a grounded answer with inline citations.

LangGraph is retained in a slim form to handle intent routing and the RAG workflow graph.

---

## 2. Design Notes

- **Two shared Dockerfiles** cover all custom services: `infra/docker/app.cpu.Dockerfile` (orchestrator, ingest) and `infra/docker/app.gpu.Dockerfile` (inference).
- **No per-service Dockerfiles.** All service-specific Dockerfiles have been removed.
- **Dev refresh uses bind mounts.** Code edits under `api/`, `services/inference/`, or `scripts/` take effect on container restart without rebuilding the image.
- **Three local models** are used by the inference service:
  - `BAAI/bge-m3` — dense embedding
  - `BAAI/bge-reranker-v2-m3` — passage reranking
  - `Qwen/Qwen3.5-4B` — LLM generation (served via vLLM)

---

## 3. Repository Structure

```text
.
├── Makefile
├── README.md
├── api/
│   └── orchestrator/
│       ├── config.py
│       ├── main.py
│       ├── pipeline/        # LangGraph workflow, RAG logic, citations
│       ├── routers/         # chat endpoint
│       └── services/        # inference client, retriever, web search, llm
├── infra/
│   └── docker/
│       ├── app.cpu.Dockerfile
│       └── app.gpu.Dockerfile
├── scripts/
│   ├── ingest.py
│   └── chunker.py
├── services/
│   └── inference/
│       ├── inference_app.py  # embedding + reranking in one service
│       └── requirements.txt
├── tests/
│   ├── orchestrator/
│   ├── scripts/
│   └── services/
├── docker-compose.dev.yaml
├── docker-compose.yaml
└── pyproject.toml
```

---

## 4. Runtime Services

| Service | Port | Notes |
|---|---|---|
| `orchestrator` | 8000 | FastAPI + LangGraph |
| `inference` | 8001 | Embedding + reranking (GPU) |
| `llm` | 8004 | vLLM OpenAI-compatible endpoint (GPU) |
| `qdrant` | 6333/6334 | Vector database |
| `webui` | 8080 | Open WebUI |
| `ingest` | — | Profile `ingest`, runs once |

---

## 5. Prerequisites

- `git`
- Docker with the `docker compose` plugin
- NVIDIA Container Toolkit (for GPU services)
- `uv`
- Python 3.12 (for local non-Docker workflows)

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 6. Clone and Configure

```bash
git clone https://github.com/minhquana1906/Multimodal-Economic-RAG-Assistant.git
cd Multimodal-Economic-RAG-Assistant
cp .env.example .env
```

Edit `.env` and set at minimum:

- `HF_TOKEN` — required for Hugging Face model downloads
- `LLM__MODEL` — default `Qwen/Qwen3.5-4B`
- `LLM__URL` — set to `http://llm:8004/v1` for the compose stack
- `SERVICES__QDRANT_COLLECTION` — default `econ_vn_news`
- `OBSERVABILITY__TAVILY_API_KEY` — optional, enables web search fallback
- `CLOUDFLARE_TUNNEL_TOKEN` — optional, only for the tunnel service

For production compose, also set:

```env
DOCKERHUB_NAMESPACE=your-dockerhub-namespace
VLLM_LLM_GPU_MEMORY_UTILIZATION=0.92
VLLM_LLM_MAX_MODEL_LEN=8192
```

---

## 7. Local Python Environment

```bash
uv sync --dev
uv run pytest
```

---

## 8. Development Stack

The dev compose builds `inference` and `orchestrator` from the shared Dockerfiles and mounts local source directories as bind mounts. Code changes under `api/` or `services/inference/` are picked up on container restart without rebuilding.

Start the dev stack:

```bash
docker compose -f docker-compose.dev.yaml up -d --build
```

Or using Make:

```bash
make dev-build-up
```

After the stack is up, verify:

```bash
curl http://localhost:8000/health
curl http://localhost:6333/collections
```

Stop:

```bash
make dev-down
```

Run ingestion:

```bash
make dev-ingest
```

---

## 9. Production Stack

The production compose uses pre-built images from Docker Hub and no bind mounts.

Build and push images:

```bash
make images-build
make images-push
```

Start:

```bash
docker compose -f docker-compose.yaml up -d
```

Run ingestion:

```bash
make ingest
```

---

## 10. Testing

```bash
uv run pytest
uv run pytest -m integration   # integration tests only
```

The orchestrator regression matrix covers single-turn factual queries, follow-up conversations, sparse-sensitive keyword queries, no-context responses, web fallback, and streaming inline citations.

---

## 11. Troubleshooting

**Model services never become healthy**
- Check `HF_TOKEN` is set and valid.
- Verify GPU memory is sufficient for the selected models.
- Confirm NVIDIA Container Toolkit is installed and configured.

**Orchestrator starts but answers fail**
- Check `LLM__URL` and `LLM__MODEL` in `.env`.
- Verify Qdrant has data in `SERVICES__QDRANT_COLLECTION`.

**Re-ingestion fails**
- Use `INGEST__FORCE_RECREATE=true` to rebuild the collection from scratch (destructive).

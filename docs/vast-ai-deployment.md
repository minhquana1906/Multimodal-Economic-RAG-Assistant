# Vast.ai Deployment

This repository now has two Compose entrypoints:

- `docker-compose.dev.yaml` for local development with bind mounts
- `docker-compose.vast.yaml` for prebuilt images on a remote 2x RTX 3090 Vast.ai VM

## Pinned infra versions

The stack now pins the moving third-party images through env-backed defaults so stream/render diagnostics happen against a reproducible target:

- Open WebUI: `ghcr.io/open-webui/open-webui:${OPEN_WEBUI_IMAGE_TAG:-v0.7.2}`
- Qdrant: `qdrant/qdrant:${QDRANT_IMAGE_TAG:-v1.17.0}`
- Cloudflared: `cloudflare/cloudflared:${CLOUDFLARED_IMAGE_TAG:-2026.3.0}`

Override them only when you are intentionally testing an upgrade.

## GPU layout

- GPU 0: `embedding`, `reranker`, `guard`, `asr`, `tts`
- GPU 1: `llm`, `vlm`

`llm` and `vlm` both use the official `vllm/vllm-openai` image and apply soft VRAM caps through:

- `VLLM_LLM_GPU_MEMORY_UTILIZATION`
- `VLLM_VLM_GPU_MEMORY_UTILIZATION`

The defaults are `0.45` and `0.45` to leave headroom for CUDA/runtime overhead. If `Qwen/Qwen3-VL-4B-Instruct` needs more room on your instance, raise the VLM value and lower the LLM value in `.env.vast`.

## Local build and push

1. Log in to Docker Hub:

```bash
docker login
```

2. Prepare environment values:

```bash
cp .env.vast.example .env.vast
export DOCKERHUB_NAMESPACE=your-dockerhub-namespace
export IMAGE_TAG=vast-2026-03-24
```

3. Build and push the application images:

```bash
make images-build-push DOCKERHUB_NAMESPACE=$DOCKERHUB_NAMESPACE IMAGE_TAG=$IMAGE_TAG
```

This pushes:

- `mera-embedding`
- `mera-reranker`
- `mera-guard`
- `mera-asr`
- `mera-tts`
- `mera-orchestrator`
- `mera-ingest`

## Remote deploy on Vast.ai

1. Copy or create `.env.vast` on the VM.
2. Pull images:

```bash
make vast-pull
```

3. Start the core stack:

```bash
make vast-up
```

4. Optional profiles:

```bash
docker compose --env-file .env.vast -f docker-compose.vast.yaml --profile audio up -d
docker compose --env-file .env.vast -f docker-compose.vast.yaml --profile ingest up ingest
```

## Notes

- `orchestrator` is pointed at `http://llm:8004/v1` by default in the Vast.ai stack.
- `vlm` is deployed and exposed on `:8007`, but the current backend code does not call it yet.
- The vLLM docs recommend `ipc: host` for shared memory when running the official Docker image: https://docs.vllm.ai/en/stable/deployment/docker.html

## SSE and proxy compatibility

For streaming markdown validation, treat the backend as canonical and only debug OpenWebUI or proxy behavior after you have proven backend parity.

- Backend expectation:
  - `stream=false` and concatenated `stream=true` deltas must produce the exact same Python `str`.
  - The backend answer already contains the final inline markdown citations such as `[title](url)`.
  - The transport must not append a trailing `Nguồn:` block.
- Reverse proxy guidance:
  - Disable proxy buffering for SSE.
  - Disable response compression on the SSE route while diagnosing formatting issues.
  - Preserve `Cache-Control: no-cache` and `X-Accel-Buffering: no` from the backend response.
- OpenWebUI guidance:
  - Validate against the pinned Open WebUI image first.
  - Only inspect chunk batching or renderer behavior after capturing raw SSE from the backend.

## Multiline markdown verification procedure

Use the same prompt twice, once with `stream=false` and once with `stream=true`, and compare the answers after concatenating all `choices[0].delta.content` chunks.

Recommended prompt shape:

```text
Trả lời bằng 2 đoạn ngắn, rồi tạo 2 bullet points in đậm một phần, và trích dẫn inline nguồn tương ứng.
```

Checklist:

1. Confirm the non-streaming response contains the final inline markdown links.
2. Confirm the streaming response renders the same paragraphs, blank lines, bullets, and bold markers after reconstruction.
3. If the raw SSE payload is correct but OpenWebUI still renders incorrectly, inspect OpenWebUI stream batching and any proxy buffering in front of it.

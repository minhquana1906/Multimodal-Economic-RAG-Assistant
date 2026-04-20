FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/app/.venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.12 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python

WORKDIR /app

COPY pyproject.toml ./
COPY uv.lock ./
COPY services ./services

# Install shared runtime deps plus CUDA-compatible inference extras once.
RUN pip install --break-system-packages uv \
    && uv sync --frozen --no-dev \
    && uv pip install --python .venv/bin/python \
        torch==2.10.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 \
    # Keep the default PyPI index for non-PyTorch packages.
    && uv pip install --python .venv/bin/python \
        fastapi \
        uvicorn \
        loguru \
        'transformers<5' \
        FlagEmbedding

# Compose can provide the concrete inference service command.

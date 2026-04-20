FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.12 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python

WORKDIR /app

COPY services ./services

# Install CUDA-compatible torch once. Inference-service deps should be installed
# in follow-up tasks once the unified inference service is introduced.
RUN pip install --break-system-packages \
    torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

CMD ["python3", "-m", "services.inference.inference_app"]


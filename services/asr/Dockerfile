FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3-pip libpython3.12t64 ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.12 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --break-system-packages \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0+cu128 torchaudio==2.10.0+cu128 torchcodec==0.10.0 \
    && pip install --break-system-packages \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r requirements.txt

COPY asr_app.py .

EXPOSE 8005

CMD ["uvicorn", "asr_app:app", "--host", "0.0.0.0", "--port", "8005"]

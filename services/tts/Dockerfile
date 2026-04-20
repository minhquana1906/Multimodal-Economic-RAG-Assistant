FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/root/.cargo/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip \
    ffmpeg libsndfile1 build-essential cmake pkg-config curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.12 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable

WORKDIR /app

COPY requirements.txt .
RUN pip install --break-system-packages --upgrade pip \
    && pip install --break-system-packages torch==2.10.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --break-system-packages -r requirements.txt

COPY tts_app.py text_preprocessor.py abbreviations.py ./

EXPOSE 8006

CMD ["uvicorn", "tts_app:app", "--host", "0.0.0.0", "--port", "8006"]

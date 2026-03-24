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

COPY requirements.txt .
RUN pip install --break-system-packages \
    torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --break-system-packages -r requirements.txt

COPY embedding_app.py .

EXPOSE 8001

CMD ["uvicorn", "embedding_app:app", "--host", "0.0.0.0", "--port", "8001"]

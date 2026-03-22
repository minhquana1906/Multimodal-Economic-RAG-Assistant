FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.12 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --break-system-packages --no-cache-dir \
    torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --break-system-packages --no-cache-dir -r requirements.txt

COPY reranker_app.py .

EXPOSE 8002

CMD ["uvicorn", "reranker_app:app", "--host", "0.0.0.0", "--port", "8002"]

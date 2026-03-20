FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.12 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python \
    && python3.12 -m ensurepip --upgrade \
    && python3.12 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]

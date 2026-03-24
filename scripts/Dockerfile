FROM astral/uv:python3.12-trixie-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --no-cache --system -r requirements.txt

COPY chunker.py ingest.py /app/

CMD ["uv", "run", "ingest.py"]

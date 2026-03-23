FROM astral/uv:python3.12-trixie-slim

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --no-cache --system -r requirements.txt

COPY . /app

RUN ls -la /app

CMD ["uv", "run", "ingest.py"]

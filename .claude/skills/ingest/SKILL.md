---
name: ingest
description: Run the data ingestion script to load documents into the Qdrant vector store
disable-model-invocation: true
---

Before running, confirm:
1. Qdrant is healthy (run /stack-status first)
2. Embedding service is healthy
3. QDRANT_COLLECTION is set correctly in .env

Then run:
```bash
docker compose --profile ingest run --rm ingest
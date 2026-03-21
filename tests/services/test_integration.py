# tests/services/test_integration.py
# Integration smoke test — verifies all three GPU services are accessible and healthy.
#
# Run *after* `docker compose up -d embedding reranker guard`:
#   pytest tests/services/test_integration.py -v
#
# URL defaults match the Docker Compose network hostnames and ports.
# Override with env vars when running outside Docker:
#   EMBEDDING_URL=http://localhost:8001 \
#   RERANKER_URL=http://localhost:8002 \
#   GUARD_URL=http://localhost:8003 \
#   pytest tests/services/test_integration.py -v
import os

import httpx
import pytest


EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://embedding:8001")
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8002")
GUARD_URL = os.getenv("GUARD_URL", "http://guard:8003")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_all_services_healthy():
    """All three services must respond with HTTP 200 and status='ok'."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Embedding service
        response = await client.get(f"{EMBEDDING_URL}/health")
        assert response.status_code == 200, f"Embedding health failed: {response.text}"
        assert response.json()["status"] == "ok"

        # Reranker service
        response = await client.get(f"{RERANKER_URL}/health")
        assert response.status_code == 200, f"Reranker health failed: {response.text}"
        assert response.json()["status"] == "ok"

        # Guard service
        response = await client.get(f"{GUARD_URL}/health")
        assert response.status_code == 200, f"Guard health failed: {response.text}"
        assert response.json()["status"] == "ok"

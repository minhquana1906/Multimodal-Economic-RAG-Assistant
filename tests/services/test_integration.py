# tests/services/test_integration.py
# Integration smoke test — verifies live services are accessible and healthy.
#
# Run *after* `docker compose up -d`:
#   pytest tests/services/test_integration.py -v
#
# URL defaults match the Docker Compose network hostnames and ports.
import os

import httpx
import pytest


INFERENCE_URL = os.getenv("SERVICES__INFERENCE_URL", "http://inference:8000")
QDRANT_URL = os.getenv("SERVICES__QDRANT_URL", "http://qdrant:6333")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8080")
LLM_URL = os.getenv("LLM__URL", "http://llm:8000")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_all_services_healthy():
    """Core services must respond with HTTP 200 and status='ok'."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Inference service
        response = await client.get(f"{INFERENCE_URL}/health")
        assert response.status_code == 200, f"Inference health failed: {response.text}"
        assert response.json()["status"] == "ok"

        # Qdrant
        response = await client.get(f"{QDRANT_URL}/healthz")
        assert response.status_code == 200, f"Qdrant health failed: {response.text}"

        # Orchestrator
        response = await client.get(f"{ORCHESTRATOR_URL}/health")
        assert response.status_code == 200, f"Orchestrator health failed: {response.text}"

        # LLM
        response = await client.get(f"{LLM_URL}/health")
        assert response.status_code == 200, f"LLM health failed: {response.text}"

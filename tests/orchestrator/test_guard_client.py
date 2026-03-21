import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_guard_check_input_fail_closed():
    """Returns False (fail-closed) when service is unreachable."""
    from orchestrator.services.guard import GuardClient
    guard = GuardClient("http://localhost:9999", timeout=0.1)
    result = await guard.check_input("test query")
    assert result is False


@pytest.mark.asyncio
async def test_guard_check_input_returns_true_for_safe():
    """Returns True when service responds with label='safe'."""
    from orchestrator.services.guard import GuardClient
    guard = GuardClient("http://guard:8003", timeout=10.0)

    mock_response = MagicMock()
    mock_response.json.return_value = {"label": "safe"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_async_cm = MagicMock()
    mock_async_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_cm):
        result = await guard.check_input("What is GDP?")

    assert result is True

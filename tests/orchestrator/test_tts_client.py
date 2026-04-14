import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_tts_client_synthesize_returns_bytes():
    """TTSClient.synthesize returns WAV bytes on success."""
    from orchestrator.services.tts import TTSClient

    tts = TTSClient(url="http://tts:8006", timeout=60.0)

    mock_response = MagicMock()
    mock_response.content = b"RIFF" + (b"\x00" * 40)
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_async_cm = MagicMock()
    mock_async_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_cm):
        result = await tts.synthesize("Xin chào Việt Nam", speed=1.0)

    assert result is not None
    assert isinstance(result, bytes)
    assert result[:4] == b"RIFF"
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert "/synthesize" in str(call_args)


@pytest.mark.asyncio
async def test_tts_client_returns_none_on_error():
    """TTSClient.synthesize returns None when service is down."""
    from orchestrator.services.tts import TTSClient

    tts = TTSClient(url="http://localhost:9999", timeout=0.1)
    result = await tts.synthesize("Xin chào", speed=1.0)
    assert result is None


@pytest.mark.asyncio
async def test_tts_client_unload():
    """TTSClient.unload sends POST /unload."""
    from orchestrator.services.tts import TTSClient

    tts = TTSClient(url="http://tts:8006", timeout=60.0)

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "unloaded"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_async_cm = MagicMock()
    mock_async_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_cm):
        await tts.unload()

    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert "/unload" in str(call_args)

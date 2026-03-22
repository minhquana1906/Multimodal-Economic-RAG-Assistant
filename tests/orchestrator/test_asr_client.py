import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_asr_client_transcribe_returns_text():
    """ASRClient.transcribe returns transcribed text on success."""
    from orchestrator.services.asr import ASRClient

    asr = ASRClient(url="http://asr:8005", timeout=30.0)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "text": "GDP Việt Nam năm 2023",
        "language": "vi",
        "duration_seconds": 3.2,
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_async_cm = MagicMock()
    mock_async_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_cm):
        result = await asr.transcribe(b"fake-audio-bytes", language="vi")

    assert result == "GDP Việt Nam năm 2023"
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert "files" in call_kwargs.kwargs or "files" in (call_kwargs[1] if len(call_kwargs) > 1 else {})


@pytest.mark.asyncio
async def test_asr_client_raises_on_service_error():
    """ASRClient.transcribe raises ASRError when service returns error."""
    from orchestrator.services.asr import ASRClient, ASRError

    asr = ASRClient(url="http://localhost:9999", timeout=0.1)
    with pytest.raises(ASRError):
        await asr.transcribe(b"fake-audio-bytes", language="vi")


@pytest.mark.asyncio
async def test_asr_client_unload():
    """ASRClient.unload sends POST /unload."""
    from orchestrator.services.asr import ASRClient

    asr = ASRClient(url="http://asr:8005", timeout=30.0)

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "unloaded"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_async_cm = MagicMock()
    mock_async_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_cm):
        await asr.unload()

    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert "/unload" in str(call_args)

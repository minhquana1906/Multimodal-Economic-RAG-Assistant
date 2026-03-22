import asyncio
import importlib
import io

import pytest
import torch
import torchaudio
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_qwen_asr_model():
    """Mock Qwen3ASRModel to avoid loading real model in tests."""
    mock_instance = MagicMock()
    mock_instance.transcribe = MagicMock()
    return mock_instance


@pytest.fixture
async def client(mock_qwen_asr_model):
    """ASGI test client with mocked model (model NOT loaded by default)."""
    with patch.dict("os.environ", {"ASR_MODEL": "Qwen/Qwen3-ASR-1.7B"}):
        import asr_app
        importlib.reload(asr_app)
        transport = ASGITransport(app=asr_app.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_health_returns_idle_when_model_not_loaded(client):
    """Health returns 200 with status='idle' when model is not loaded."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "idle"
    assert data["model_loaded"] is False


async def test_health_returns_ok_when_model_loaded(mock_qwen_asr_model):
    """Health returns 200 with status='ok' when model is loaded."""
    import asr_app
    importlib.reload(asr_app)
    asr_app.on_demand.model = mock_qwen_asr_model
    transport = ASGITransport(app=asr_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        response = await c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


async def test_health_returns_loading_during_model_load(mock_qwen_asr_model):
    """Health returns 200 with status='loading' when model is being loaded."""
    import asr_app
    importlib.reload(asr_app)
    asr_app.on_demand._loading = True
    transport = ASGITransport(app=asr_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        response = await c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "loading"
    assert data["model_loaded"] is False


@pytest.fixture
async def client_with_model(mock_qwen_asr_model):
    """ASGI test client with model pre-loaded."""
    with patch.dict("os.environ", {
        "ASR_MODEL": "Qwen/Qwen3-ASR-1.7B",
        "ASR_MAX_DURATION_S": "60",
    }):
        import asr_app
        importlib.reload(asr_app)
        asr_app.on_demand.model = mock_qwen_asr_model
        transport = ASGITransport(app=asr_app.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c, mock_qwen_asr_model


def _make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a valid WAV file with silence for testing."""
    import struct
    import wave

    num_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
    buf.seek(0)
    return buf.read()


async def test_transcribe_returns_text(client_with_model):
    """POST /transcribe returns transcribed text."""
    client, mock_model = client_with_model

    mock_result = MagicMock()
    mock_result.text = "GDP Việt Nam năm 2023"
    mock_result.language = "Vietnamese"
    mock_model.transcribe.return_value = [mock_result]

    wav_bytes = _make_wav_bytes(duration_s=2.0)

    with patch("asr_app.on_demand.get_model", new_callable=AsyncMock, return_value=mock_model):
        with patch("asr_app._decode_audio", return_value=(torch.zeros(1, 32000), 16000)):
            response = await client.post(
                "/transcribe",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"language": "vi"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "GDP Việt Nam năm 2023"
    assert "duration_seconds" in data


async def test_transcribe_rejects_too_long_audio(client_with_model):
    """POST /transcribe returns 413 for audio exceeding max duration."""
    client, mock_model = client_with_model

    with patch("asr_app.on_demand.get_model", new_callable=AsyncMock, return_value=mock_model):
        with patch("asr_app._decode_audio", return_value=(torch.zeros(1, 90 * 16000), 16000)):
            wav_bytes = _make_wav_bytes(duration_s=1.0)
            response = await client.post(
                "/transcribe",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"language": "vi"},
            )

    assert response.status_code == 413
    assert "duration" in response.json()["detail"].lower()


async def test_transcribe_with_no_file_returns_422(client_with_model):
    """POST /transcribe without file returns 422."""
    client, _ = client_with_model
    response = await client.post("/transcribe")
    assert response.status_code == 422


async def test_transcribe_rejects_unsupported_format(client_with_model):
    """POST /transcribe returns 415 for unsupported content type."""
    client, _ = client_with_model
    response = await client.post(
        "/transcribe",
        files={"file": ("test.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 415
    assert "supported" in response.json()["detail"].lower()


async def test_transcribe_returns_400_on_decode_failure(client_with_model):
    """POST /transcribe returns 400 when audio cannot be decoded."""
    client, mock_model = client_with_model
    with patch("asr_app.on_demand.get_model", new_callable=AsyncMock, return_value=mock_model):
        with patch("asr_app._decode_audio", side_effect=Exception("corrupt audio data")):
            response = await client.post(
                "/transcribe",
                files={"file": ("test.wav", b"corrupt", "audio/wav")},
            )
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


async def test_transcribe_returns_400_on_empty_result(client_with_model):
    """POST /transcribe returns 400 when model returns empty text."""
    client, mock_model = client_with_model
    mock_result = MagicMock()
    mock_result.text = "   "
    mock_model.transcribe.return_value = [mock_result]
    with patch("asr_app.on_demand.get_model", new_callable=AsyncMock, return_value=mock_model):
        with patch("asr_app._decode_audio", return_value=(torch.zeros(1, 16000), 16000)):
            response = await client.post(
                "/transcribe",
                files={"file": ("test.wav", b"fake", "audio/wav")},
            )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


async def test_transcribe_returns_500_on_inference_error(client_with_model):
    """POST /transcribe returns 500 when model.transcribe raises."""
    client, mock_model = client_with_model
    mock_model.transcribe.side_effect = RuntimeError("CUDA OOM")
    with patch("asr_app.on_demand.get_model", new_callable=AsyncMock, return_value=mock_model):
        with patch("asr_app._decode_audio", return_value=(torch.zeros(1, 16000), 16000)):
            response = await client.post(
                "/transcribe",
                files={"file": ("test.wav", b"fake", "audio/wav")},
            )
    assert response.status_code == 500
    assert "failed" in response.json()["detail"].lower()
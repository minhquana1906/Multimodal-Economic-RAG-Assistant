import importlib
import io
import struct
import sys
import wave

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch


# ── Mock soundfile before tts_app can import it ──────────────────────

def _fake_sf_write(buf: io.BytesIO, audio: np.ndarray, sample_rate: int, format: str = "WAV"):
    """Write a valid WAV file using stdlib wave module (replaces soundfile.write)."""
    samples = (audio * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


_mock_sf = MagicMock()
_mock_sf.write = _fake_sf_write
sys.modules.setdefault("soundfile", _mock_sf)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_vieneu_model():
    """Mock VieNeu TTS model to avoid loading real model in tests."""
    mock_instance = MagicMock()
    # tts.infer(text=..., temperature=...) -> np.ndarray of float32
    mock_instance.infer = MagicMock(
        return_value=np.zeros(24000, dtype=np.float32),
    )
    mock_instance.close = MagicMock()
    mock_instance.list_preset_voices = MagicMock(return_value=[])
    mock_instance.get_preset_voice = MagicMock(return_value=None)
    return mock_instance


@pytest.fixture
async def client(mock_vieneu_model):
    """ASGI test client with mocked model (model NOT loaded by default)."""
    with patch.dict("os.environ", {"TTS_MODEL": "pnnbao-ump/VieNeu-TTS"}):
        import tts_app
        importlib.reload(tts_app)
        transport = ASGITransport(app=tts_app.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.fixture
async def client_with_model(mock_vieneu_model):
    """ASGI test client with model pre-loaded."""
    with patch.dict("os.environ", {"TTS_MODEL": "pnnbao-ump/VieNeu-TTS"}):
        import tts_app
        importlib.reload(tts_app)
        tts_app.on_demand.model = mock_vieneu_model
        transport = ASGITransport(app=tts_app.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c, mock_vieneu_model


# ── Health Endpoint Tests ─────────────────────────────────────────────


async def test_health_returns_idle_when_model_not_loaded(client):
    """Health returns 200 with status='idle' when model is not loaded."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "idle"
    assert data["model_loaded"] is False


async def test_health_returns_ok_when_model_loaded(mock_vieneu_model):
    """Health returns 200 with status='ok' when model is loaded."""
    import tts_app
    importlib.reload(tts_app)
    tts_app.on_demand.model = mock_vieneu_model
    transport = ASGITransport(app=tts_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        response = await c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# ── Synthesize Endpoint Tests ─────────────────────────────────────────


async def test_synthesize_returns_wav(client_with_model):
    """POST /synthesize with text returns audio/wav response."""
    client, mock_model = client_with_model

    with patch(
        "tts_app.on_demand.get_model",
        new_callable=AsyncMock,
        return_value=mock_model,
    ):
        response = await client.post(
            "/synthesize",
            json={"text": "Xin chào Việt Nam."},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    # WAV files start with RIFF header
    assert response.content[:4] == b"RIFF"


async def test_synthesize_rejects_empty_text(client_with_model):
    """POST /synthesize with empty text returns 400."""
    client, _ = client_with_model
    response = await client.post(
        "/synthesize",
        json={"text": "   "},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


async def test_synthesize_returns_500_on_inference_error(client_with_model):
    """POST /synthesize returns 500 when model inference fails."""
    client, mock_model = client_with_model
    mock_model.infer.side_effect = RuntimeError("CUDA OOM")

    with patch(
        "tts_app.on_demand.get_model",
        new_callable=AsyncMock,
        return_value=mock_model,
    ):
        response = await client.post(
            "/synthesize",
            json={"text": "Xin chào."},
        )

    assert response.status_code == 500
    assert "failed" in response.json()["detail"].lower()


# ── Unload Endpoint Tests ─────────────────────────────────────────────


async def test_unload_frees_model(mock_vieneu_model):
    """POST /unload unloads the model and health returns 'idle'."""
    import tts_app
    importlib.reload(tts_app)
    # Simulate loaded model
    tts_app.on_demand.model = mock_vieneu_model

    transport = ASGITransport(app=tts_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        # Confirm model is loaded
        resp = await c.get("/health")
        assert resp.json()["model_loaded"] is True

        # Unload
        with patch("torch.cuda.empty_cache"):
            resp = await c.post("/unload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"

        # Confirm model is gone
        resp = await c.get("/health")
        assert resp.json()["model_loaded"] is False


# ── Text Preprocessor Tests ───────────────────────────────────────────


def test_text_preprocessor_numbers():
    """normalize_numbers converts Vietnamese numbers correctly."""
    from text_preprocessor import normalize_numbers

    assert "một trăm hai mươi ba" in normalize_numbers("123")
    assert "phần trăm" in normalize_numbers("6.5%")
    assert "triệu" in normalize_numbers("1,000,000")


def test_text_preprocessor_abbreviations():
    """expand_abbreviations replaces GDP with Vietnamese equivalent."""
    from text_preprocessor import expand_abbreviations

    result = expand_abbreviations("GDP tăng trưởng tốt")
    assert "Tổng sản phẩm quốc nội" in result
    assert "GDP" not in result


def test_text_preprocessor_clean_markdown():
    """clean_for_speech strips markdown formatting."""
    from text_preprocessor import clean_for_speech

    result = clean_for_speech("## Tiêu đề\n**Nội dung** quan trọng [1]")
    # Should strip markdown header markers, bold markers, and citations
    assert "##" not in result
    assert "**" not in result
    assert "[1]" not in result
    assert "Tiêu đề" in result
    assert "Nội dung" in result


def test_text_preprocessor_split_sentences():
    """split_sentences divides text on Vietnamese sentence delimiters."""
    from text_preprocessor import split_sentences

    result = split_sentences("Câu một. Câu hai! Câu ba?")
    assert len(result) == 3
    assert result[0] == "Câu một."
    assert result[1] == "Câu hai!"
    assert result[2] == "Câu ba?"


def test_text_preprocessor_preprocess_pipeline():
    """preprocess chains all steps: clean → expand → normalize → split."""
    from text_preprocessor import preprocess

    result = preprocess("## GDP tăng 6.5% trong năm 2023.")
    # Should return list of sentences
    assert isinstance(result, list)
    assert len(result) >= 1
    # GDP should have been expanded
    full_text = " ".join(result)
    assert "GDP" not in full_text
    assert "Tổng sản phẩm quốc nội" in full_text
    # 6.5% should have been normalized
    assert "phần trăm" in full_text

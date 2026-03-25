from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from orchestrator.routers.audio import create_audio_router
from orchestrator.services.asr import ASRError


def _make_app() -> tuple[FastAPI, MagicMock, MagicMock]:
    mock_asr = MagicMock()
    mock_asr.transcribe = AsyncMock(return_value="GDP Việt Nam tăng trưởng ổn định")
    mock_asr.unload = AsyncMock()

    mock_tts = MagicMock()
    mock_tts.synthesize = AsyncMock(return_value=b"RIFF\x00\x00\x00\x00WAVEfmt ")
    mock_tts.unload = AsyncMock()

    app = FastAPI()
    app.include_router(create_audio_router(mock_asr, mock_tts))
    return app, mock_asr, mock_tts


@pytest.mark.asyncio
async def test_audio_transcriptions_returns_text_and_proxies_audio_to_asr():
    app, mock_asr, mock_tts = _make_app()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1", "language": "vi"},
            files={"file": ("question.webm", b"fake-audio", "audio/webm")},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "GDP Việt Nam tăng trưởng ổn định"}
    mock_tts.unload.assert_awaited_once()
    mock_asr.transcribe.assert_awaited_once_with(
        b"fake-audio",
        language="vi",
        content_type="audio/webm",
    )


@pytest.mark.asyncio
async def test_audio_transcriptions_preserves_asr_status_code():
    app, mock_asr, _ = _make_app()
    mock_asr.transcribe.side_effect = ASRError("Audio duration too long", status_code=413)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1", "language": "vi"},
            files={"file": ("question.wav", b"fake-audio", "audio/wav")},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "Audio duration too long"}


@pytest.mark.asyncio
async def test_audio_speech_returns_wav_bytes_from_tts():
    app, mock_asr, mock_tts = _make_app()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": "Xin chào Việt Nam",
                "voice": "alloy",
                "speed": 1.2,
            },
        )

    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")
    assert response.headers["content-type"] == "audio/wav"
    mock_asr.unload.assert_awaited_once()
    mock_tts.synthesize.assert_awaited_once_with("Xin chào Việt Nam", speed=1.2)


@pytest.mark.asyncio
async def test_audio_speech_returns_502_when_tts_is_unavailable():
    app, _, mock_tts = _make_app()
    mock_tts.synthesize.return_value = None

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": "Xin chào Việt Nam",
                "voice": "alloy",
            },
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "TTS synthesis failed"}

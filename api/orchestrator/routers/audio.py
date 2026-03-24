from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from orchestrator.services.asr import ASRClient, ASRError
from orchestrator.services.tts import TTSClient


class AudioSpeechRequest(BaseModel):
    model: str
    input: str = Field(min_length=1)
    voice: str
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.1, le=2.0)


def create_audio_router(asr: ASRClient, tts: TTSClient) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/audio/transcriptions")
    @router.post("/audio/transcriptions")
    async def audio_transcriptions(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: str = Form("vi"),
    ):
        del model  # OpenAI-compatible field accepted for UI compatibility.
        await tts.unload()

        audio_bytes = await file.read()
        content_type = file.content_type or "audio/wav"

        try:
            text = await asr.transcribe(
                audio_bytes,
                language=language,
                content_type=content_type,
            )
        except ASRError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        return {"text": text}

    @router.post("/v1/audio/speech")
    @router.post("/audio/speech")
    async def audio_speech(request: AudioSpeechRequest):
        del request.model  # OpenAI-compatible field accepted for UI compatibility.
        del request.voice
        del request.response_format

        await asr.unload()
        audio_bytes = await tts.synthesize(request.input, speed=request.speed)
        if audio_bytes is None:
            raise HTTPException(status_code=502, detail="TTS synthesis failed")

        return Response(content=audio_bytes, media_type="audio/wav")

    return router

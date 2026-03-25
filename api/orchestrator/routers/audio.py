from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, Field

from orchestrator.services.asr import ASRClient, ASRError
from orchestrator.services.tts import TTSClient


class AudioSpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str = Field(min_length=1)
    voice: str = "alloy"
    response_format: str = "wav"
    speed: float = Field(default=1.0, ge=0.1, le=2.0)


def create_audio_router(asr: ASRClient, tts: TTSClient) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/audio/transcriptions")
    @router.post("/audio/transcriptions")
    async def audio_transcriptions(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: str = Form("vi"),
    ):
        logger.info(f"ASR request: model={model} language={language} content_type={file.content_type}")
        await tts.unload()

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        content_type = file.content_type or "audio/wav"

        try:
            text = await asr.transcribe(
                audio_bytes,
                language=language,
                content_type=content_type,
            )
        except ASRError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        logger.info(f"ASR result: text_len={len(text)}")
        return {"text": text}

    @router.post("/v1/audio/speech")
    @router.post("/audio/speech")
    async def audio_speech(request: AudioSpeechRequest):
        logger.info(
            f"TTS request: input_len={len(request.input)} speed={request.speed} "
            f"format={request.response_format}"
        )
        await asr.unload()

        audio_bytes = await tts.synthesize(request.input, speed=request.speed)
        if audio_bytes is None:
            raise HTTPException(status_code=502, detail="TTS synthesis failed")

        logger.info(f"TTS response: {len(audio_bytes)} bytes")
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=\"speech.wav\"",
            },
        )

    return router

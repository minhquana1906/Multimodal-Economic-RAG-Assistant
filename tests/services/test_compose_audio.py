from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_all_compose_files_configure_openwebui_to_use_openai_compatible_audio():
    for compose_path in [
        ROOT / "docker-compose.yml",
        ROOT / "docker-compose.dev.yaml",
        ROOT / "docker-compose.vast.yaml",
    ]:
        content = compose_path.read_text(encoding="utf-8")

        assert "AUDIO_STT_ENGINE" in content
        assert "AUDIO_STT_OPENAI_API_BASE_URL" in content
        assert "AUDIO_STT_OPENAI_API_KEY" in content
        assert "AUDIO_TTS_ENGINE" in content
        assert "AUDIO_TTS_OPENAI_API_BASE_URL" in content
        assert "AUDIO_TTS_OPENAI_API_KEY" in content

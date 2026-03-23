from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_model_service_dockerfiles_use_runtime_images_only():
    for dockerfile in [
        "services/asr/Dockerfile",
        "services/embedding/Dockerfile",
        "services/guard/Dockerfile",
        "services/reranker/Dockerfile",
    ]:
        content = _read(dockerfile)
        assert "cudnn-runtime" in content
        assert "AS builder" not in content
        assert "cudnn-devel" not in content
        assert "flash-attn" not in content
        assert "COPY --from=builder /opt/venv /opt/venv" not in content


def test_tts_dockerfile_remains_runtime_only():
    content = _read("services/tts/Dockerfile")
    assert "cudnn-runtime" in content
    assert "AS builder" not in content
    assert "flash-attn" not in content

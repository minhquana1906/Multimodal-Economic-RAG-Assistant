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
        assert "python3.12-dev" not in content


def test_tts_dockerfile_remains_single_stage_runtime_build():
    content = _read("services/tts/Dockerfile")
    assert "cudnn-runtime" in content
    assert "AS builder" not in content
    assert "AS runtime" not in content
    assert "flash-attn" not in content
    assert "COPY --from=builder /opt/venv /opt/venv" not in content


def test_tts_dockerfile_installs_native_build_toolchain():
    content = _read("services/tts/Dockerfile")
    for package in ["build-essential", "cmake", "pkg-config", "curl", "ca-certificates"]:
        assert package in content
    assert "sh.rustup.rs" in content
    assert 'PATH="/root/.cargo/bin:$PATH"' in content


def test_tts_dockerfile_does_not_upgrade_apt_managed_pip():
    content = _read("services/tts/Dockerfile")
    assert "--upgrade pip" not in content


def test_audio_service_dockerfiles_preinstall_matching_cuda_torch_stack():
    for dockerfile in [
        "services/asr/Dockerfile",
        "services/tts/Dockerfile",
    ]:
        content = _read(dockerfile)
        assert "torch==2.10.0+cu128" in content
        assert "torchaudio==2.10.0+cu128" in content
        assert "https://download.pytorch.org/whl/cu128" in content


def test_audio_service_requirements_pin_torchaudio_to_torch_minor():
    for requirements in [
        "services/asr/requirements.txt",
        "services/tts/requirements.txt",
    ]:
        content = _read(requirements)
        assert "torchaudio==2.10.0" in content
        assert "torchaudio>=" not in content


def test_asr_dockerfile_installs_ffmpeg_for_torchcodec_decode():
    content = _read("services/asr/Dockerfile")
    assert "ffmpeg" in content


def test_asr_requirements_include_torchcodec():
    content = _read("services/asr/requirements.txt")
    assert "torchcodec" in content


def test_asr_dockerfile_raises_pip_timeout_for_large_cuda_wheels():
    content = _read("services/asr/Dockerfile")
    assert "PIP_DEFAULT_TIMEOUT=300" in content
    assert "PIP_RETRIES=10" in content


def test_asr_dockerfile_installs_shared_python_runtime_for_torchcodec():
    content = _read("services/asr/Dockerfile")
    assert "libpython3.12t64" in content or "libpython3.12 " in content

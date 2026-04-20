from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_standalone_dev_compose_exists_and_includes_full_stack():
    content = (ROOT / "docker-compose.dev.yaml").read_text(encoding="utf-8")

    for service in [
        "qdrant:",
        "inference:",
        "orchestrator:",
        "webui:",
        "ingest:",
    ]:
        assert service in content


def test_dev_compose_mounts_all_build_context_code_paths():
    content = (ROOT / "docker-compose.dev.yaml").read_text(encoding="utf-8")

    assert "./services/inference:/app" in content
    assert "./api:/app" in content
    assert "./scripts:/app" in content


def test_dev_compose_persists_huggingface_cache_for_model_and_ingest_services():
    content = (ROOT / "docker-compose.dev.yaml").read_text(encoding="utf-8")

    assert "HF_HOME: /data/huggingface" in content
    assert "HUGGINGFACE_HUB_CACHE: /data/huggingface/hub" in content
    assert "TRANSFORMERS_CACHE:" not in content
    assert "./.cache/huggingface:/data/huggingface" in content


def test_dev_compose_uses_anchors_for_shared_dev_config():
    content = (ROOT / "docker-compose.dev.yaml").read_text(encoding="utf-8")

    assert "x-hf-cache-env:" in content
    assert "x-hf-cache-volume:" in content
    assert "x-gpu-reservation:" in content
    assert "<<: *gpu-reservation" in content


def test_dev_compose_sets_inference_env_defaults():
    content = (ROOT / "docker-compose.dev.yaml").read_text(encoding="utf-8")

    assert "PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True" in content
    assert "QDRANT_COLLECTION: ${SERVICES__QDRANT_COLLECTION}" in content


def test_makefile_contains_direct_dev_compose_targets():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for target in [
        "dev-up:",
        "dev-down:",
        "dev-build:",
        "dev-restart:",
        "dev-logs:",
        "dev-ps:",
        "dev-ingest:",
    ]:
        assert target in makefile

    assert "docker compose -f docker-compose.dev.yaml" in makefile


def test_gitignore_excludes_huggingface_cache_directory():
    content = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".cache/huggingface/" in content

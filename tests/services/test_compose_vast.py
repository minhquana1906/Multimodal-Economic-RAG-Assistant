from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_vast_compose_exists_and_includes_dual_gpu_stack():
    content = (ROOT / "docker-compose.vast.yaml").read_text(encoding="utf-8")

    for service in [
        "qdrant:",
        "embedding:",
        "reranker:",
        "guard:",
        "orchestrator:",
        "llm:",
        "vlm:",
        "webui:",
        "tunnel:",
        "ingest:",
        "asr:",
        "tts:",
    ]:
        assert service in content


def test_vast_compose_uses_image_based_deployment_and_shared_cache():
    content = (ROOT / "docker-compose.vast.yaml").read_text(encoding="utf-8")

    assert "DOCKERHUB_NAMESPACE" in content
    assert "pull_policy: always" in content
    assert "hf_cache:" in content
    assert "HF_HOME: /data/huggingface" in content
    assert "HUGGINGFACE_HUB_CACHE: /data/huggingface/hub" in content
    assert "TRANSFORMERS_CACHE:" not in content
    assert "hf_cache:/data/huggingface" in content
    assert "hf_cache:/root/.cache/huggingface" in content
    assert "build:" not in content
    assert "./services/" not in content
    assert "./api:/app" not in content


def test_vast_compose_pins_gpu_zero_and_gpu_one_service_groups():
    content = (ROOT / "docker-compose.vast.yaml").read_text(encoding="utf-8")

    assert 'device_ids: ["0"]' in content
    assert 'device_ids: ["1"]' in content
    assert "--gpu-memory-utilization" in content
    assert "VLLM_LLM_GPU_MEMORY_UTILIZATION" in content
    assert "VLLM_VLM_GPU_MEMORY_UTILIZATION" in content
    assert "${LLM__MODEL}" in content
    assert "${VLM_MODEL}" in content
    assert "EMBEDDING_MODEL: ${SERVICES__EMBEDDING_MODEL}" in content
    assert "QDRANT_COLLECTION: ${SERVICES__QDRANT_COLLECTION}" in content
    assert "ASR_MODEL: ${SERVICES__ASR_MODEL}" in content
    assert "TTS_MODEL: ${SERVICES__TTS_MODEL}" in content


def test_makefile_contains_direct_vast_compose_targets():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for target in [
        "vast-pull:",
        "vast-up:",
        "vast-down:",
        "vast-logs:",
        "vast-ps:",
    ]:
        assert target in makefile

    assert "docker compose --env-file .env.vast -f docker-compose.vast.yaml" in makefile


def test_makefile_contains_image_build_and_push_targets():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for target in [
        "images-build:",
        "images-push:",
        "images-build-push:",
    ]:
        assert target in makefile

    assert "DOCKERHUB_NAMESPACE ?=" in makefile
    assert "IMAGE_TAG ?= latest" in makefile
    assert "docker build -t $(IMAGE_EMBEDDING) services/embedding" in makefile
    assert "docker push $(IMAGE_ORCHESTRATOR)" in makefile

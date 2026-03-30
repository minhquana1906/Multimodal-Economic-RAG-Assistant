from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_all_compose_files_extend_huggingface_timeouts_for_startup_sensitive_services():
    for compose_path in [
        ROOT / "docker-compose.yaml",
        ROOT / "docker-compose.dev.yaml",
        ROOT / "docker-compose.yml.legacy",
    ]:
        content = compose_path.read_text(encoding="utf-8")

        assert content.count("HF_HUB_ETAG_TIMEOUT") >= 3
        assert content.count("HF_HUB_DOWNLOAD_TIMEOUT") >= 3


def test_all_compose_files_restart_core_model_services_and_allow_longer_warmup():
    for compose_path in [
        ROOT / "docker-compose.yaml",
        ROOT / "docker-compose.dev.yaml",
        ROOT / "docker-compose.yml.legacy",
    ]:
        content = compose_path.read_text(encoding="utf-8")

        assert content.count("restart: unless-stopped") >= 3
        assert content.count("start_period: 300s") >= 3

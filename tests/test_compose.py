"""Tests for Docker Compose configuration correctness."""

import yaml
from pathlib import Path


def load_compose(path: str) -> dict:
    """Load and parse a docker-compose YAML file."""
    return yaml.safe_load(Path(path).read_text())


def test_dev_compose_uses_shared_app_images():
    compose = load_compose("docker-compose.dev.yaml")
    services = compose["services"]
    assert "inference" in services
    assert "orchestrator" in services
    assert "guard" not in services
    assert "asr" not in services
    assert "tts" not in services


def test_prod_compose_uses_shared_app_images():
    compose = load_compose("docker-compose.yaml")
    services = compose["services"]
    assert "inference" in services
    assert "orchestrator" in services
    assert "guard" not in services
    assert "asr" not in services
    assert "tts" not in services

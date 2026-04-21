"""Restore a Qdrant collection from a snapshot.

Downloads the snapshot from Hugging Face Hub if not present locally.
"""

import os
import sys
from pathlib import Path

import requests
from loguru import logger

QDRANT_URL = os.environ.get("SERVICES__QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("SERVICES__QDRANT_COLLECTION", "academic_chunks")
SNAPSHOT_HF_REPO = os.environ.get("SNAPSHOT_HF_REPO", "quannguyen204/economic-rag-snapshots")
SNAPSHOT_FILENAME = os.environ.get("SNAPSHOT_FILENAME", "academic_chunks-3178864631652020-2026-04-20-08-40-03.snapshot")
DATA_DIR = Path(__file__).parent.parent / "data"


def _download_from_hf(filename: str) -> Path:
    """Download snapshot from a public HF dataset repo into DATA_DIR."""
    from huggingface_hub import hf_hub_download

    logger.info(f"downloading {filename} from {SNAPSHOT_HF_REPO}")
    local_path = hf_hub_download(
        repo_id=SNAPSHOT_HF_REPO,
        filename=filename,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
    )
    return Path(local_path)


def find_or_download_snapshot() -> Path:
    """Return local snapshot path, downloading from HF Hub if not found."""
    local = sorted(DATA_DIR.glob(f"{COLLECTION}-*.snapshot"))
    if local:
        return local[-1]

    if not SNAPSHOT_HF_REPO:
        raise RuntimeError(
            f"No local snapshot found in {DATA_DIR} and SNAPSHOT_HF_REPO is not set."
        )

    filename = SNAPSHOT_FILENAME or f"{COLLECTION}-latest.snapshot"
    return _download_from_hf(filename)


def restore(snapshot_path: Path) -> None:
    """Upload snapshot file to Qdrant and recover the collection."""
    url = f"{QDRANT_URL}/collections/{COLLECTION}/snapshots/upload"
    logger.info(f"uploading {snapshot_path.name} → {url}")
    with snapshot_path.open("rb") as fh:
        resp = requests.post(
            url,
            files={"snapshot": fh},
            params={"priority": "snapshot"},
            timeout=600,
        )
    if not resp.ok:
        logger.error(f"upload failed {resp.status_code}: {resp.text}")
        sys.exit(1)
    logger.info(f"collection '{COLLECTION}' restored from snapshot")


if __name__ == "__main__":
    restore(find_or_download_snapshot())

"""Upload a local snapshot file to Qdrant and recover the collection."""

import os
import sys
from pathlib import Path

import requests
from loguru import logger

QDRANT_URL = os.environ.get("SERVICES__QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("SERVICES__QDRANT_COLLECTION", "academic_chunks")
DATA_DIR = Path(__file__).parent.parent / "data"


def find_snapshot() -> Path:
    """Return the most recent snapshot for the target collection."""
    snapshots = sorted(DATA_DIR.glob(f"{COLLECTION}-*.snapshot"))
    if not snapshots:
        raise FileNotFoundError(
            f"No snapshot file matching '{COLLECTION}-*.snapshot' in {DATA_DIR}"
        )
    return snapshots[-1]


def restore(snapshot_path: Path) -> None:
    """Upload snapshot file and recover the Qdrant collection."""
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
    restore(find_snapshot())

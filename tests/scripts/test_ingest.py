"""
tests/scripts/test_ingest.py — Unit tests for ingest.py collection setup + idempotency.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test 1: create_collection sets up dense + sparse vectors
# ---------------------------------------------------------------------------


def test_create_collection_with_dense_and_sparse():
    """create_collection() must configure both a dense (COSINE) and sparse (BM25) vector."""
    from ingest import DENSE_DIM, create_collection

    mock_client = MagicMock()
    create_collection(mock_client, "test_collection", dense_dim=DENSE_DIM)

    mock_client.create_collection.assert_called_once()
    call_kwargs = mock_client.create_collection.call_args.kwargs

    # Collection name
    assert call_kwargs["collection_name"] == "test_collection"

    # Dense vector config
    vectors_config = call_kwargs["vectors_config"]
    assert "dense" in vectors_config, f"Expected 'dense' key; got {list(vectors_config)}"
    from qdrant_client.models import Distance
    assert vectors_config["dense"].size == DENSE_DIM
    assert vectors_config["dense"].distance == Distance.COSINE

    # Sparse vector config
    sparse_config = call_kwargs["sparse_vectors_config"]
    assert "sparse" in sparse_config, f"Expected 'sparse' key; got {list(sparse_config)}"


# ---------------------------------------------------------------------------
# Test 2: Idempotency — skip if collection already has ≥ 400K points
# ---------------------------------------------------------------------------


def test_should_skip_ingestion_when_enough_points():
    """should_skip_ingestion() returns True when point count >= expected_min."""
    from ingest import should_skip_ingestion

    mock_client = MagicMock()
    mock_info = MagicMock()
    mock_info.points_count = 450_000
    mock_client.get_collection.return_value = mock_info

    result = should_skip_ingestion(mock_client, "econ_vn_news", expected_min=400_000)
    assert result is True


def test_should_not_skip_ingestion_when_too_few_points():
    """should_skip_ingestion() returns False when point count < expected_min."""
    from ingest import should_skip_ingestion

    mock_client = MagicMock()
    mock_info = MagicMock()
    mock_info.points_count = 10_000  # Not enough
    mock_client.get_collection.return_value = mock_info

    result = should_skip_ingestion(mock_client, "econ_vn_news", expected_min=400_000)
    assert result is False


def test_should_not_skip_ingestion_when_collection_missing():
    """should_skip_ingestion() returns False when the collection does not exist."""
    from ingest import should_skip_ingestion

    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Collection not found")

    result = should_skip_ingestion(mock_client, "econ_vn_news", expected_min=400_000)
    assert result is False

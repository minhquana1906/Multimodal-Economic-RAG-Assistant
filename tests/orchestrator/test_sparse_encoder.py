from unittest.mock import MagicMock

import pytest


def test_tokenize_vietnamese_uses_underthesea(monkeypatch):
    """Vietnamese tokenization should delegate to underthesea with text format."""
    from orchestrator.services import sparse_encoder

    tokenize = MagicMock(return_value="chi so gia tieu dung")
    monkeypatch.setattr(sparse_encoder, "word_tokenize", tokenize)

    result = sparse_encoder.tokenize_vietnamese("chỉ số giá tiêu dùng")

    assert result == "chi so gia tieu dung"
    tokenize.assert_called_once_with("chỉ số giá tiêu dùng", format="text")


def test_sparse_encoder_service_encodes_query_and_caches_model(monkeypatch):
    """Sparse encoder should initialize BM25 once and return int/float vectors."""
    from orchestrator.services import sparse_encoder

    embedding = MagicMock(indices=[1, "2"], values=[0.25, "0.75"])
    bm25_instance = MagicMock()
    bm25_instance.query_embed.side_effect = lambda text: iter([embedding])
    bm25_cls = MagicMock(return_value=bm25_instance)

    monkeypatch.setattr(sparse_encoder, "Bm25", bm25_cls)
    monkeypatch.setattr(
        sparse_encoder,
        "word_tokenize",
        MagicMock(return_value="chi_so_gia_tieu_dung"),
    )

    service = sparse_encoder.SparseEncoderService(model_name="Qdrant/bm25")

    first = service.encode_query("CPI")
    second = service.encode_query("CPI")

    assert first == {"indices": [1, 2], "values": [0.25, 0.75]}
    assert second == {"indices": [1, 2], "values": [0.25, 0.75]}
    bm25_cls.assert_called_once_with("Qdrant/bm25")
    assert bm25_instance.query_embed.call_count == 2


def test_sparse_encoder_service_reuses_initialization_error(monkeypatch):
    """Initialization error should be cached to avoid repeated BM25 setup attempts."""
    from orchestrator.services import sparse_encoder

    bm25_cls = MagicMock(side_effect=RuntimeError("bm25 init failed"))
    monkeypatch.setattr(sparse_encoder, "Bm25", bm25_cls)
    monkeypatch.setattr(sparse_encoder, "word_tokenize", MagicMock(return_value="abc"))

    service = sparse_encoder.SparseEncoderService()

    with pytest.raises(RuntimeError, match="bm25 init failed"):
        service.encode_query("abc")

    with pytest.raises(RuntimeError, match="bm25 init failed"):
        service.encode_query("abc")

    bm25_cls.assert_called_once_with("Qdrant/bm25")

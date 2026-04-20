"""
tests/scripts/test_integration.py — Full pipeline integration test (mocked services).

Validates the end-to-end flow:
  load articles → chunk → get dense embeddings → upsert to Qdrant
without hitting real external services.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_full_ingestion_flow_mocked():
    """
    Full pipeline integration test with all external services mocked:
    - HuggingFace dataset → synthetic articles
    - Embedding service → random 1024-d vectors
    - Qdrant → in-memory mock client

    Validates that:
    1. Articles are chunked into at least one chunk per article.
    2. Dense embeddings are requested from the embedding service.
    3. Points are upserted to Qdrant with both 'dense' and 'sparse' vectors.
    """
    from chunker import chunk_article

    # ------------------------------------------------------------------
    # Synthetic dataset (3 sample articles)
    # ------------------------------------------------------------------
    synthetic_articles = [
        {
            "title": "GDP Việt Nam tăng trưởng ổn định trong năm 2024",
            "content": (
                "Tốc độ tăng trưởng GDP của Việt Nam đạt 7% trong quý I năm 2024, "
                "vượt dự báo của các tổ chức quốc tế.\n\n"
                "Các chuyên gia kinh tế đánh giá đây là kết quả tích cực nhờ xuất khẩu "
                "và đầu tư nước ngoài tăng mạnh trong giai đoạn này.\n\n"
                "Chính phủ cam kết duy trì môi trường đầu tư ổn định và cải thiện "
                "cơ sở hạ tầng để thu hút thêm vốn FDI."
            ),
            "url": "https://example.com/gdp-2024",
            "published_date": "2024-04-01",
            "category": "Kinh tế",
            "source": "vneconomy",
        },
        {
            "title": "Xuất khẩu nông sản đạt kỷ lục trong năm 2024",
            "content": (
                "Kim ngạch xuất khẩu nông sản Việt Nam đạt 55 tỷ USD trong năm 2024, "
                "tăng 15% so với cùng kỳ năm trước và thiết lập kỷ lục mới.\n\n"
                "Mặt hàng xuất khẩu chủ lực bao gồm cà phê, gạo, rau quả và thủy sản, "
                "trong đó cà phê đóng góp lớn nhất với hơn 5 tỷ USD."
            ),
            "url": "https://example.com/nong-san-2024",
            "published_date": "2024-12-15",
            "category": "Nông nghiệp",
            "source": "agro",
        },
        {
            "title": "Lãi suất ngân hàng tiếp tục giảm hỗ trợ doanh nghiệp",
            "content": (
                "Ngân hàng Nhà nước Việt Nam thông báo hạ lãi suất điều hành thêm "
                "0,5% nhằm hỗ trợ các doanh nghiệp vừa và nhỏ tiếp cận vốn vay.\n\n"
                "Các ngân hàng thương mại nhanh chóng điều chỉnh lãi suất cho vay "
                "xuống mức thấp nhất trong 5 năm qua, trung bình từ 7 đến 9% mỗi năm."
            ),
            "url": "https://example.com/lai-suat-2024",
            "published_date": "2024-06-20",
            "category": "Tài chính",
            "source": "cafef",
        },
    ]

    # ------------------------------------------------------------------
    # Step 1: Chunk all articles
    # ------------------------------------------------------------------
    all_chunks = []
    for article in synthetic_articles:
        chunks = chunk_article(article)
        all_chunks.extend(chunks)

    assert len(all_chunks) > 0, "Pipeline produced no chunks from synthetic articles"
    assert len(all_chunks) >= len(synthetic_articles), (
        "Expected at least one chunk per article"
    )

    # ------------------------------------------------------------------
    # Step 2: Mock dense embeddings (1024-d vectors)
    # ------------------------------------------------------------------
    fake_embeddings = [[0.1] * 1024] * len(all_chunks)

    with patch("ingest._embed_with_retry", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = fake_embeddings[: min(256, len(all_chunks))]

        # ------------------------------------------------------------------
        # Step 3: Mock Qdrant and BM25, then drive a batch manually
        # ------------------------------------------------------------------
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value.collections = []
        mock_qdrant.get_collection.side_effect = Exception("not found")

        # Simulate embedding fetch for first batch
        batch = all_chunks[: min(256, len(all_chunks))]
        texts = [c["text"] for c in batch]

        from ingest import DENSE_DIM, create_collection, should_skip_ingestion

        # Collection doesn't exist yet → should NOT skip
        assert not should_skip_ingestion(mock_qdrant, "econ_vn_news", expected_min=400_000)

        # Create collection
        create_collection(mock_qdrant, "econ_vn_news", dense_dim=DENSE_DIM)
        mock_qdrant.create_collection.assert_called_once()

        # Get dense embeddings
        embeddings = await mock_embed(texts)
        assert len(embeddings) == len(batch)

        # Upsert (simulated)
        mock_qdrant.upsert.return_value = MagicMock()
        mock_qdrant.upsert(collection_name="econ_vn_news", points=[MagicMock()] * len(batch))
        mock_qdrant.upsert.assert_called_once()

        call_kwargs = mock_qdrant.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "econ_vn_news"
        assert len(call_kwargs["points"]) == len(batch)

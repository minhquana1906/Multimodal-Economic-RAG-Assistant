import pytest
from chunker import chunk_article, make_chunk_id, make_article_id


# ---------------------------------------------------------------------------
# Test 1: title_lead chunk contains title + first paragraph
# ---------------------------------------------------------------------------

def test_chunk_article_title_lead():
    article = {
        "title": "GDP Việt Nam tăng 7% trong quý đầu năm nay",
        "content": "Đây là đoạn mở đầu dài hơn 50 ký tự cho bài báo này.\n\nĐoạn thứ hai cũng dài và chứa nhiều thông tin quan trọng.",
        "url": "https://example.com/gdp-viet-nam",
        "published_date": "2024-01-01",
        "category": "Kinh tế",
        "source": "vneconomy",
    }
    chunks = chunk_article(article)

    # Must produce at least 1 chunk (the title_lead)
    assert len(chunks) >= 1
    title_lead = chunks[0]
    assert title_lead["chunk_type"] == "title_lead"
    assert title_lead["chunk_index"] == 0
    assert "GDP" in title_lead["text"]
    assert article["title"] in title_lead["text"]
    # title_lead must include the first paragraph content
    assert "đoạn mở đầu" in title_lead["text"]


# ---------------------------------------------------------------------------
# Test 2: Body paragraphs are created for subsequent paragraphs
# ---------------------------------------------------------------------------

def test_chunk_article_body_paragraphs():
    article = {
        "title": "Xuất khẩu nông sản Việt Nam đạt kỷ lục mới",
        "content": (
            "Đoạn mở đầu dài hơn năm mươi ký tự, giới thiệu bài viết.\n\n"
            "Đoạn thứ hai cũng có độ dài vượt quá năm mươi ký tự, chứa nội dung chính.\n\n"
            "Đoạn thứ ba cũng có độ dài vượt quá năm mươi ký tự, tiếp tục phát triển chủ đề."
        ),
        "url": "https://example.com/xuat-khau",
        "published_date": "2024-02-15",
        "category": "Nông nghiệp",
        "source": "agro",
    }
    chunks = chunk_article(article)

    types = [c["chunk_type"] for c in chunks]
    assert "title_lead" in types
    body_chunks = [c for c in chunks if c["chunk_type"] == "body_paragraph"]
    assert len(body_chunks) >= 2, f"Expected >=2 body paragraphs, got {len(body_chunks)}: {body_chunks}"

    # chunk_index must increment
    indices = [c["chunk_index"] for c in chunks]
    assert indices == sorted(indices)
    assert indices[0] == 0  # title_lead is index 0


# ---------------------------------------------------------------------------
# Test 3: Short paragraphs (< 50 chars) are merged into the next paragraph
# ---------------------------------------------------------------------------

def test_chunk_article_short_paragraphs_merged():
    article = {
        "title": "Lãi suất ngân hàng giảm mạnh trong năm nay",
        "content": (
            "Đoạn mở đầu đủ dài để tạo title_lead chunk hợp lệ nhé.\n\n"
            "Ngắn.\n\n"                             # <50 chars — must be merged
            "Cũng ngắn thôi.\n\n"                   # <50 chars — must be merged together
            "Đây là đoạn đủ dài để kết thúc quá trình merge các đoạn ngắn phía trên."
        ),
        "url": "https://example.com/lai-suat",
        "published_date": "2024-03-10",
        "category": "Tài chính",
        "source": "cafef",
    }
    chunks = chunk_article(article)

    body_chunks = [c for c in chunks if c["chunk_type"] == "body_paragraph"]
    # The two short paragraphs should be merged: we should have fewer body chunks
    # than raw paragraphs (3 raw body paras → merged into fewer)
    assert len(body_chunks) < 3, (
        f"Short paragraphs should be merged; got {len(body_chunks)} body chunks"
    )
    # All body chunk texts must be >= 50 chars (or be the final remainder)
    non_final = body_chunks[:-1]
    for c in non_final:
        assert len(c["text"]) >= 50, (
            f"Non-final body chunk is shorter than 50 chars: {c['text']!r}"
        )


# ---------------------------------------------------------------------------
# Bonus: deterministic IDs
# ---------------------------------------------------------------------------

def test_chunk_deterministic_ids():
    url = "https://example.com/article"
    id1 = make_chunk_id(url, 0)
    id2 = make_chunk_id(url, 0)
    assert id1 == id2
    assert isinstance(id1, int)
    assert id1 > 0

    # Different index → different ID
    id3 = make_chunk_id(url, 1)
    assert id1 != id3

import hashlib


def make_chunk_id(url: str, chunk_index: int) -> int:
    """Deterministic chunk ID from URL + index (SHA256, fits in int63)."""
    content = f"{url}#{chunk_index}"
    return int(hashlib.sha256(content.encode()).hexdigest()[:16], 16) % (2**63)


def make_article_id(url: str) -> int:
    """Deterministic article ID from URL (SHA256, fits in int63)."""
    return int(hashlib.sha256(url.encode()).hexdigest()[:16], 16) % (2**63)


def _merge_short_paragraphs(paragraphs: list[str], min_len: int = 50) -> list[str]:
    """Merge consecutive short paragraphs until each is at least min_len chars."""
    merged = []
    buffer = ""
    for para in paragraphs:
        if not para:
            continue
        if buffer:
            buffer = f"{buffer} {para}"
        else:
            buffer = para
        if len(buffer) >= min_len:
            merged.append(buffer)
            buffer = ""
    if buffer:
        # Append remaining text even if short (last segment)
        merged.append(buffer)
    return merged


def chunk_article(article: dict) -> list[dict]:
    """Chunk an article into title_lead + body_paragraph chunks.

    Strategy:
    - title_lead: title + first paragraph (min 50 chars for that paragraph)
    - body_paragraph: remaining paragraphs, merged when < 50 chars
    """
    title = (article.get("title") or "").strip()
    content = (article.get("content") or "").strip()
    url = article.get("url") or ""
    published_date = article.get("published_date") or ""
    category = article.get("category") or ""
    source = article.get("source") or ""

    if not title or not content:
        return []

    raw_paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    if not raw_paragraphs:
        return []

    chunks = []

    # --- title_lead chunk ---
    first_para = raw_paragraphs[0]
    title_lead_text = f"{title}\n\n{first_para}"
    if len(title_lead_text) >= 50:
        chunks.append(
            {
                "chunk_type": "title_lead",
                "chunk_index": 0,
                "text": title_lead_text,
                "title": title,
                "url": url,
                "published_date": published_date,
                "category": category,
                "source": source,
            }
        )

    # --- body_paragraph chunks ---
    remaining = raw_paragraphs[1:]
    merged = _merge_short_paragraphs(remaining, min_len=50)
    for i, para in enumerate(merged, start=1):
        chunks.append(
            {
                "chunk_type": "body_paragraph",
                "chunk_index": i,
                "text": para,
                "title": title,
                "url": url,
                "published_date": published_date,
                "category": category,
                "source": source,
            }
        )

    return chunks

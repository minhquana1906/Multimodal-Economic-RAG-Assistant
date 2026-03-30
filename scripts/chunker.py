import hashlib
import re


MIN_CHUNK_CHARS = 120
TARGET_CHUNK_CHARS = 900
TARGET_CHUNK_WORDS = 160
MAX_CHUNK_CHARS = 1200
MAX_CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 30

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def make_chunk_id(url: str, chunk_index: int) -> int:
    """Deterministic chunk ID from URL + index (SHA256, fits in int63)."""
    content = f"{url}#{chunk_index}"
    return int(hashlib.sha256(content.encode()).hexdigest()[:16], 16) % (2**63)


def make_article_id(url: str) -> int:
    """Deterministic article ID from URL (SHA256, fits in int63)."""
    return int(hashlib.sha256(url.encode()).hexdigest()[:16], 16) % (2**63)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _word_count(text: str) -> int:
    return len(text.split())


def _fits_limits(
    text: str,
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    max_words: int = MAX_CHUNK_WORDS,
) -> bool:
    return len(text) <= max_chars and _word_count(text) <= max_words


def _should_flush(text: str, *, target_chars: int, target_words: int) -> bool:
    return len(text) >= target_chars or _word_count(text) >= target_words


def _split_paragraphs(content: str) -> list[str]:
    """Split content into structural paragraphs, tolerating inconsistent newlines."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in _PARAGRAPH_SPLIT_RE.split(normalized)
        if paragraph.strip()
    ]
    if len(paragraphs) > 1:
        return paragraphs

    # Some sources use single newlines only, and others collapse the whole article into one block.
    single_line_paragraphs = [
        " ".join(line.split())
        for line in normalized.split("\n")
        if line.strip()
    ]
    return single_line_paragraphs or paragraphs


def _split_by_words(
    text: str,
    *,
    max_chars: int,
    max_words: int,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    words = text.split()
    if not words:
        return []

    segments: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        segment = " ".join(words[start:end]).strip()

        while len(segment) > max_chars and end > start + 1:
            end -= 1
            segment = " ".join(words[start:end]).strip()

        if not segment:
            break

        segments.append(segment)
        if end >= len(words):
            break

        next_start = max(end - overlap_words, start + 1)
        start = next_start

    return segments


def _split_oversized_text(
    text: str,
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    max_words: int = MAX_CHUNK_WORDS,
    target_chars: int = TARGET_CHUNK_CHARS,
    target_words: int = TARGET_CHUNK_WORDS,
) -> list[str]:
    """Split long text into bounded segments while keeping sentence boundaries when possible."""
    normalized = " ".join(_normalize_text(text).split())
    if not normalized:
        return []
    if _fits_limits(normalized, max_chars=max_chars, max_words=max_words):
        return [normalized]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()]
    if len(sentences) <= 1:
        return _split_by_words(
            normalized,
            max_chars=max_chars,
            max_words=max_words,
        )

    segments: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        if not _fits_limits(sentence, max_chars=max_chars, max_words=max_words):
            if current:
                segments.append(" ".join(current).strip())
                current = []
            segments.extend(
                _split_by_words(
                    sentence,
                    max_chars=max_chars,
                    max_words=max_words,
                )
            )
            continue

        if not current:
            current = [sentence]
            continue

        current_text = " ".join(current).strip()
        candidate = f"{current_text} {sentence}".strip()
        if _fits_limits(candidate, max_chars=max_chars, max_words=max_words) and not (
            _should_flush(
                current_text,
                target_chars=min(target_chars, max_chars),
                target_words=min(target_words, max_words),
            )
        ):
            current.append(sentence)
            continue

        segments.append(current_text)
        current = [sentence]

    if current:
        segments.append(" ".join(current).strip())

    return segments


def _merge_short_paragraphs(paragraphs: list[str], min_len: int = 50) -> list[str]:
    """Merge consecutive short segments without violating chunk size limits."""
    merged: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        paragraph = " ".join(paragraph.split()).strip()
        if not paragraph:
            continue

        if not buffer:
            buffer = paragraph
            continue

        if len(buffer) >= min_len:
            merged.append(buffer)
            buffer = paragraph
            continue

        candidate = f"{buffer}\n\n{paragraph}".strip()
        if _fits_limits(candidate):
            buffer = candidate
            continue

        merged.append(buffer)
        buffer = paragraph

    if buffer:
        if merged and len(buffer) < min_len:
            candidate = f"{merged[-1]}\n\n{buffer}".strip()
            if _fits_limits(candidate):
                merged[-1] = candidate
            else:
                merged.append(buffer)
        else:
            merged.append(buffer)

    return merged


def chunk_article(article: dict) -> list[dict]:
    """Chunk an article into bounded title_lead + body chunks.

    Strategy:
    - preserve a title-led first chunk,
    - split oversized paragraphs even when the source has no blank lines,
    - merge tiny residual paragraphs without exceeding the hard chunk limits.
    """
    title = (article.get("title") or "").strip()
    content = (article.get("content") or "").strip()
    url = article.get("url") or ""
    published_date = article.get("published_date") or ""
    category = article.get("category") or ""
    source = article.get("source") or ""

    if not title or not content:
        return []

    raw_paragraphs = _split_paragraphs(content)
    if not raw_paragraphs:
        return []

    chunks = []

    title_budget_chars = max(MAX_CHUNK_CHARS - len(title) - 2, MIN_CHUNK_CHARS)
    title_budget_words = max(MAX_CHUNK_WORDS - _word_count(title), 1)
    title_target_chars = min(TARGET_CHUNK_CHARS, title_budget_chars)
    title_target_words = max(min(TARGET_CHUNK_WORDS, title_budget_words), 1)

    first_segments = _split_oversized_text(
        raw_paragraphs[0],
        max_chars=title_budget_chars,
        max_words=title_budget_words,
        target_chars=title_target_chars,
        target_words=title_target_words,
    )
    if not first_segments:
        return []

    title_lead_text = f"{title}\n\n{first_segments[0]}".strip()
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

    body_segments = list(first_segments[1:])
    for paragraph in raw_paragraphs[1:]:
        body_segments.extend(_split_oversized_text(paragraph))

    merged = _merge_short_paragraphs(body_segments, min_len=50)
    for i, paragraph in enumerate(merged, start=1):
        chunks.append(
            {
                "chunk_type": "body_paragraph",
                "chunk_index": i,
                "text": paragraph,
                "title": title,
                "url": url,
                "published_date": published_date,
                "category": category,
                "source": source,
            }
        )

    return chunks

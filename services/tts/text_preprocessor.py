from __future__ import annotations

import re

from abbreviations import ECONOMIC_ABBREVIATIONS


_DIGITS = [
    "không", "một", "hai", "ba", "bốn",
    "năm", "sáu", "bảy", "tám", "chín",
]



def _read_two_digits(n: int) -> str:
    """Read a two-digit number (10–99) in Vietnamese."""
    assert 10 <= n <= 99
    tens, ones = divmod(n, 10)
    if tens == 1:
        # mười ...
        result = "mười"
        if ones == 0:
            return result
        if ones == 5:
            return result + " lăm"
        return result + " " + _DIGITS[ones]
    # hai mươi, ba mươi, ...
    result = _DIGITS[tens] + " mươi"
    if ones == 0:
        return result
    if ones == 1:
        return result + " mốt"
    if ones == 4:
        return result + " tư"
    if ones == 5:
        return result + " lăm"
    return result + " " + _DIGITS[ones]


def _read_three_digits(n: int) -> str:
    """Read a three-digit number (0–999) in Vietnamese.

    When used as a group inside a larger number, the caller is responsible for
    providing the hundreds digit context.  Here we handle standalone 0–999.
    """
    if n < 10:
        return _DIGITS[n]
    if n < 100:
        return _read_two_digits(n)

    hundreds, remainder = divmod(n, 100)
    result = _DIGITS[hundreds] + " trăm"
    if remainder == 0:
        return result
    if remainder < 10:
        return result + " lẻ " + _DIGITS[remainder]
    return result + " " + _read_two_digits(remainder)


def _read_integer(n: int) -> str:
    """Read a non-negative integer in Vietnamese."""
    if n < 0:
        return "âm " + _read_integer(-n)
    if n < 10:
        return _DIGITS[n]
    if n < 100:
        return _read_two_digits(n)
    if n < 1_000:
        return _read_three_digits(n)

    parts: list[str] = []

    ty, remainder = divmod(n, 1_000_000_000)
    if ty > 0:
        parts.append(_read_three_digits(ty) + " tỷ")

    trieu, remainder = divmod(remainder, 1_000_000)
    if trieu > 0:
        parts.append(_read_three_digits(trieu) + " triệu")
    elif ty > 0 and (remainder > 0 or trieu == 0):
        if remainder > 0:
            parts.append("không triệu")

    nghin, don_vi = divmod(remainder, 1_000)
    if nghin > 0:
        parts.append(_read_three_digits(nghin) + " nghìn")
    elif (ty > 0 or trieu > 0) and don_vi > 0:
        parts.append("không nghìn")

    if don_vi > 0:
        if (ty > 0 or trieu > 0 or nghin > 0) and don_vi < 100:
            if don_vi < 10:
                parts.append("không trăm lẻ " + _DIGITS[don_vi])
            else:
                parts.append("không trăm " + _read_two_digits(don_vi))
        else:
            parts.append(_read_three_digits(don_vi))
    elif n == 0:
        return "không"

    return " ".join(parts)



def normalize_numbers(text: str) -> str:
    # Percentages (decimal or integer)
    def _pct(m: re.Match) -> str:
        num = m.group(1)
        return _read_decimal_or_int(num) + " phần trăm"

    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*%", _pct, text)

    # Quarter/year: Q1/2023, Q2-2024, etc.
    def _quarter(m: re.Match) -> str:
        q = int(m.group(1))
        year = int(m.group(2))
        return "quý " + _DIGITS[q] + " năm " + _read_integer(year)

    text = re.sub(r"[Qq](\d)[/\-](\d{4})", _quarter, text)

    # Integers with thousand separators MUST run before decimal regex,
    # otherwise "1,000,000" would be matched as decimal "1,000" first.
    def _int_with_sep(m: re.Match) -> str:
        raw = m.group(0).replace(",", "").replace(".", "")
        return _read_integer(int(raw))

    text = re.sub(r"\d{1,3}(?:[.,]\d{3})+(?![.,]\d)", _int_with_sep, text)

    # Decimal numbers (use comma or dot as decimal separator)
    def _decimal(m: re.Match) -> str:
        return _read_decimal_or_int(m.group(0))

    text = re.sub(r"\d+[.,]\d+", _decimal, text)

    # Plain integers
    def _plain_int(m: re.Match) -> str:
        return _read_integer(int(m.group(0)))

    text = re.sub(r"\b\d+\b", _plain_int, text)

    return text


def _read_decimal_or_int(s: str) -> str:
    """Read a string that may be an integer or decimal (with . or , separator)."""
    # Normalise separator to dot
    s = s.replace(",", ".")
    if "." in s:
        int_part, frac_part = s.split(".", 1)
        int_word = _read_integer(int(int_part))
        # Read fractional digits as a number (e.g. .14 → mười bốn, .5 → năm)
        frac_word = _read_integer(int(frac_part))
        return int_word + " phẩy " + frac_word
    return _read_integer(int(s))


def expand_abbreviations(
    text: str,
    abbrev_dict: dict[str, str] | None = None,
) -> str:
    """Replace known abbreviations with their full Vietnamese forms.

    Uses :data:`ECONOMIC_ABBREVIATIONS` by default; pass *abbrev_dict* to
    override or extend.
    """
    lookup = dict(ECONOMIC_ABBREVIATIONS)
    if abbrev_dict:
        lookup.update(abbrev_dict)

    # Sort by length descending so longer abbreviations match first
    sorted_abbrevs = sorted(lookup.keys(), key=len, reverse=True)

    for abbr in sorted_abbrevs:
        # Word-boundary match, case-sensitive
        pattern = r"\b" + re.escape(abbr) + r"\b"
        text = re.sub(pattern, lookup[abbr], text)

    return text


def clean_for_speech(text: str) -> str:
    """Strip markdown formatting, citations, URLs, and other non-speech elements."""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove markdown images: ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # Remove markdown links but keep text: [text](url)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # Remove citations like [1], [2,3], [source]
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(r"\[(?:source|citation|ref)\w*\]", "", text, flags=re.IGNORECASE)

    # Remove markdown bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)

    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove markdown code blocks and inline code
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove markdown horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove markdown list markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Remove markdown blockquotes
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Collapse multiple whitespace / blank lines
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized chunks for TTS synthesis.

    Uses Vietnamese sentence-ending punctuation (. ! ? ;) as delimiters,
    keeping the delimiter attached to the preceding sentence.  Very long
    sentences are further split on commas.
    """
    MAX_CHUNK = 256  # VieNeu max_chars default

    # Split on sentence-ending punctuation
    raw = re.split(r"(?<=[.!?;])\s+", text.strip())

    sentences: list[str] = []
    for segment in raw:
        segment = segment.strip()
        if not segment:
            continue
        if len(segment) <= MAX_CHUNK:
            sentences.append(segment)
        else:
            # Further split long segments on commas
            sub_parts = re.split(r"(?<=,)\s+", segment)
            current = ""
            for part in sub_parts:
                if current and len(current) + 1 + len(part) > MAX_CHUNK:
                    sentences.append(current.strip())
                    current = part
                else:
                    current = (current + " " + part).strip() if current else part
            if current.strip():
                sentences.append(current.strip())

    return sentences


def preprocess(text: str) -> list[str]:
    """Full preprocessing pipeline: clean → expand → normalize → split.

    Returns a list of sentence strings ready for TTS synthesis.
    """
    text = clean_for_speech(text)
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    return split_sentences(text)

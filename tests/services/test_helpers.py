# tests/services/test_helpers.py
# Unit tests for pure-logic helpers in the reranker and guard services.
# conftest.py appends all service directories to sys.path so these imports work.


def test_format_pair():
    from reranker_app import format_pair, PREFIX, SUFFIX, INSTRUCTION

    result = format_pair("test query", "test doc", INSTRUCTION)
    assert result.startswith(PREFIX)
    assert result.endswith(SUFFIX)
    assert "<Query>: test query" in result
    assert "<Document>: test doc" in result
    assert INSTRUCTION in result


def test_parse_safety_label_safe():
    from guard_app import parse_safety_label

    assert parse_safety_label("Safety: Safe\nExplanation") == "safe"


def test_parse_safety_label_unsafe_or_controversial():
    from guard_app import parse_safety_label

    assert parse_safety_label("Safety: Unsafe") == "unsafe"
    assert parse_safety_label("Safety: Controversial") == "unsafe"
    assert parse_safety_label("garbage") == "unsafe"  # Fail-closed


def test_extract_label_categories_refusal_filters_none():
    from guard_app import extract_label_categories_refusal

    label, categories, refusal = extract_label_categories_refusal(
        "Safety: Safe\nNone\nRefusal: No"
    )

    assert label == "Safe"
    assert categories == []
    assert refusal == "No"

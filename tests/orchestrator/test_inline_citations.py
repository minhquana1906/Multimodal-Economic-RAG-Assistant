from __future__ import annotations


def test_render_citation_reference_uses_markdown_link():
    from orchestrator.pipeline.rag import _render_citation_reference

    rendered = _render_citation_reference(
        {
            "title": "GDP",
            "url": "https://example.com/gdp",
            "source": "example.com",
        }
    )

    assert rendered == "[GDP](https://example.com/gdp)"


def test_render_citation_reference_falls_back_to_plain_text_without_url():
    from orchestrator.pipeline.rag import _render_citation_reference

    rendered = _render_citation_reference(
        {
            "title": "GDP",
            "url": "",
            "source": "example.com",
        }
    )

    assert rendered == "GDP (example.com)"


def test_extract_citation_ids_ignores_unknown_text():
    from orchestrator.pipeline.rag import _extract_citation_ids

    assert _extract_citation_ids("GDP tăng [[cite:hybrid:1]] nhưng không có [[bad:id]]") == [
        "hybrid:1"
    ]

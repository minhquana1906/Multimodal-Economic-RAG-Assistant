# Phase 2C: Data Pipeline Crawlers — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-03-22-multimodal-stt-tts-vlm-data-pipeline-design.md` §6
**Depends on:** Phase 1 (complete)
**Expected Duration:** 12-15 hours
**Deliverable:** Multi-source crawl framework + HTML/PDF parsers

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CRAWL & PARSE FRAMEWORK                       │
│                                                                   │
│   ┌──────────────┐                                               │
│   │ BaseCrawler   │◄── Abstract base class                       │
│   │ • crawl()     │    All crawlers inherit this                 │
│   │ • rate_limit()│                                               │
│   │ • dedup()     │                                               │
│   │ • robots_txt()│                                               │
│   └──────┬───────┘                                               │
│          │                                                        │
│   ┌──────┼─────────────────────────────┐                         │
│   │      │                              │                         │
│   ▼      ▼                              ▼                         │
│ WebCrawler  PDFCrawler         AcademicCrawler                   │
│ (httpx+BS4) (requests+dl)      (Selenium+httpx)                  │
│   │         │                       │                             │
│   ▼         ▼                       ▼                             │
│ HTMLParser  PDFParser          PDFParser                          │
│ (trafilatura(pdfplumber        (same)                            │
│  + BS4)     + PyMuPDF)                                           │
│   │         │                                                     │
│   ▼         ▼                                                     │
│ data/raw/{source}/     data/images/{hash}_{pg}_{i}.png           │
│ data/parsed/{source}/                                             │
└─────────────────────────────────────────────────────────────────┘
```

## Source Configuration Pattern

```
┌──────────────────────────────────────────────────────┐
│  sources/vnexpress.py                                 │
│                                                       │
│  SOURCE_CONFIG = {                                    │
│    "name": "vnexpress",                              │
│    "base_url": "https://vnexpress.net/kinh-doanh",   │
│    "type": "web",                                     │
│    "quality": "medium",                               │
│    "selectors": {                                     │
│      "article_list": "article.item-news",            │
│      "title": "h1.title-detail",                     │
│      "content": "article.fck_detail",                │
│      "date": "span.date",                            │
│      "category": "ul.breadcrumb li:nth-child(2)",    │
│    },                                                 │
│    "pagination": {                                    │
│      "type": "page_param",                           │
│      "param": "page",                                │
│      "max_pages": 100,                               │
│    },                                                 │
│    "rate_limit": 1.0,  # requests per second         │
│  }                                                    │
└──────────────────────────────────────────────────────┘
```

## Tasks

### Task 1: Base Crawler Framework (2h)

**Create:** `scripts/crawlers/__init__.py`
**Create:** `scripts/crawlers/base.py`

```python
# BaseCrawler ABC:
# - __init__(config, data_dir)
# - async crawl() → list[RawDocument]
# - rate_limit() — async sleep between requests
# - is_duplicate(url) → bool — URL hash dedup
# - check_robots_txt(url) → bool
# - save_raw(content, source, url) — write to data/raw/
#
# RawDocument dataclass:
# - url, title, content_path, source, crawled_at, content_type
```

### Task 2: Source Registry & Config (1h)

**Create:** `scripts/crawlers/config.py`
- `SOURCES: dict[str, SourceConfig]` — all source configurations
- `SourceConfig` dataclass: name, base_url, type, quality, selectors, pagination, rate_limit

**Create:** `scripts/crawlers/sources/` — per-source configs:
- `vnexpress.py`, `cafef.py`, `vietnambiz.py`, `vneconomy.py`
- `cafef_reports.py` (PDF), `vietstock_reports.py` (PDF)
- `scholar_vn.py` (academic)

### Task 3: Web Crawler (3h)

**Create:** `scripts/crawlers/web_crawler.py`
- Inherits `BaseCrawler`
- Uses `httpx.AsyncClient` + `BeautifulSoup`
- Per-source CSS selectors from config
- Pagination handling (page params, next links)
- Article list page → individual article pages
- Save raw HTML to `data/raw/{source}/{date}/{hash}.html`

### Task 4: PDF Crawler (2h)

**Create:** `scripts/crawlers/pdf_crawler.py`
- Inherits `BaseCrawler`
- Direct URL download for known report pages
- Selenium fallback for dynamic pages (login walls, JS rendering)
- Save PDF to `data/raw/{source}/{date}/{hash}.pdf`
- Extract basic metadata from download page (title, date)

### Task 5: HTML Parser (2h)

**Create:** `scripts/parsers/__init__.py`
**Create:** `scripts/parsers/html_parser.py`
- Primary: `trafilatura.extract()` for clean article text
- Fallback: `BeautifulSoup` with per-source selectors
- Output: `ParsedDocument` with title, content, date, category, source, url, doc_type
- Handle Vietnamese encoding (UTF-8)
- Strip ads, nav, sidebars, related articles

### Task 6: PDF Parser (3h)

**Create:** `scripts/parsers/pdf_parser.py`
- `pdfplumber` for text extraction + table detection
- `PyMuPDF (fitz)` for image extraction
- Per-page layout analysis:
  - Text blocks → paragraphs
  - Table regions (detected by cell grid lines) → image extraction
  - Image regions (no grid lines) → chart image extraction
- Save extracted images to `data/images/{doc_hash}_{page}_{idx}.png`
- Output: `ParsedDocument` with text_sections, visual_elements list

**Create:** `scripts/parsers/visual_extractor.py`
- `detect_visual_type(image) -> "table" | "chart" | "figure"`
- Heuristic: check for grid lines (table) vs curves/bars (chart)
- Simple approach: if pdfplumber detects table cells → table, else → chart
- Extract region as PNG at 300 DPI

### Task 7: Tests (2h)

**Create:** `tests/scripts/test_crawlers.py`
- `test_base_crawler_dedup`
- `test_base_crawler_rate_limit`
- `test_web_crawler_parses_article`
- `test_source_config_loading`

**Create:** `tests/scripts/test_parsers.py`
- `test_html_parser_extracts_article`
- `test_pdf_parser_extracts_text`
- `test_pdf_parser_detects_tables`
- `test_visual_extractor_classifies`

### Verification

```bash
pytest tests/scripts/test_crawlers.py tests/scripts/test_parsers.py -v

# Manual crawl test (single source, 5 pages)
python -m scripts.crawlers.web_crawler --source vnexpress --max-pages 5 --data-dir ./data
ls data/raw/vnexpress/
ls data/parsed/vnexpress/
```

---

## Summary

| Task | Description | Effort |
|------|-------------|--------|
| 1 | Base Crawler Framework | 2h |
| 2 | Source Registry & Config | 1h |
| 3 | Web Crawler | 3h |
| 4 | PDF Crawler | 2h |
| 5 | HTML Parser | 2h |
| 6 | PDF Parser + Visual Extractor | 3h |
| 7 | Tests | 2h |
| **Total** | **7 tasks** | **12-15h** |

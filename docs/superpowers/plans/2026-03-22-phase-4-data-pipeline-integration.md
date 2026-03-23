# Phase 4: Data Pipeline Integration — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-03-22-multimodal-stt-tts-vlm-data-pipeline-design.md` §6
**Depends on:** Phase 2C (crawlers), Phase 3B (VLM enricher)
**Expected Duration:** 10-12 hours
**Deliverable:** Complete crawl-to-Qdrant pipeline with multi-collection support

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              DATA PIPELINE ORCHESTRATOR                       │
│              scripts/pipeline/orchestrator.py                 │
│                                                               │
│   ┌────────────┐                                             │
│   │ STAGE 1    │  Crawl raw data                             │
│   │ crawl()    │  → data/raw/{source}/                       │
│   └─────┬──────┘                                             │
│         │                                                     │
│   ┌─────▼──────┐                                             │
│   │ STAGE 2    │  Parse HTML/PDF                             │
│   │ parse()    │  Extract visual elements                    │
│   │            │  → data/parsed/, data/images/               │
│   └─────┬──────┘                                             │
│         │                                                     │
│   ┌─────▼──────┐                                             │
│   │ STAGE 3    │  VLM enrich charts/tables                   │
│   │ enrich()   │  Extended chunking                          │
│   │            │  → unified chunks list                      │
│   └─────┬──────┘                                             │
│         │                                                     │
│   ┌─────▼──────┐                                             │
│   │ STAGE 4    │  Embed (dense+sparse)                       │
│   │ index()    │  Upsert to Qdrant "econ_knowledge"          │
│   └────────────┘                                             │
│                                                               │
│   CLI: python scripts/ingest_crawled.py                      │
│         --sources vnexpress,cafef,cafef_reports               │
│         --max-pages 100                                       │
│         --data-dir ./data                                     │
└─────────────────────────────────────────────────────────────┘
```

## Extended Chunking Strategy

```
Document Type        Chunk Types
─────────────────    ──────────────────────────────────
Web Article       →  title_lead + body_paragraph
PDF Report        →  title_lead + section_content + table_content + chart_description
Academic Paper    →  title_lead (title+abstract) + section_content + table_content + chart_description
Knowledge Blog    →  title_lead + body_paragraph + definition
```

## Qdrant Multi-Collection Layout

```
┌──────────────────────┐    ┌──────────────────────────────┐
│   econ_vn_news       │    │   econ_knowledge              │
│   (existing, HF)     │    │   (NEW, crawled)              │
│                      │    │                                │
│   ~1.3M chunks       │    │   ~500K+ chunks (estimated)   │
│   308K articles      │    │   Multiple doc types           │
│                      │    │                                │
│   dense + sparse     │    │   dense + sparse               │
│   chunk_type:        │    │   chunk_type:                  │
│   • title_lead       │    │   • title_lead                 │
│   • body_paragraph   │    │   • body_paragraph             │
│                      │    │   • section_content            │
│                      │    │   • table_content              │
│                      │    │   • chart_description          │
│                      │    │   • definition                 │
│                      │    │                                │
│                      │    │   + image_path (nullable)      │
│                      │    │   + structured_data (nullable) │
│                      │    │   + doc_type                   │
│                      │    │   + source_quality             │
└──────────────────────┘    └──────────────────────────────┘
```

## Tasks

### Task 1: Extended Chunker (3h)

**Modify:** `scripts/chunker.py`
- Add new chunk types: `section_content`, `table_content`, `chart_description`, `definition`
- PDF-aware chunking: detect section headings (larger font / bold), split by sections
- Table chunks: VLM markdown output as text, no max length
- Chart chunks: VLM description as text
- Definition chunks: detect Q&A patterns ("X là gì?"), glossary-style definitions
- Max chunk size: 1500 chars (split at sentence boundary via `underthesea`)

```python
def chunk_parsed_document(doc: ParsedDocument) -> list[Chunk]:
    """Extended chunker for all document types."""
```

### Task 2: Pipeline Orchestrator (3h)

**Create:** `scripts/pipeline/orchestrator.py`
- Main pipeline class: `DataPipelineOrchestrator`
- `async run(sources, max_pages, data_dir)`
- Stage 1: call crawlers for specified sources
- Stage 2: call parsers on raw data
- Stage 3: call VLM enricher on visual elements + chunker on text
- Stage 4: call embedder + BM25 + upsert
- Progress tracking with loguru
- Error handling: skip failed documents, continue pipeline
- Resume support: track processed doc IDs in `data/processed.jsonl`

### Task 3: Multi-Collection Manager (2h)

**Create:** `scripts/pipeline/multi_collection.py`
- `create_collection(name, schema)` — idempotent
- `upsert_batch(collection, points)` — batched upsert
- `get_collection_stats(name)` — point count, etc.
- Schema for `econ_knowledge` with extended payload fields

```python
ECON_KNOWLEDGE_SCHEMA = {
    "vectors": {
        "dense": VectorParams(size=1024, distance=Distance.COSINE),
        "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
    },
    "payload_schema": {
        "doc_id": "keyword",
        "chunk_type": "keyword",
        "doc_type": "keyword",
        "source_quality": "keyword",
        "image_path": "keyword",
    }
}
```

### Task 4: CLI Entry Point (2h)

**Create:** `scripts/ingest_crawled.py`
- argparse CLI:
  - `--sources` — comma-separated source names (or "all")
  - `--max-pages` — per-source page limit
  - `--data-dir` — base data directory
  - `--skip-crawl` — skip stage 1 (use existing raw data)
  - `--skip-vlm` — skip VLM enrichment (text-only chunks)
- Calls `DataPipelineOrchestrator.run()`
- Reports final stats: documents crawled, chunks created, points indexed

**Update:** `docker-compose.yml`
- Add `crawl-ingest` service under `profiles: [crawl]`
- Depends on: qdrant, embedding (+ vlm if not --skip-vlm)

### Task 5: Tests (2h)

**Create:** `tests/scripts/test_chunker_extended.py`
- `test_chunk_pdf_report_produces_sections`
- `test_chunk_table_content`
- `test_chunk_chart_description`
- `test_chunk_definition`
- `test_chunk_max_length_split`

**Create:** `tests/scripts/test_data_pipeline.py`
- `test_pipeline_orchestrator_stages`
- `test_multi_collection_create`
- `test_multi_collection_upsert`
- `test_cli_argument_parsing`

### Verification

```bash
pytest tests/scripts/test_chunker_extended.py tests/scripts/test_data_pipeline.py -v

# Manual integration test
docker compose --profile crawl up -d qdrant embedding
python scripts/ingest_crawled.py \
  --sources vnexpress \
  --max-pages 5 \
  --data-dir ./data \
  --skip-vlm

# Check Qdrant
curl http://localhost:6333/collections/econ_knowledge
```

---

## Summary

| Task | Description | Effort |
|------|-------------|--------|
| 1 | Extended Chunker | 3h |
| 2 | Pipeline Orchestrator | 3h |
| 3 | Multi-Collection Manager | 2h |
| 4 | CLI Entry Point + Docker | 2h |
| 5 | Tests | 2h |
| **Total** | **5 tasks** | **10-12h** |

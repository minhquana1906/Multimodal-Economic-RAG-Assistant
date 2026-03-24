## Test Workflow

The canonical local unit-test entrypoint for this repository is:

```bash
uv run pytest
```

That command is the default contract for local verification during the backend RAG remediation work. The matching shortcuts are:

```bash
make test
make test-integration
```

`integration` tests are opt-in and skipped by default in local runs. Use them only when Docker services, pinned images, and any required GPU-backed dependencies are available:

```bash
uv run pytest -m integration
```

## What The Regression Matrix Covers

`tests/orchestrator/test_regression_matrix.py` defines the baseline regression matrix for the remediation plan. It keeps the acceptance surface explicit for:

- single-turn factual chat,
- follow-up questions that depend on earlier turns,
- sparse-sensitive keyword and entity-heavy retrieval cases,
- no-context responses,
- web-fallback queries, and
- citation-sensitive multiline answers where streaming must preserve inline markdown links.

The matrix is intentionally repo-facing. It does not replace focused router, pipeline, or service tests. It records which user-visible behaviors the later remediation tasks must protect while the backend conversation, retrieval, citation, tracing, and streaming contracts are being refactored.

## Test Scope

These tests should pass locally with `uv run pytest` without requiring the full Docker stack:

- orchestrator unit tests,
- service unit tests that rely on lightweight test doubles, and
- regression-matrix checks for the backend remediation plan.

These tests require additional setup and are not part of the default local run:

- `integration`-marked tests,
- Docker Compose validation against running services,
- GPU-backed inference checks, and
- manual OpenWebUI / LangSmith verification steps from the remediation plan.

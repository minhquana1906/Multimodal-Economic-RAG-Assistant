from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegressionScenario:
    case_id: str
    description: str
    messages: tuple[tuple[str, str], ...]
    retrieval_mode: str
    acceptance_signals: tuple[str, ...]
    requires_stream_parity: bool = False


REGRESSION_MATRIX = (
    RegressionScenario(
        case_id="single_turn_factual",
        description="Single-turn factual question should stay stable through backend refactors.",
        messages=(("user", "GDP Viet Nam tang bao nhieu trong nam 2024?"),),
        retrieval_mode="hybrid",
        acceptance_signals=("preserves factual answer text", "returns citations"),
    ),
    RegressionScenario(
        case_id="follow_up_with_history",
        description="Follow-up query depends on earlier turns and should exercise conversation state.",
        messages=(
            ("user", "Tom tat tinh hinh thi truong bat dong san."),
            ("assistant", "Thi truong dang phuc hoi cham."),
            ("user", "Con trai phieu doanh nghiep trong boi canh do thi sao?"),
        ),
        retrieval_mode="hybrid",
        acceptance_signals=("uses prior-turn context", "rewrites follow-up query"),
    ),
    RegressionScenario(
        case_id="sparse_sensitive_keyword",
        description="Entity-heavy query should remain in the acceptance set once sparse retrieval is wired.",
        messages=(("user", "Nghi dinh 08 trai phieu doanh nghiep tac dong den NVL ra sao?"),),
        retrieval_mode="hybrid",
        acceptance_signals=("preserves entity keywords", "exercises sparse retrieval path"),
    ),
    RegressionScenario(
        case_id="no_context_query",
        description="No-context query should return the explicit no-context response instead of hallucinating.",
        messages=(("user", "Thong tin noi bo chua tung cong bo ve doanh nghiep X?"),),
        retrieval_mode="dense_only",
        acceptance_signals=("returns no-context answer",),
    ),
    RegressionScenario(
        case_id="web_fallback",
        description="Queries that miss local context should preserve web provenance once fallback runs.",
        messages=(("user", "Tin moi nhat ve chinh sach FED hom nay la gi?"),),
        retrieval_mode="web_fallback",
        acceptance_signals=("preserves web provenance", "returns web citation candidates"),
    ),
    RegressionScenario(
        case_id="streaming_inline_citations",
        description="Streaming must preserve markdown formatting and inline citations byte-for-byte.",
        messages=(("user", "Lap bang tom tat ngan ve GDP va kem nguon truc tiep."),),
        retrieval_mode="hybrid",
        acceptance_signals=(
            "preserves multiline markdown",
            "renders inline markdown links",
            "streaming matches non-streaming answer",
        ),
        requires_stream_parity=True,
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_regression_matrix_covers_plan_flows():
    assert [scenario.case_id for scenario in REGRESSION_MATRIX] == [
        "single_turn_factual",
        "follow_up_with_history",
        "sparse_sensitive_keyword",
        "no_context_query",
        "web_fallback",
        "streaming_inline_citations",
    ]


def test_regression_matrix_cases_define_explicit_acceptance_signals():
    for scenario in REGRESSION_MATRIX:
        assert scenario.messages
        assert scenario.messages[-1][0] == "user"
        assert scenario.retrieval_mode in {"hybrid", "dense_only", "web_fallback"}
        assert scenario.acceptance_signals


def test_repo_documents_uv_pytest_contract():
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")

    assert "uv run pytest" in readme
    assert "integration" in readme.lower()
    assert "regression matrix" in readme.lower()


def test_repo_registers_integration_tests_as_opt_in():
    pyproject = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")

    assert 'not integration' in pyproject
    assert "integration:" in pyproject


def test_makefile_exposes_repo_test_entrypoints():
    makefile = (_repo_root() / "Makefile").read_text(encoding="utf-8")

    assert "\ntest:\n\tuv run pytest\n" in f"\n{makefile}"
    assert "\ntest-integration:\n\tuv run pytest -m integration\n" in f"\n{makefile}"

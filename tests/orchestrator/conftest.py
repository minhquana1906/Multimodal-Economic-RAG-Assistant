"""
conftest.py for orchestrator tests.

The orchestrator Python package lives inside the Docker build-context directory
api/orchestrator/.  We add that build-context directory (api/) to sys.path so
that `import orchestrator` resolves correctly when running tests from the repo root.
"""
import sys
from pathlib import Path

from loguru import logger

_root = Path(__file__).parents[2]          # repo root
_orch_ctx = str(_root / "api")             # api/ Docker context dir

if _orch_ctx not in sys.path:
    sys.path.insert(0, _orch_ctx)

# Register custom domain levels so service clients can call logger.log("RETRIEVAL", ...)
# etc. without setup_logging() being invoked (mirrors tracing.DOMAIN_LEVELS).
_DOMAIN_LEVELS = [
    ("RETRIEVAL", 25, "<cyan>"),
    ("RERANK",    26, "<blue>"),
    ("GUARD",     27, "<yellow>"),
    ("LLM",       28, "<magenta>"),
]
for _name, _no, _color in _DOMAIN_LEVELS:
    try:
        logger.level(_name, no=_no, color=_color)
    except (TypeError, ValueError):
        pass  # already registered

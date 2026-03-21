"""
conftest.py for orchestrator tests.

The orchestrator Python package lives inside the Docker build-context directory
orchestrator/orchestrator/.  We add that build-context directory to sys.path so
that `import orchestrator` resolves correctly when running tests from the repo root.
"""
import sys
from pathlib import Path

_root = Path(__file__).parents[2]          # repo root
_orch_ctx = str(_root / "orchestrator")    # orchestrator/ Docker context dir

if _orch_ctx not in sys.path:
    sys.path.insert(0, _orch_ctx)

import sys
from pathlib import Path

# Register both service paths so `import app` can find the right module
# when tests are run with PYTHONPATH=services/<name>.
#
# We use sys.path.append (NOT insert) so that an explicit PYTHONPATH set at
# invocation time always takes precedence over these fallback entries:
#
#   PYTHONPATH=services/reranker .venv/bin/python -m pytest ...
#   PYTHONPATH=services/embedding .venv/bin/python -m pytest ...
#
# Python prepends PYTHONPATH entries to sys.path before pytest starts, so
# appending here means the caller-supplied path wins.

_root = Path(__file__).parents[2]

for _service in ("embedding", "reranker", "guard"):
    _svc_path = str(_root / "services" / _service)
    if _svc_path not in sys.path:
        sys.path.append(_svc_path)

# Evict any already-cached app module so the next import picks
# up the correct one from whichever service path is first on sys.path.
sys.modules.pop("app", None)

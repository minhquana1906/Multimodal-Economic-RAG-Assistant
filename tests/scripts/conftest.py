import sys
from pathlib import Path

# Add the scripts/ directory to sys.path so tests can import `chunker`, `ingest`, etc.
# directly (same pattern as tests/services/conftest.py for service modules).
_root = Path(__file__).parents[2]
_scripts_path = str(_root / "scripts")
if _scripts_path not in sys.path:
    sys.path.insert(0, _scripts_path)

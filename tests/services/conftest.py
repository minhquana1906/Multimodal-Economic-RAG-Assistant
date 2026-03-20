import sys
from pathlib import Path

print(f"[conftest] sys.modules has 'main': {'main' in sys.modules}")
if 'main' in sys.modules:
    print(f"[conftest] main was: {sys.modules['main'].__file__}")

# Insert services/embedding at the FRONT of sys.path so that
# `import main` finds services/embedding/main.py before the
# placeholder root-level main.py.
_embedding_path = str(Path(__file__).parents[2] / "services" / "embedding")
if _embedding_path not in sys.path:
    sys.path.insert(0, _embedding_path)

# Evict any already-cached root main module so the next import picks
# up the correct one from services/embedding/.
sys.modules.pop("main", None)

print(f"[conftest] After fix, sys.path[0]: {sys.path[0]}")

import sys
from pathlib import Path

# Insert services/embedding at the FRONT of sys.path so that
# `import app` finds services/embedding/app.py before any other module.
_embedding_path = str(Path(__file__).parents[2] / "services" / "embedding")
if _embedding_path not in sys.path:
    sys.path.insert(0, _embedding_path)

# Evict any already-cached app module so the next import picks
# up the correct one from services/embedding/.
sys.modules.pop("app", None)

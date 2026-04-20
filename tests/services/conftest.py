import importlib.machinery
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Register service paths so `import inference_app` / `import guard_app` can find
# the right module when tests are run with PYTHONPATH=services/<name>.
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

for _service in ("inference", "guard", "asr", "tts"):
    _svc_path = str(_root / "services" / _service)
    if _svc_path not in sys.path:
        sys.path.append(_svc_path)

# FlagEmbedding is not installed in the test env; provide a stub so service
# modules can import normally and tests can patch the imported symbols.
if "FlagEmbedding" not in sys.modules:
    fake_flag_embedding = types.ModuleType("FlagEmbedding")
    fake_flag_embedding.BGEM3FlagModel = MagicMock(name="BGEM3FlagModel")
    fake_flag_embedding.FlagReranker = MagicMock(name="FlagReranker")
    sys.modules["FlagEmbedding"] = fake_flag_embedding

# Test env does not install transformers; provide a lightweight stub so service
# modules can import normally and tests can patch the imported symbols.
if "transformers" not in sys.modules:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = MagicMock(name="AutoModelForCausalLM")
    fake_transformers.AutoTokenizer = MagicMock(name="AutoTokenizer")
    sys.modules["transformers"] = fake_transformers


def _load_or_stub_sentence_transformers() -> None:
    try:
        __import__("sentence_transformers")
    except Exception:
        fake_sentence_transformers = types.ModuleType("sentence_transformers")
        fake_sentence_transformers.SentenceTransformer = MagicMock(
            name="SentenceTransformer"
        )
        sys.modules["sentence_transformers"] = fake_sentence_transformers


class _FakeTensor:
    def __init__(self, value):
        self._array = np.asarray(value)

    @property
    def shape(self):
        return self._array.shape

    def mean(self, dim=None, keepdim=False):
        return _FakeTensor(self._array.mean(axis=dim, keepdims=keepdim))

    def squeeze(self, dim=None):
        if dim is None:
            return _FakeTensor(np.squeeze(self._array))
        return _FakeTensor(np.squeeze(self._array, axis=dim))

    def unsqueeze(self, dim):
        return _FakeTensor(np.expand_dims(self._array, axis=dim))

    def numpy(self):
        return np.asarray(self._array)

    def exp(self):
        return _FakeTensor(np.exp(self._array))

    def item(self):
        return float(np.asarray(self._array).item())

    def to(self, *_args, **_kwargs):
        return self

    def tolist(self):
        return self._array.tolist()

    def __getitem__(self, item):
        return _FakeTensor(self._array[item])

    def __setitem__(self, item, value):
        if isinstance(value, _FakeTensor):
            self._array[item] = value._array
            return
        self._array[item] = value

    def __array__(self, dtype=None):
        return np.asarray(self._array, dtype=dtype)


def _install_torch_stub() -> None:
    fake_torch = types.ModuleType("torch")

    class _OutOfMemoryError(Exception):
        pass

    fake_torch.Tensor = _FakeTensor
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.tensor = lambda value: _FakeTensor(value)
    fake_torch.zeros = lambda *shape: _FakeTensor(np.zeros(shape))
    fake_torch.from_numpy = lambda value: _FakeTensor(value)
    fake_torch.device = lambda value: value
    fake_torch.no_grad = lambda: nullcontext()
    fake_torch.inference_mode = lambda: nullcontext()
    fake_torch.cuda = types.SimpleNamespace(
        empty_cache=lambda: None,
        OutOfMemoryError=_OutOfMemoryError,
        is_available=lambda: False,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
        max_memory_allocated=lambda: 0,
    )

    # Set __spec__ so importlib.util.find_spec("torch") returns a spec instead of
    # raising ValueError (triggered by libraries like `datasets` that probe for torch).
    fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    sys.modules["torch"] = fake_torch


def _install_torchaudio_stub() -> None:
    fake_torchaudio = types.ModuleType("torchaudio")

    class _Resample:
        def __init__(self, orig_freq, new_freq):
            self.orig_freq = orig_freq
            self.new_freq = new_freq

        def __call__(self, waveform):
            return waveform

    fake_torchaudio.load = MagicMock(name="load")
    fake_torchaudio.transforms = types.SimpleNamespace(Resample=_Resample)
    sys.modules["torchaudio"] = fake_torchaudio


def _load_or_stub_module(module_name: str, fallback) -> None:
    try:
        __import__(module_name)
    except Exception:
        fallback()


_load_or_stub_module("torch", _install_torch_stub)
_load_or_stub_module("torchaudio", _install_torchaudio_stub)
_load_or_stub_sentence_transformers()

# Evict any already-cached service modules so the next import picks
# up the correct one from whichever service path is first on sys.path.
for _mod in (
    "inference_app",
    "guard_app",
    "asr_app",
    "tts_app",
    "text_preprocessor",
    "abbreviations",
):
    sys.modules.pop(_mod, None)

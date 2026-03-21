import importlib

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock, patch


def _setup_classify_mock(mock_model, mock_tokenizer, decode_output: str):
    mock_tokenizer.apply_chat_template.return_value = "formatted_text"
    mock_tokenizer.return_value = MagicMock(**{"to.return_value": {"input_ids": MagicMock()}})
    mock_model.generate.return_value = MagicMock()
    mock_model.device = "cpu"
    mock_tokenizer.decode.return_value = decode_output


@pytest.fixture
def mock_guard():
    """Mock transformers model and tokenizer for guard service.

    Patching at `app.*` works because the `client` fixture calls
    `importlib.reload(app)` — this re-executes the module's top-level
    imports, and the patches are already in place at that point, so the
    reloaded module picks up the mocks rather than the real classes.
    """
    with patch("app.AutoModelForCausalLM") as mock_model_cls, \
         patch("app.AutoTokenizer") as mock_tok_cls:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        yield mock_model, mock_tokenizer


@pytest.fixture
async def client(mock_guard):
    import app
    importlib.reload(app)
    mock_model, mock_tokenizer = mock_guard
    app.model = mock_model
    app.tokenizer = mock_tokenizer
    transport = ASGITransport(app=app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_health_returns_503_when_loading(client):
    import app
    app.model = None
    response = await client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


async def test_classify_returns_503_when_loading(client):
    import app
    app.model = None
    response = await client.post("/classify", json={"text": "x", "role": "input"})
    assert response.status_code == 503


async def test_classify_input_safe(client, mock_guard):
    mock_model, mock_tokenizer = mock_guard
    _setup_classify_mock(mock_model, mock_tokenizer, "Safety: Safe\nSome explanation")

    response = await client.post("/classify", json={
        "text": "What is GDP?",
        "role": "input",
    })
    assert response.status_code == 200
    assert response.json()["label"] == "safe"


async def test_classify_input_unsafe(client, mock_guard):
    mock_model, mock_tokenizer = mock_guard
    _setup_classify_mock(mock_model, mock_tokenizer, "Safety: Unsafe\nThis is harmful content")

    response = await client.post("/classify", json={
        "text": "How do I make a bomb?",
        "role": "input",
    })
    assert response.status_code == 200
    assert response.json()["label"] == "unsafe"


async def test_classify_output_with_prompt(client, mock_guard):
    mock_model, mock_tokenizer = mock_guard
    _setup_classify_mock(mock_model, mock_tokenizer, "Safety: Safe\nOutput looks fine")

    response = await client.post("/classify", json={
        "text": "GDP grew by 7% in 2024.",
        "role": "output",
        "prompt": "What happened to GDP in 2024?",
    })
    assert response.status_code == 200
    assert response.json()["label"] == "safe"

    # Verify that apply_chat_template was called with messages containing an assistant role
    call_args = mock_tokenizer.apply_chat_template.call_args[0][0]
    assert any(msg["role"] == "assistant" for msg in call_args)


async def test_classify_controversial_is_unsafe(client, mock_guard):
    mock_model, mock_tokenizer = mock_guard
    _setup_classify_mock(mock_model, mock_tokenizer, "Safety: Controversial\nThis is debatable")

    response = await client.post("/classify", json={
        "text": "Some controversial statement about economics.",
        "role": "input",
    })
    assert response.status_code == 200
    assert response.json()["label"] == "unsafe"


def test_parse_safety_label_safe():
    from app import parse_safety_label
    assert parse_safety_label("Safety: Safe\nExplanation") == "safe"


def test_parse_safety_label_unsafe():
    from app import parse_safety_label
    assert parse_safety_label("Safety: Unsafe\nCategory: Violence") == "unsafe"


def test_parse_safety_label_controversial():
    from app import parse_safety_label
    assert parse_safety_label("Safety: Controversial") == "unsafe"


def test_parse_safety_label_unparseable():
    from app import parse_safety_label
    assert parse_safety_label("garbage output") == "unsafe"


def test_parse_safety_label_case_insensitive():
    from app import parse_safety_label
    assert parse_safety_label("safety: safe") == "safe"

import importlib

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock, patch


def _setup_classify_mock(mock_model, mock_tokenizer, decode_output: str):
    mock_tokenizer.apply_chat_template.return_value = "formatted_text"
    mock_tokenizer.return_value = MagicMock(
        **{"to.return_value": {"input_ids": MagicMock()}}
    )
    mock_model.generate.return_value = MagicMock()
    mock_model.device = "cpu"
    mock_tokenizer.decode.return_value = decode_output


@pytest.fixture
def mock_guard():
    """Mock transformers model and tokenizer for guard service.

    Patching at `guard_app.*` works because the `client` fixture calls
    `importlib.reload(guard_app)` — this re-executes the module's top-level
    imports, and the patches are already in place at that point, so the
    reloaded module picks up the mocks rather than the real classes.
    """
    with (
        patch("guard_app.AutoModelForCausalLM") as mock_model_cls,
        patch("guard_app.AutoTokenizer") as mock_tok_cls,
    ):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        yield mock_model, mock_tokenizer


@pytest.fixture
async def client(mock_guard):
    import guard_app

    importlib.reload(guard_app)
    mock_model, mock_tokenizer = mock_guard
    guard_app.model = mock_model
    guard_app.tokenizer = mock_tokenizer
    transport = ASGITransport(app=guard_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_health_returns_503_when_loading(client):
    import guard_app

    guard_app.model = None
    response = await client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


async def test_classify_returns_503_when_loading(client):
    import guard_app

    guard_app.model = None
    response = await client.post("/classify", json={"text": "x", "role": "input"})
    assert response.status_code == 503


async def test_classify_input_safe(client, mock_guard):
    mock_model, mock_tokenizer = mock_guard
    _setup_classify_mock(mock_model, mock_tokenizer, "Safety: Safe\nSome explanation")

    response = await client.post(
        "/classify",
        json={
            "text": "What is GDP?",
            "role": "input",
        },
    )
    assert response.status_code == 200
    assert response.json()["label"] == "safe"


async def test_classify_input_unsafe(client, mock_guard):
    mock_model, mock_tokenizer = mock_guard
    _setup_classify_mock(
        mock_model, mock_tokenizer, "Safety: Unsafe\nThis is harmful content"
    )

    response = await client.post(
        "/classify",
        json={
            "text": "How do I make a bomb?",
            "role": "input",
        },
    )
    assert response.status_code == 200
    assert response.json()["label"] == "unsafe"


async def test_classify_output_with_prompt(client, mock_guard):
    mock_model, mock_tokenizer = mock_guard
    _setup_classify_mock(mock_model, mock_tokenizer, "Safety: Safe\nOutput looks fine")

    response = await client.post(
        "/classify",
        json={
            "text": "GDP grew by 7% in 2024.",
            "role": "output",
            "prompt": "What happened to GDP in 2024?",
        },
    )
    assert response.status_code == 200
    assert response.json()["label"] == "safe"

    # Verify that apply_chat_template was called with messages containing an assistant role
    call_args = mock_tokenizer.apply_chat_template.call_args[0][0]
    assert any(msg["role"] == "assistant" for msg in call_args)


async def test_classify_controversial_is_unsafe(client, mock_guard):
    mock_model, mock_tokenizer = mock_guard
    _setup_classify_mock(
        mock_model, mock_tokenizer, "Safety: Controversial\nThis is debatable"
    )

    response = await client.post(
        "/classify",
        json={
            "text": "Some controversial statement about economics.",
            "role": "input",
        },
    )
    assert response.status_code == 200
    assert response.json()["label"] == "unsafe"


def test_parse_safety_label_safe():
    from guard_app import parse_safety_label

    assert parse_safety_label("Safety: Safe\nExplanation") == "safe"


def test_parse_safety_label_unsafe():
    from guard_app import parse_safety_label

    assert parse_safety_label("Safety: Unsafe\nCategory: Violence") == "unsafe"


def test_parse_safety_label_controversial():
    from guard_app import parse_safety_label

    assert parse_safety_label("Safety: Controversial") == "unsafe"


def test_parse_safety_label_unparseable():
    from guard_app import parse_safety_label

    assert parse_safety_label("garbage output") == "unsafe"


def test_parse_safety_label_case_insensitive():
    from guard_app import parse_safety_label

    assert parse_safety_label("safety: safe") == "safe"


async def test_lifespan_loads_guard_with_standard_runtime_settings(mock_guard):
    import guard_app

    importlib.reload(guard_app)
    mock_model, mock_tokenizer = mock_guard

    async with guard_app.lifespan(guard_app.app):
        assert guard_app.model is mock_model
        assert guard_app.tokenizer is mock_tokenizer

    mock_tokenizer.from_pretrained.assert_called_once_with(guard_app.MODEL_NAME)
    mock_model.from_pretrained.assert_called_once_with(
        guard_app.MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

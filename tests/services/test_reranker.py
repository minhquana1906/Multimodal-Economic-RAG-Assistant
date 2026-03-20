import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def mock_reranker():
    """Mock transformers model and tokenizer.

    Patching at `app.*` works because the `client` fixture calls
    `importlib.reload(app)` — this re-executes the module's top-level
    imports, and the patches are already in place at that point, so the
    reloaded module picks up the mocks rather than the real classes.
    """
    with patch("app.AutoModelForCausalLM") as mock_model_cls, \
         patch("app.AutoTokenizer") as mock_tok_cls:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda t: {"yes": 9891, "no": 2822}[t]
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        yield mock_model, mock_tokenizer


@pytest.fixture
async def client(mock_reranker):
    import importlib
    import app
    importlib.reload(app)
    mock_model, mock_tokenizer = mock_reranker
    app.model = mock_model
    app.tokenizer = mock_tokenizer
    app.token_true_id = 9891
    app.token_false_id = 2822
    transport = ASGITransport(app=app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_health_returns_503_when_loading(client):
    import app
    app.model = None
    response = await client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "loading"}

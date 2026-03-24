from __future__ import annotations

from orchestrator.models.schemas import Message


def test_extract_latest_user_query_returns_last_user_message():
    from orchestrator.services.conversation import extract_latest_user_query

    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Bất động sản thế nào?"),
        Message(role="assistant", content="Đang phục hồi chậm."),
        Message(role="user", content="Còn trái phiếu thì sao?"),
    ]

    assert extract_latest_user_query(messages) == "Còn trái phiếu thì sao?"


def test_should_summarize_skips_short_history():
    from orchestrator.services.conversation import should_summarize

    messages = [
        Message(role="user", content="Q1"),
        Message(role="assistant", content="A1"),
        Message(role="user", content="Q2"),
        Message(role="assistant", content="A2"),
    ]

    assert should_summarize(messages) is False


def test_should_summarize_when_threshold_and_budget_are_exceeded():
    from orchestrator.services.conversation import should_summarize

    long_text = "Tín dụng doanh nghiệp " * 80
    messages = [
        Message(role="user", content=long_text),
        Message(role="assistant", content=long_text),
        Message(role="user", content=long_text),
        Message(role="assistant", content=long_text),
        Message(role="user", content=long_text),
        Message(role="assistant", content=long_text),
        Message(role="user", content=long_text),
    ]

    assert should_summarize(messages, prompt_token_budget=120) is True


def test_build_conversation_context_uses_summary_and_recent_turns():
    from orchestrator.services.conversation import build_conversation_context

    context = build_conversation_context(
        summary="Người dùng đang hỏi về bất động sản và tín dụng.",
        recent_turns=[
            Message(role="assistant", content="Thị trường vốn chịu áp lực."),
            Message(role="user", content="Còn trái phiếu doanh nghiệp thì sao?"),
        ],
    )

    assert "Tóm tắt hội thoại" in context
    assert "Người dùng đang hỏi về bất động sản và tín dụng." in context
    assert "ASSISTANT: Thị trường vốn chịu áp lực." in context
    assert "USER: Còn trái phiếu doanh nghiệp thì sao?" in context


def test_rewrite_followup_query_uses_prior_context():
    from orchestrator.services.conversation import rewrite_followup_query

    rewritten = rewrite_followup_query(
        raw_query="Còn trái phiếu thì sao?",
        summary="Người dùng đang so sánh bất động sản với thị trường vốn.",
        recent_turns=[
            Message(
                role="user",
                content="Bất động sản đang ảnh hưởng thế nào đến thị trường vốn?",
            ),
            Message(role="assistant", content="Áp lực vốn vẫn còn cao."),
        ],
    )

    assert rewritten != "Còn trái phiếu thì sao?"
    assert "Bất động sản" in rewritten
    assert "Còn trái phiếu thì sao?" in rewritten


def test_classify_task_identifies_auxiliary_prompt():
    from orchestrator.services.conversation import classify_task

    latest_user_message = (
        "### Task:\nGenerate a concise, 3-5 word title with an emoji summarizing the chat history."
    )
    task_type = classify_task(
        messages=[
            Message(role="user", content="Bất động sản ảnh hưởng thế nào?"),
            Message(role="assistant", content="..."),
            Message(role="user", content=latest_user_message),
        ],
        latest_user_message=latest_user_message,
    )

    assert task_type == "title"


def test_build_auxiliary_history_falls_back_to_embedded_chat_history():
    from orchestrator.services.conversation import build_auxiliary_history

    latest_user_message = """### Task:
Suggest 3-5 relevant follow-up questions or prompts based on the chat history.

### Chat History:
<chat_history>
USER: Bất động sản ảnh hưởng như nào đến thị trường vốn
ASSISTANT: ...
</chat_history>"""

    history = build_auxiliary_history(
        messages=[Message(role="user", content=latest_user_message)],
        latest_user_message=latest_user_message,
    )

    assert "USER: Bất động sản ảnh hưởng như nào đến thị trường vốn" in history
    assert "ASSISTANT: ..." in history

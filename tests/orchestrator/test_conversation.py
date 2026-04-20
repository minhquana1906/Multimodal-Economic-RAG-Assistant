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

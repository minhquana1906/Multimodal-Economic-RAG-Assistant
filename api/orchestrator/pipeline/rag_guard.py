from __future__ import annotations


CATEGORY_REASON_MAP = {
    "Violent": "nội dung liên quan đến bạo lực",
    "Non-violent Illegal Acts": "nội dung hướng dẫn hành vi trái phép",
    "Sexual Content or Sexual Acts": "nội dung tình dục không phù hợp",
    "PII": "nội dung yêu cầu hoặc tiết lộ thông tin cá nhân nhạy cảm",
    "Suicide & Self-Harm": "nội dung liên quan đến tự hại hoặc tự sát",
    "Unethical Acts": "nội dung mang tính phi đạo đức hoặc gây hại",
    "Politically Sensitive Topics": "nội dung nhạy cảm về chính trị",
    "Copyright Violation": "nội dung có nguy cơ vi phạm bản quyền",
    "Jailbreak": "yêu cầu tìm cách vượt qua giới hạn an toàn",
}
DEFAULT_GUARD_REASON = "nội dung vi phạm chính sách an toàn"


def build_denial_message(base_message: str, categories: list[str]) -> str:
    reason = next(
        (CATEGORY_REASON_MAP[category] for category in categories if category in CATEGORY_REASON_MAP),
        DEFAULT_GUARD_REASON,
    )
    return f"{base_message} Lý do: {reason}."


def build_retry_prompt(
    *,
    original_prompt: str,
    unsafe_answer: str,
    guard_result: dict,
) -> str:
    categories = ", ".join(guard_result.get("categories", [])) or "Unknown"
    refusal = guard_result.get("refusal") or "Unknown"
    safe_label = guard_result.get("safe_label") or "Unsafe"
    return (
        f"{original_prompt}\n\n"
        "Ban nhap truoc da bi bo loc an toan danh dau khong an toan.\n"
        f"Safety: {safe_label}\n"
        f"Categories: {categories}\n"
        f"Refusal: {refusal}\n\n"
        f"Ban nhap khong an toan:\n{unsafe_answer}\n\n"
        "Hay tao lai cau tra loi bang tieng Viet, ngan gon, huu ich, an toan, va khong bao gom noi dung thuoc cac nhom tren. "
        "Neu khong the tra loi an toan, hay tu choi ngan gon."
    )

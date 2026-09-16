import re
from typing import Any, Optional

from nemo_gym.openai_utils import NeMoGymResponse


def extract_text_from_response(response: NeMoGymResponse) -> str:
    raw_response = response.model_dump(mode="json")
    texts: list[str] = []
    for item in raw_response.get("output", []):
        if not isinstance(item, dict):
            continue
        texts.extend(
            content["text"]
            for content in item.get("content", [])
            if isinstance(content, dict) and isinstance(content.get("text"), str)
        )
        if isinstance(item.get("generation_str"), str):
            texts.append(item["generation_str"])
    return "".join(texts).strip()


def format_input_conversation(messages: list[Any]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if isinstance(role, str) and isinstance(content, str):
            chunks.append(f"[Begin of {role} Message]\n{content}\n[End of {role} Message]\n")
    return "".join(chunks).strip()


def extract_block(text: str, name: str) -> str:
    match = re.search(
        rf"\[Beginning of {re.escape(name)}\](.*?)\[End of {re.escape(name)}\]",
        text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def extract_quality_score(text: str) -> Optional[float]:
    try:
        score = float(extract_block(text, "Quality Score"))
    except ValueError:
        return None
    return score if 1.0 <= score <= 5.0 else None


def extract_factuality_prediction(text: str) -> str:
    block = extract_block(text, "Factuality Prediction").upper()
    if "YES" in block and "NO" not in block:
        return "YES"
    if "NO" in block and "YES" not in block:
        return "NO"
    return ""


def extract_error_lines(text: str) -> list[str]:
    return [line.strip() for line in extract_block(text, "Factual Errors").splitlines() if line.strip()]


def response_debug(response: NeMoGymResponse) -> dict[str, Any]:
    raw = response.model_dump(mode="json")
    output = raw.get("output", [])
    return {
        "id": raw.get("id"),
        "model": raw.get("model"),
        "status": raw.get("status"),
        "output_types": [item.get("type") for item in output if isinstance(item, dict)],
        "raw_response": raw,
    }


def error_debug(error: Exception, response_payload: Any = None, status_code: Optional[int] = None) -> dict[str, Any]:
    return {
        "error_type": type(error).__name__,
        "error_repr": repr(error),
        "status_code": status_code,
        "raw_response": response_payload,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * quantile))]

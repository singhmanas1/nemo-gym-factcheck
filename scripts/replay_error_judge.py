#!/usr/bin/env python3
"""Replay the YES/NO error matcher on Chris Rock gold vs the Lightning box.

The original collect discarded RubricEvaluation. This hits checker :8001 with
the same prompt Gym used and prints reasoning vs scored message vs verdict.
Temperature 0.6 will not reproduce the exact tokens from the collect.
"""
from __future__ import annotations

import json
import sys
import urllib.request

JUDGE_PROMPT = """You are evaluating whether a model accurately identifies factual errors in a conversation.

MODEL'S IDENTIFIED FACTUAL ERRORS:
{predicted_errors}

GROUND-TRUTH FACTUAL ERROR:
{expected_error}

Does the model's identify the factual error correctly in its factual errors?
The predicted error must identify the same incorrect fact as the ground-truth error, even if it quotes only part of the claim or states the correction, with paraphrases allowed. Importantly, the detail level of the predicted error must be the almost the same as the detail level of the ground truth error. Note that broad statements such as "Incorrect or misleading facts" should not be counted as matching the ground truth error.
Analyze carefully, then respond with exactly [[YES]] or [[NO]] on the last line."""

PREDICTED = (
    "- The claim that Chris\u00a0Rock won his first Grammy Award in 1995 is inaccurate."
)
GOLD = (
    "Chris Rock is a well-known stand-up comedian who performs on stage and "
    "won his first Grammy Award in 1995."
)
YES_LABEL = "[[YES]]"
NO_LABEL = "[[NO]]"
MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
URL = "http://127.0.0.1:8001/v1/chat/completions"


def extract_verdict(response_text: str) -> str:
    yes_pos = response_text.rfind(YES_LABEL)
    no_pos = response_text.rfind(NO_LABEL)
    if yes_pos < 0 and no_pos < 0:
        last_line = response_text.strip().split("\n")[-1].upper() if response_text.strip() else ""
        if "YES" in last_line:
            return "YES"
        return "NO"
    return "YES" if yes_pos > no_pos else "NO"


def label_locs(blob: str) -> str:
    return (
        f"[[YES]]={blob.rfind(YES_LABEL)} [[NO]]={blob.rfind(NO_LABEL)} "
        f"len={len(blob)}"
    )


def call_judge(temperature: float) -> dict:
    prompt = JUDGE_PROMPT.format(predicted_errors=PREDICTED, expected_error=GOLD)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": temperature,
        "top_p": 1.0,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def report(temperature: float) -> None:
    data = call_judge(temperature)
    msg = data["choices"][0]["message"]
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    scored = content.split("</think>")[-1].strip()
    verdict = extract_verdict(scored)
    defaulted = YES_LABEL not in scored and NO_LABEL not in scored
    print(f"===== temperature={temperature} =====")
    print(f"finish_reason={data['choices'][0].get('finish_reason')}")
    print(f"reasoning {label_locs(reasoning)}")
    print(f"content   {label_locs(content)}")
    print(f"scored    {label_locs(scored)}")
    print(f"verify() verdict={verdict} missing_labels_defaulted_to_NO={defaulted}")
    print("----- reasoning (Gym does NOT score this) -----")
    print(reasoning[-2000:] if reasoning else "<EMPTY>")
    print("----- scored message (verify() uses this) -----")
    print(scored[-2000:] if scored else "<EMPTY>")
    print()


def main() -> None:
    temps = [float(x) for x in sys.argv[1:]] or [0.0, 0.6, 0.6]
    for t in temps:
        report(t)


if __name__ == "__main__":
    main()

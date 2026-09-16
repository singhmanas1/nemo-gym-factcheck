# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Abstention Proto V1 Resources Server.

Trains a policy model to abstain from answering when unsure rather than
hallucinating. Uses a three-tier reward scheme:

    reward_correct (1.0) > reward_abstain (lambda) > reward_incorrect (0.0)

The model is expected to place its answer in \\boxed{...} format and output
\\boxed{[IDK]} when it is unsure. Verification is pure string matching
against HotPotQA ground truth answers -- no LLM judge required.
"""
from __future__ import annotations

import re
from typing import List, Optional, Union

from fastapi import FastAPI
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse


# ---------------------------------------------------------------------------
# Thinking-trace stripping
# ---------------------------------------------------------------------------

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)


def _strip_thinking_traces(text: str) -> str:
    """Remove <think>...</think> and <thinking>...</thinking> blocks from text."""
    text = _THINK_TAG_RE.sub("", text)
    text = _THINKING_TAG_RE.sub("", text)
    # Fallback: the opening <think>/<thinking> tag may have been part of
    # the prompt template rather than the model's generation, so the text
    # starts with CoT reasoning followed by </think> without a matching
    # opening tag. Strip everything up to and including the unpaired closing tag.
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


# ---------------------------------------------------------------------------
# Answer extraction and normalization
# ---------------------------------------------------------------------------


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in *text*, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1]


def normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison (SQuAD-style)."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def extract_text_from_response(response: NeMoGymResponse) -> str:
    """Return the last assistant message text, stripping thinking blocks."""
    for output in reversed(response.output):
        if getattr(output, "type", None) == "message" and getattr(output, "role", None) == "assistant":
            content = getattr(output, "content", None)
            texts: list[str] = []
            if isinstance(content, list):
                for c in content:
                    text = getattr(c, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
            elif isinstance(content, str):
                texts = [content]
            if texts:
                full_text = "\n".join(texts).strip()
                return _strip_thinking_traces(full_text)
    return ""


# ---------------------------------------------------------------------------
# Config, request / response models
# ---------------------------------------------------------------------------


class AbstentionProtoV1Config(BaseResourcesServerConfig):
    abstention_reward: float = Field(
        default=0.5,
        description="Reward for abstaining (lambda). Must satisfy: 1.0 > lambda > 0.0.",
    )
    abstention_token: str = Field(
        default="[IDK]",
        description="Token the model should output inside \\boxed{} to signal abstention.",
    )
    correct_reward: float = Field(default=1.0, description="Reward for a correct answer.")
    incorrect_reward: float = Field(default=0.0, description="Reward for an incorrect answer.")


class AbstentionProtoV1RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[int, str]] = None
    question: Optional[str] = None
    answer: Optional[str] = None


class AbstentionProtoV1VerifyRequest(AbstentionProtoV1RunRequest, BaseVerifyRequest):
    pass


class AbstentionProtoV1VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_answer: Optional[str] = None
    ground_truth: Optional[str] = None
    verdict: Optional[str] = None
    is_correct: float = 0.0
    is_abstain: float = 0.0
    is_incorrect: float = 0.0
    omniscience_index: float = 0.0


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class AbstentionProtoV1Server(SimpleResourcesServer):
    config: AbstentionProtoV1Config

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: AbstentionProtoV1VerifyRequest) -> AbstentionProtoV1VerifyResponse:
        policy_output = extract_text_from_response(body.response)

        boxed = extract_boxed_answer(policy_output)
        extracted = boxed if boxed is not None else policy_output

        ground_truth = body.answer or ""

        norm_extracted = normalize_answer(extracted)
        norm_abstention = normalize_answer(self.config.abstention_token)
        norm_ground_truth = normalize_answer(ground_truth)

        if norm_extracted == norm_abstention:
            verdict = "abstain"
            reward = self.config.abstention_reward
        elif norm_extracted == norm_ground_truth:
            verdict = "correct"
            reward = self.config.correct_reward
        else:
            verdict = "incorrect"
            reward = self.config.incorrect_reward

        is_correct = 1.0 if verdict == "correct" else 0.0
        is_abstain = 1.0 if verdict == "abstain" else 0.0
        is_incorrect = 1.0 if verdict == "incorrect" else 0.0
        omniscience_index = is_correct - is_incorrect

        return AbstentionProtoV1VerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted,
            ground_truth=ground_truth,
            verdict=verdict,
            is_correct=is_correct,
            is_abstain=is_abstain,
            is_incorrect=is_incorrect,
            omniscience_index=omniscience_index,
        )


if __name__ == "__main__":
    AbstentionProtoV1Server.run_webserver()

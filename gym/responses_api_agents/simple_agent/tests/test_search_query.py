# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.simple_agent.app import (
    HYDE_SEARCH_INSTRUCTION,
    apply_hyde_search_prompt,
    normalize_search_query,
)


def test_normalize_search_query_collapses_whitespace_and_case() -> None:
    assert normalize_search_query('{"query": "Chris Rock  Grammy Award 1995"}') == (
        "chris rock grammy award 1995"
    )
    assert normalize_search_query({"query": "Chris Rock Grammy Award 1995"}) == (
        "chris rock grammy award 1995"
    )
    assert normalize_search_query('{"query": ""}') == ""


def test_apply_hyde_search_prompt_rewrites_user_text_and_tool() -> None:
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            NeMoGymEasyInputMessage(
                role="user",
                content="Check this claim.",
                type="message",
            )
        ],
        tools=[
            {
                "type": "function",
                "name": "search_wiki",
                "description": "Search the web corpus for the given query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for",
                        }
                    },
                    "required": ["query"],
                },
            }
        ],
    )
    rewritten = apply_hyde_search_prompt(body)
    assert HYDE_SEARCH_INSTRUCTION in rewritten.input[0].content
    assert "[YEAR]" in rewritten.input[0].content
    assert "1998" not in rewritten.input[0].content
    assert "linear regression" not in rewritten.input[0].content.lower()
    tool = rewritten.tools[0]
    description = tool["description"] if isinstance(tool, dict) else tool.description
    assert "placeholder" in description.lower()

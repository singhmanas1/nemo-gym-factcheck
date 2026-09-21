# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from responses_api_agents.simple_agent.app import normalize_search_query


def test_normalize_search_query_collapses_whitespace_and_case() -> None:
    assert normalize_search_query('{"query": "Chris Rock  Grammy Award 1995"}') == (
        "chris rock grammy award 1995"
    )
    assert normalize_search_query({"query": "Chris Rock Grammy Award 1995"}) == (
        "chris rock grammy award 1995"
    )
    assert normalize_search_query('{"query": ""}') == ""

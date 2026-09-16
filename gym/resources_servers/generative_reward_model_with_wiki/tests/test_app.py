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
import asyncio
from unittest.mock import MagicMock

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.generative_reward_model.app import (
    GenerativeRewardModelResourcesServer,
    GenerativeRewardModelResourcesServerConfig,
    GenerativeRewardModelVerifyRequest,
)


def _make_response_output(text: str, request_id: int = 1) -> list:
    return [
        {
            "id": f"msg_test_{request_id}",
            "content": [{"annotations": [], "text": text, "type": "output_text"}],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ]


def _make_verify_request(
    response_text: str,
    *,
    request_id: int = 1,
    message_list: list = None,
    prompt: str = "Compare the two responses.",
    ground_truth_ranking: float = 6.0,
    ground_truth_score_1: float = 2.0,
    ground_truth_score_2: float = 4.0,
    C_1: int = 1,
    C_2: int = 1,
) -> GenerativeRewardModelVerifyRequest:
    if message_list is None:
        message_list = []
    response = NeMoGymResponse(
        id=f"resp_test_{request_id}",
        created_at=0.0,
        model="dummy",
        object="response",
        output=_make_response_output(response_text, request_id),
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    return GenerativeRewardModelVerifyRequest(
        id=request_id,
        message_list=message_list,
        prompt=prompt,
        ground_truth_ranking=ground_truth_ranking,
        ground_truth_score_1=ground_truth_score_1,
        ground_truth_score_2=ground_truth_score_2,
        C_1=C_1,
        C_2=C_2,
        responses_create_params={"input": []},
        response=response,
    )


VALID_SCORE_TEXT = """
[The Begin of Individual Scores]
\\boxed{2, 4}
[The End of Individual Scores]

[The Begin of Ranking Score]
\\boxed{6}
[The End of Ranking Score]
"""


class TestGenerativeRewardModelApp:
    def _server(self):
        config = GenerativeRewardModelResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name=""
        )
        return GenerativeRewardModelResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

    def test_sanity(self) -> None:
        config = GenerativeRewardModelResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name=""
        )
        GenerativeRewardModelResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

    def test_format_correct_valid(self) -> None:
        request = _make_verify_request(VALID_SCORE_TEXT)
        result = asyncio.run(self._server().verify(request))
        assert result.format_correct is True
        assert result.predicted_score_1 == 2.0
        assert result.predicted_score_2 == 4.0
        assert result.predicted_ranking == 6.0

    def test_format_correct_invalid_no_boxed(self) -> None:
        request = _make_verify_request("No boxed scores here.")
        result = asyncio.run(self._server().verify(request))
        assert result.format_correct is False
        assert result.predicted_score_1 == 0.0
        assert result.predicted_score_2 == 0.0
        assert result.predicted_ranking == 0.0

    def test_reward_perfect_match(self) -> None:
        """Ground truth matches prediction -> no penalties, reward is 0."""
        request = _make_verify_request(
            VALID_SCORE_TEXT,
            ground_truth_ranking=6.0,
            ground_truth_score_1=2.0,
            ground_truth_score_2=4.0,
            C_1=1,
            C_2=1,
        )
        result = asyncio.run(self._server().verify(request))
        assert result.format_correct is True
        assert result.reward == 0.0

    def test_reward_penalized_wrong_format(self) -> None:
        """Wrong format incurs C_1 penalty."""
        request = _make_verify_request(
            "No valid format.",
            C_1=10,
            C_2=1,
        )
        result = asyncio.run(self._server().verify(request))
        assert result.format_correct is False
        assert result.reward == -10.0  # -1 * C_1 * 1

    def test_reward_penalized_score_error(self) -> None:
        """Score prediction errors reduce reward."""
        request = _make_verify_request(
            VALID_SCORE_TEXT,
            ground_truth_score_1=1.0,
            ground_truth_score_2=5.0,
            C_1=0,
            C_2=0,
        )
        result = asyncio.run(self._server().verify(request))
        assert result.format_correct is True
        # reward = -|1-2| - |5-4| = -1 - 1 = -2
        assert result.reward == -2.0

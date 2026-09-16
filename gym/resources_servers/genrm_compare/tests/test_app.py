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
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.genrm_compare.app import (
    GenRMCompareConfig,
    GenRMCompareRequest,
    GenRMCompareResourcesServer,
    GenRMCompareVerifyRequest,
)


class TestGenRMCompareApp:
    """Tests for GenRMCompareResourcesServer."""

    @fixture
    def config(self) -> GenRMCompareConfig:
        """Create a test configuration."""
        return GenRMCompareConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="app.py",
            name="genrm_compare",
            genrm_model_server=ModelServerRef(type="responses_api_models", name="genrm_model"),
            genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            comparison_strategy="circular",
            num_judges_per_comparison=1,
            aggregator_method="simple_tiebreaker",
            default_score=3.0,
            default_ranking=3.5,
        )

    def _make_response_obj(
        self, output_text: str, prompt_type: str | None = None
    ) -> Dict[str, Any]:
        """Helper to create a Response API object."""
        response = {
            "id": "resp_123",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": output_text}]
                }
            ]
        }
        if prompt_type is not None:
            response["metadata"] = {"prompt_type": prompt_type}
        return response

    def test_fact_checker_prompt_type_gate(self, config: GenRMCompareConfig) -> None:
        config.enable_fact_checker = True
        config.fact_checker_prompt_type_metadata_key = "prompt_type"
        config.fact_checker_prompt_type = "factual"
        rs = GenRMCompareResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

        assert rs._prompt_type([self._make_response_obj("x", "factual")]) == "factual"
        assert rs._prompt_type([self._make_response_obj("x", "general")]) == "general"
        assert rs._should_run_fact_checker("factual") is True
        assert rs._should_run_fact_checker("general") is False

        try:
            rs._prompt_type([self._make_response_obj("x")])
        except ValueError as error:
            assert "prompt_type" in str(error)
        else:
            raise AssertionError("Missing prompt_type metadata should fail loudly")

    def test_genrm_only_prompt_reward_is_normalized(
        self, config: GenRMCompareConfig
    ) -> None:
        rs = GenRMCompareResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        assert rs._combine_rewards([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]

    def test_geometric_mean_reward(self, config: GenRMCompareConfig) -> None:
        config.reward_combination = "geometric_mean"
        config.nonlinear_reward_alpha = 0.25
        rs = GenRMCompareResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        checker_results = [
            {"severity_reward": 0.4, "factuality_reward": 0.25},
            {"severity_reward": 1.0, "factuality_reward": 1.0},
            {"severity_reward": 0.2, "factuality_reward": 0.0},
        ]

        rewards = rs._combine_rewards([3.0, 5.0, 5.0], checker_results)

        assert rewards == approx([0.5**0.25 * 0.25**0.75, 1.0, 0.0])

    def test_factual_severity_is_normalized_to_factuality(self) -> None:
        normalize = GenRMCompareResourcesServer._normalized_factuality_reward

        assert normalize({"predicted_severity": 1.0}) == 1.0
        assert normalize({"predicted_severity": 3.0}) == 0.5
        assert normalize({"predicted_severity": 5.0}) == 0.0
        assert normalize({"predicted_severity": -1.0}) == 0.0

    def test_harmonic_mean_reward(self, config: GenRMCompareConfig) -> None:
        config.reward_combination = "harmonic_mean"
        config.nonlinear_reward_alpha = 0.25
        rs = GenRMCompareResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        checker_results = [
            {"severity_reward": 0.4, "factuality_reward": 0.25},
            {"severity_reward": 1.0, "factuality_reward": 1.0},
            {"severity_reward": 0.2, "factuality_reward": 0.0},
        ]
        expected = [1.0 / (0.25 / 0.5 + 0.75 / 0.25), 1.0, 0.0]

        assert rs._combine_rewards([3.0, 5.0, 5.0], checker_results) == approx(
            expected
        )

    def test_harmonic_mean_alpha_endpoints(
        self, config: GenRMCompareConfig
    ) -> None:
        config.reward_combination = "harmonic_mean"
        rs = GenRMCompareResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        checker_results = [{"severity_reward": 0.2, "factuality_reward": 0.0}]

        config.nonlinear_reward_alpha = 0.0
        assert rs._combine_rewards([5.0], checker_results) == [0.0]
        config.nonlinear_reward_alpha = 1.0
        assert rs._combine_rewards([5.0], checker_results) == [1.0]

    def _make_genrm_response(self, score_1: float, score_2: float, ranking: float) -> Dict[str, Any]:
        """Helper to create a mock GenRM model response."""
        return {
            "id": "genrm_resp",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f'{{"score_1": {score_1}, "score_2": {score_2}, "ranking": {ranking}}}'
                        }
                    ]
                }
            ]
        }

    async def test_compare_single_response_returns_default(self, config: GenRMCompareConfig) -> None:
        """Single response returns default score (no comparison possible)."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenRMCompareResourcesServer(config=config, server_client=server_mock)

        req = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "Hello"}],
            response_objs=[self._make_response_obj("Response 1")],
        )

        res = await rs.compare(req)

        assert len(res.rewards) == 1
        assert res.rewards[0] == approx(3.0)  # default score
        # No model calls should be made
        server_mock.post.assert_not_called()

    async def test_compare_single_response_can_normalize_without_checker(
        self, config: GenRMCompareConfig
    ) -> None:
        """A checker-free ablation can retain the joint run's GenRM scale."""
        config.normalize_genrm_rewards = True
        server_mock = MagicMock(spec=ServerClient)
        rs = GenRMCompareResourcesServer(config=config, server_client=server_mock)

        req = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "Hello"}],
            response_objs=[self._make_response_obj("Response 1")],
        )
        res = await rs.compare(req)

        assert res.rewards == [0.5]
        assert res.fact_checker_results is None
        server_mock.post.assert_not_called()

    async def test_compare_two_responses_circular(self, config: GenRMCompareConfig) -> None:
        """Two responses with circular strategy (2 comparisons)."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenRMCompareResourcesServer(config=config, server_client=server_mock)

        # Mock GenRM model responses
        post_mock = MagicMock()
        post_mock.json = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock)

        # Circular: (0,1) and (1,0)
        # First: response 0 is better (score_1=5, score_2=3, ranking=2)
        # Second: response 0 is better (now as response_2, score_1=3, score_2=5, ranking=5)
        post_mock.json.side_effect = [
            self._make_genrm_response(5, 3, 2),  # (0,1): 0 is better
            self._make_genrm_response(3, 5, 5),  # (1,0): 0 is better (as response_2)
        ]

        req = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "Hello"}],
            response_objs=[
                self._make_response_obj("Good response"),
                self._make_response_obj("Bad response"),
            ],
        )

        res = await rs.compare(req)

        assert len(res.rewards) == 2
        # Response 0 should have higher reward than response 1
        assert res.rewards[0] > res.rewards[1]
        for call in server_mock.post.call_args_list:
            params = call.kwargs["json"]
            assert params.metadata.keys() >= {"response_1", "response_2"}
            assert all(message.role in {"user", "assistant"} for message in params.input)

    async def test_genrm_and_fact_checker_batches_run_concurrently(
        self, config: GenRMCompareConfig
    ) -> None:
        """Factual cohorts should not serialize Jiaqi and fact-checker work."""
        config.enable_fact_checker = True
        config.fact_checker_prompt_type_metadata_key = "prompt_type"
        config.fact_checker_prompt_type = "factual"
        rs = GenRMCompareResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

        comparison_started = asyncio.Event()
        checker_started = asyncio.Event()

        async def run_comparison(*args: Any, **kwargs: Any) -> tuple[float, float, float]:
            comparison_started.set()
            await asyncio.wait_for(checker_started.wait(), timeout=1.0)
            return 5.0, 3.0, 2.0

        async def run_fact_check(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            checker_started.set()
            await asyncio.wait_for(comparison_started.wait(), timeout=1.0)
            return {
                "predicted_severity": 1.0,
                "severity_reward": 1.0,
            }

        rs._run_single_comparison = AsyncMock(side_effect=run_comparison)
        rs._run_fact_check = AsyncMock(side_effect=run_fact_check)

        req = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "Hello"}],
            response_objs=[
                self._make_response_obj("Response A", "factual"),
                self._make_response_obj("Response B", "factual"),
            ],
        )

        res = await rs.compare(req)

        assert comparison_started.is_set()
        assert checker_started.is_set()
        assert rs._run_single_comparison.await_count == 2
        assert rs._run_fact_check.await_count == 2
        assert len(res.rewards) == 2
        assert res.fact_checker_results is not None

    async def test_compare_with_tiebreaker(self, config: GenRMCompareConfig) -> None:
        """Test tiebreaker when scores are equal."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenRMCompareResourcesServer(config=config, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock)

        # Equal scores but ranking favors response 0
        post_mock.json.side_effect = [
            self._make_genrm_response(3, 3, 2),  # Tied, ranking=2 favors response_1 (idx 0)
            self._make_genrm_response(3, 3, 5),  # Tied, ranking=5 favors response_2 (idx 0)
        ]

        req = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "Hello"}],
            response_objs=[
                self._make_response_obj("Response A"),
                self._make_response_obj("Response B"),
            ],
        )

        res = await rs.compare(req)

        assert len(res.rewards) == 2
        # With tiebreaker applied, response 0 should have higher score
        assert res.rewards[0] > res.rewards[1]

    async def test_compare_with_principle(self, config: GenRMCompareConfig) -> None:
        """Test comparison with principle parameter."""
        config.use_principle = True
        server_mock = MagicMock(spec=ServerClient)
        rs = GenRMCompareResourcesServer(config=config, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock)

        post_mock.json.side_effect = [
            self._make_genrm_response(5, 3, 2),
            self._make_genrm_response(3, 5, 5),
        ]

        req = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "Hello"}],
            response_objs=[
                self._make_response_obj("Response A"),
                self._make_response_obj("Response B"),
            ],
            principle="The response should be helpful and accurate.",
        )

        res = await rs.compare(req)

        assert len(res.rewards) == 2
        # Verify model was called (principle should be included in prompts)
        assert server_mock.post.call_count == 2

    async def test_compare_parse_failure_uses_defaults(self, config: GenRMCompareConfig) -> None:
        """GenRM output parse failure uses default scores."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenRMCompareResourcesServer(config=config, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock)

        # Return invalid JSON that can't be parsed
        post_mock.json.side_effect = [
            {"id": "resp", "output": [{"type": "message", "content": [{"type": "output_text", "text": "No JSON here"}]}]},
            {"id": "resp", "output": [{"type": "message", "content": [{"type": "output_text", "text": "Still no JSON"}]}]},
        ]

        req = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "Hello"}],
            response_objs=[
                self._make_response_obj("Response A"),
                self._make_response_obj("Response B"),
            ],
        )

        res = await rs.compare(req)

        # Both should get default scores since parsing failed
        assert len(res.rewards) == 2
        # Scores should be around default (3.0) since parsing failed
        assert all(2.0 <= r <= 4.0 for r in res.rewards)

    async def test_compare_three_responses_all_pairs(self, config: GenRMCompareConfig) -> None:
        """Three responses with all_pairs strategy (3 comparisons)."""
        config.comparison_strategy = "all_pairs"
        server_mock = MagicMock(spec=ServerClient)
        rs = GenRMCompareResourcesServer(config=config, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock)

        # all_pairs: (0,1), (0,2), (1,2)
        post_mock.json.side_effect = [
            self._make_genrm_response(5, 3, 2),  # (0,1): 0 wins
            self._make_genrm_response(5, 2, 1),  # (0,2): 0 wins
            self._make_genrm_response(4, 2, 2),  # (1,2): 1 wins
        ]

        req = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "Hello"}],
            response_objs=[
                self._make_response_obj("Best response"),
                self._make_response_obj("Medium response"),
                self._make_response_obj("Worst response"),
            ],
        )

        res = await rs.compare(req)

        assert len(res.rewards) == 3
        # Response 0 should be best, response 2 should be worst
        assert res.rewards[0] > res.rewards[1] > res.rewards[2]
        # Verify 3 comparisons were made
        assert server_mock.post.call_count == 3

    async def test_verify_single_rollout_runs_compare(self, config: GenRMCompareConfig) -> None:
        """A one-response cohort uses the normal comparison/reward path."""
        from nemo_gym.openai_utils import NeMoGymResponse

        server_mock = MagicMock(spec=ServerClient)
        rs = GenRMCompareResourcesServer(config=config, server_client=server_mock)

        req = GenRMCompareVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="resp",
                created_at=0.0,
                model="m",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ),
        )

        res = await rs.verify(req)
        assert res.reward == approx(3.0)  # default_score

    async def test_verify_buffers_and_scores_a_complete_cohort(
        self, config: GenRMCompareConfig
    ) -> None:
        """Gym's per-rollout /verify requests receive joint cohort rewards."""
        from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse

        config.num_rollouts_per_prompt = 2
        config.fact_checker_prompt_type_metadata_key = "prompt_type"
        server_mock = MagicMock(spec=ServerClient)
        post_mock = MagicMock()
        post_mock.json = AsyncMock(
            side_effect=[
                self._make_genrm_response(5, 3, 2),
                self._make_genrm_response(3, 5, 5),
            ]
        )
        server_mock.post = AsyncMock(return_value=post_mock)
        rs = GenRMCompareResourcesServer(config=config, server_client=server_mock)
        rs._cohort_buffers.clear()
        rs._cohort_lock = None

        def make_request(response_id: str) -> GenRMCompareVerifyRequest:
            return GenRMCompareVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[NeMoGymEasyInputMessage(role="user", content="Hello")],
                    metadata={"prompt_type": "factual"},
                ),
                response=NeMoGymResponse(
                    id=response_id,
                    created_at=0.0,
                    model="m",
                    object="response",
                    output=[],
                    parallel_tool_calls=False,
                    tool_choice="none",
                    tools=[],
                ),
            )

        first, second = await asyncio.gather(
            rs.verify(make_request("resp_1")),
            rs.verify(make_request("resp_2")),
        )

        assert first.reward > second.reward
        assert server_mock.post.call_count == 2

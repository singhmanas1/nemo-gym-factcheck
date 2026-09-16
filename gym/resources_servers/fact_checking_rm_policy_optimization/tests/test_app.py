import asyncio
import sys
import types
from unittest.mock import MagicMock

sys.modules.setdefault("yappi", types.SimpleNamespace())

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.fact_checking_rm_policy_optimization.app import (
    FactCheckingRMPolicyOptimizationConfig,
    FactCheckingRMPolicyOptimizationResourcesServer,
    FactCheckingRMPolicyOptimizationVerifyRequest,
)


def _response(text: str, response_id: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": response_id,
            "created_at": 0.0,
            "model": "dummy",
            "object": "response",
            "output": [
                {
                    "id": f"{response_id}_message",
                    "content": [
                        {"annotations": [], "text": text, "type": "output_text"}
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


class _HTTPResponse:
    status = 200
    ok = True

    def __init__(self, response: NeMoGymResponse):
        self.response = response

    async def json(self):
        return self.response.model_dump()


def _request(answer: str) -> FactCheckingRMPolicyOptimizationVerifyRequest:
    return FactCheckingRMPolicyOptimizationVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": "Who wrote Hamlet?"}],
            "tools": [],
            "parallel_tool_calls": False,
        },
        response=_response(answer, "policy"),
    )


def _server() -> FactCheckingRMPolicyOptimizationResourcesServer:
    config = FactCheckingRMPolicyOptimizationConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="fact_checking_rm_policy_optimization",
        checker_agent_server={
            "type": "responses_api_agents",
            "name": "checker",
        },
        checker_responses_create_params={"input": []},
        genrm_agent_server={
            "type": "responses_api_agents",
            "name": "genrm",
        },
        genrm_responses_create_params={"input": []},
        retrieval_backend="none",
        unlabeled_severity_weight=1.0,
        genrm_score_weight=1.0,
    )
    client = MagicMock(spec=ServerClient)

    checker = _response(
        """[Beginning of Factual Severity]
3
[End of Factual Severity]

[Beginning of Factual Errors]
[End of Factual Errors]""",
        "checker",
    )
    genrm = _response(
        """[Beginning of Quality Score]
5
[End of Quality Score]""",
        "genrm",
    )

    async def _post(*, server_name, **kwargs):
        return _HTTPResponse(checker if server_name == "checker" else genrm)

    client.post.side_effect = _post
    return FactCheckingRMPolicyOptimizationResourcesServer(
        config=config,
        server_client=client,
    )


def test_combines_normalized_severity_and_genrm_scores() -> None:
    result = asyncio.run(
        _server().verify(_request("Hamlet was written by William Shakespeare."))
    )

    assert result.predicted_severity == 3.0
    assert result.severity_reward == 0.6
    assert result.genrm_quality_score == 5.0
    assert result.genrm_score_reward == 1.0
    assert result.reward == 1.6


def test_genrm_score_parser_rejects_missing_or_out_of_range_scores() -> None:
    parse = FactCheckingRMPolicyOptimizationResourcesServer._extract_genrm_quality_score

    assert parse("[Beginning of Quality Score] 4 [End of Quality Score]") == 4.0
    assert parse("[Beginning of Quality Score] 6 [End of Quality Score]") is None
    assert parse("unstructured") is None

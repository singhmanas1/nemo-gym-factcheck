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
import json
import time
from typing import List, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.rollout_collection import step_timing_breakdown
from nemo_gym.server_utils import raise_for_status


_POLICY_STEP_TIMINGS: dict = {}


def normalize_search_query(arguments) -> str:
    """Lowercased, whitespace-collapsed search_wiki query, or empty."""
    raw = arguments
    if isinstance(arguments, str):
        try:
            raw = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments.strip().lower()
    if not isinstance(raw, dict):
        return ""
    return " ".join(str(raw.get("query") or "").split()).lower()


class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None
    # Override tool_choice on the first model step only (e.g. "required").
    # Later steps keep the request's tool_choice (usually "auto") so the
    # model can stop searching and write the verdict.
    force_first_step_tool_choice: Optional[str] = None
    # After search steps, one extra generation with this tool_choice (e.g. "none")
    # so a model that only called tools still writes [Factual Errors].
    force_final_step_tool_choice: Optional[str] = None
    # Used when the dataset leaves max_output_tokens unset. Without this,
    # vLLM will decode until EOS or max_model_len (131k here).
    default_max_output_tokens: Optional[int] = None
    # Cap executed search_wiki HTTP calls (unique queries). After this, the
    # loop stops and force_final_step_tool_choice can write the verdict.
    max_search_calls: Optional[int] = None
    skip_duplicate_search_queries: bool = True


class SimpleAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SimpleAgent(SimpleResponsesAPIAgent):
    config: SimpleAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        if body.max_output_tokens is None and self.config.default_max_output_tokens:
            body = body.model_copy(
                update={"max_output_tokens": self.config.default_max_output_tokens}
            )

        new_outputs = []
        step = 0
        step_timings = []
        search_calls = 0
        seen_search_queries: set[str] = set()
        sample_id = request.headers.get("x-fact-checking-sample-id")
        resource_headers = {"x-fact-checking-sample-id": sample_id} if sample_id else {}
        model_server_cookies = None  # update the cookies on every model response
        resources_server_cookies = request.cookies  # update the cookies on every resources server response

        async def _generate(tool_choice: Optional[str] = None) -> NeMoGymResponse:
            nonlocal model_server_cookies
            new_body = body.model_copy(update={"input": body.input + new_outputs})
            if tool_choice is not None:
                new_body = new_body.model_copy(update={"tool_choice": tool_choice})
            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_response)
            model_response_json = await model_response.json()
            model_server_cookies = model_response.cookies
            try:
                return NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

        def _has_assistant_message(outputs) -> bool:
            return any(
                getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant"
                for o in outputs
            )

        while True:
            step += 1
            step_tool_choice = None
            if step == 1 and self.config.force_first_step_tool_choice:
                step_tool_choice = self.config.force_first_step_tool_choice
            t_gen = time.perf_counter()
            model_response = await _generate(step_tool_choice)
            generate_s = time.perf_counter() - t_gen
            output = model_response.output
            new_outputs.extend(output)

            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                step_timings.append(
                    {
                        "step": step,
                        "kind": "generate",
                        "tool_choice": step_tool_choice or "auto",
                        "policy_generate_s": round(generate_s, 4),
                        "n_function_calls": 0,
                        "search_wiki_s": 0.0,
                        "incomplete": "max_output_tokens",
                    }
                )
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            tool_calls = []
            new_unique_searches = 0
            if not all_fn_calls and all_output_messages:
                step_timings.append(
                    {
                        "step": step,
                        "kind": "generate",
                        "tool_choice": step_tool_choice or "auto",
                        "policy_generate_s": round(generate_s, 4),
                        "n_function_calls": 0,
                        "search_wiki_s": 0.0,
                    }
                )
                break

            for output_function_call in all_fn_calls:
                query_norm = (
                    normalize_search_query(output_function_call.arguments)
                    if output_function_call.name == "search_wiki"
                    else ""
                )
                skip_reason = None
                if output_function_call.name == "search_wiki":
                    if (
                        self.config.max_search_calls is not None
                        and search_calls >= self.config.max_search_calls
                    ):
                        skip_reason = "search_budget"
                    elif (
                        self.config.skip_duplicate_search_queries
                        and query_norm
                        and query_norm in seen_search_queries
                    ):
                        skip_reason = "duplicate_query"

                if skip_reason:
                    if skip_reason == "search_budget":
                        stub = (
                            "Search budget exhausted. Use the evidence already retrieved "
                            "and write the tagged verdict."
                        )
                    else:
                        stub = (
                            "Duplicate query skipped; results were already returned "
                            "for this query."
                        )
                    tool_calls.append(
                        {
                            "name": output_function_call.name,
                            "s": 0.0,
                            "skipped": skip_reason,
                        }
                    )
                    new_outputs.append(
                        NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=output_function_call.call_id,
                            output=stub,
                        )
                    )
                    continue

                t_tool = time.perf_counter()
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                    cookies=resources_server_cookies,
                    headers=resource_headers,
                )
                # We don't raise for status here since it's a valid return for the API to error e.g. if the model outputs an invalid call or something.
                resources_server_cookies = api_response.cookies
                elapsed = time.perf_counter() - t_tool
                tool_calls.append(
                    {
                        "name": output_function_call.name,
                        "s": round(elapsed, 4),
                    }
                )

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=(await api_response.content.read()).decode(),
                )
                new_outputs.append(tool_response)
                if output_function_call.name == "search_wiki":
                    search_calls += 1
                    new_unique_searches += 1
                    if query_norm:
                        seen_search_queries.add(query_norm)

            step_timings.append(
                {
                    "step": step,
                    "kind": "generate+tools" if all_fn_calls else "generate",
                    "tool_choice": step_tool_choice or "auto",
                    "policy_generate_s": round(generate_s, 4),
                    "n_function_calls": len(all_fn_calls),
                    "search_wiki_s": round(
                        sum(c["s"] for c in tool_calls if c["name"] == "search_wiki"), 4
                    ),
                    "search_calls": search_calls,
                    "tool_calls": tool_calls,
                }
            )

            if self.config.max_search_calls is not None and (
                search_calls >= self.config.max_search_calls
                or (all_fn_calls and new_unique_searches == 0)
            ):
                break

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                break

        if (
            self.config.force_final_step_tool_choice
            and new_outputs
            and not _has_assistant_message(new_outputs)
        ):
            t_gen = time.perf_counter()
            model_response = await _generate(self.config.force_final_step_tool_choice)
            new_outputs.extend(model_response.output)
            step_timings.append(
                {
                    "step": step + 1,
                    "kind": "final_none",
                    "tool_choice": self.config.force_final_step_tool_choice,
                    "policy_generate_s": round(time.perf_counter() - t_gen, 4),
                    "n_function_calls": 0,
                    "search_wiki_s": 0.0,
                }
            )

        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)
        response.headers["X-Factcheck-Step-Timings"] = json.dumps(step_timings)
        if sample_id:
            _POLICY_STEP_TIMINGS[sample_id] = step_timings

        model_response.output = new_outputs
        return model_response

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        cookies = request.cookies
        sample_id = getattr(body, "id", None)
        agent_headers = {"x-fact-checking-sample-id": str(sample_id)} if sample_id is not None else {}
        run_t0 = time.perf_counter()

        t0 = time.perf_counter()
        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies
        seed_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
            headers=agent_headers,
        )
        await raise_for_status(response)
        cookies = response.cookies
        policy_loop_s = time.perf_counter() - t0
        raw_steps = response.headers.get("X-Factcheck-Step-Timings") or response.headers.get(
            "x-factcheck-step-timings"
        )
        try:
            policy_steps = json.loads(raw_steps) if raw_steps else []
        except json.JSONDecodeError:
            policy_steps = []
        if not policy_steps and sample_id is not None:
            policy_steps = _POLICY_STEP_TIMINGS.pop(str(sample_id), [])
        elif sample_id is not None:
            _POLICY_STEP_TIMINGS.pop(str(sample_id), None)

        verify_request = SimpleAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": await response.json()}
        )

        t0 = time.perf_counter()
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        verify_http_s = time.perf_counter() - t0
        payload = await verify_response.json()
        timings = dict(payload.get("timings") or {})
        timings.update(
            {
                "seed_session_s": round(seed_s, 4),
                "policy_loop_s": round(policy_loop_s, 4),
                "policy_generate_s": round(
                    sum(float(s.get("policy_generate_s") or 0.0) for s in policy_steps), 4
                ),
                "search_wiki_http_s": round(
                    sum(float(s.get("search_wiki_s") or 0.0) for s in policy_steps), 4
                ),
                "policy_steps": policy_steps,
                "verify_http_s": round(verify_http_s, 4),
                "run_s": round(time.perf_counter() - run_t0, 4),
            }
        )
        payload["timings"] = timings
        payload["timing_breakdown"] = step_timing_breakdown(timings)
        return SimpleAgentVerifyResponse.model_validate(payload)


if __name__ == "__main__":
    SimpleAgent.run_webserver()

from __future__ import annotations

import asyncio
from collections import deque
from difflib import SequenceMatcher
import random
import re
import time
from typing import Any, Optional

from aiohttp import ClientResponseError
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.config_types import AgentServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import raise_for_status
from resources_servers.fact_checking_reward_model_dev.app import (
    FactCheckingRewardModelDevConfig,
    FactCheckingRewardModelDevResourcesServer,
)
from resources_servers.veriscore_rm_policy_optimization.prompts import (
    QUALITY_RM_PROMPT_TEMPLATE,
    VERISCORE_ATOMIZATION_PROMPT_TEMPLATE,
    VERISCORE_JUDGE_PROMPT_TEMPLATE,
)
from resources_servers.veriscore_rm_policy_optimization.utils import (
    error_debug,
    extract_error_lines,
    extract_factuality_prediction,
    extract_quality_score,
    extract_text_from_response,
    format_input_conversation,
    percentile,
    response_debug,
)


# Runtime knobs for the quality RM call, VeriScore atomization, retrieval, judging, and reward mix.
class VeriScoreRMPolicyOptimizationConfig(FactCheckingRewardModelDevConfig):
    name: str = "veriscore_rm_policy_optimization"
    rm_agent_server: AgentServerRef
    rm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    prompt_template: str = Field(default=QUALITY_RM_PROMPT_TEMPLATE)
    atomization_prompt_template: str = Field(default=VERISCORE_ATOMIZATION_PROMPT_TEMPLATE)
    judge_prompt_template: str = Field(default=VERISCORE_JUDGE_PROMPT_TEMPLATE)
    factuality_weight: float = 0.5
    quality_weight: float = 0.5
    reward_for_parse_failure: float = 0.0
    veriscore_max_sentences: int = 24
    veriscore_max_atoms: int = 32
    veriscore_atomization_concurrency: int = 4
    veriscore_judge_concurrency: int = 4
    veriscore_atomization_max_output_tokens: int = 1024
    veriscore_judge_max_output_tokens: int = 256
    veriscore_evidence_top_k: int = 3
    veriscore_evidence_max_chars_per_hit: int = 3000
    veriscore_near_duplicate_similarity_threshold: float = 0.88
    veriscore_timing_log_every: int = 128


# Request types stay permissive because Nemo Gym examples can carry task-specific metadata.
class VeriScoreRMPolicyOptimizationRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class VeriScoreRMPolicyOptimizationVerifyRequest(
    VeriScoreRMPolicyOptimizationRunRequest, BaseVerifyRequest
):
    pass


# The verifier returns both reward-facing values and detailed traces for rollout inspection.
class VeriScoreRMPolicyOptimizationVerifyResponse(BaseVerifyResponse):
    rm_response_text: str
    rm_prompt: str
    rm_factuality_prediction: str
    factuality_prediction: str
    quality_score: Optional[float]
    hallucinated: Optional[bool]
    rm_num_errors: int
    factuality_reward: float
    quality_reward: float
    veriscore_atoms: list[str]
    veriscore_decisions: list[dict[str, Any]]
    veriscore_supported: int
    veriscore_contradicted: int
    veriscore_inconclusive: int
    rm_response_debug: dict[str, Any]


class VeriScoreRMPolicyOptimizationResourcesServer(FactCheckingRewardModelDevResourcesServer):
    config: VeriScoreRMPolicyOptimizationConfig
    _TIMING_KEYS = (
        "verify_wall_ms",
        "rm_wall_ms",
        "veriscore_wall_ms",
        "atomization_wall_ms",
        "search_wall_ms",
        "judge_wall_ms",
    )

    async def _call_judge_model(self, prompt: str, max_output_tokens: int) -> str:
        # Atomization and verdict judging share the same judge model server.
        if not self.config.judge_model_server or not self.config.judge_responses_create_params:
            raise RuntimeError("judge_model_server and judge_responses_create_params are required.")
        params = self.config.judge_responses_create_params.model_copy(deep=True)
        params.max_output_tokens = max_output_tokens
        params.tools = []
        params.tool_choice = "none"
        params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        response_obj = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params,
        )
        await raise_for_status(response_obj)
        return extract_text_from_response(NeMoGymResponse.model_validate(await response_obj.json()))

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Keep segmentation intentionally lightweight; max_sentences bounds bad splits.
        cleaned = re.sub(r"\n{2,}", "\n", text.strip())
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if part.strip()]

    @staticmethod
    def _parse_atoms(output: str) -> list[str]:
        # Accept simple bullet/numbered atom lists and ignore explicit no-claim outputs.
        if re.search(r"\bNo verifiable claim\b", output, flags=re.IGNORECASE):
            return []
        atoms: list[str] = []
        for line in output.splitlines():
            line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line.strip())
            if line and not line.lower().startswith(("facts:", "fact:")):
                atoms.append(line)
        return atoms

    @staticmethod
    def _atom_key(atom: str) -> str:
        # Exact dedup uses a normalized string key before the looser similarity pass.
        return re.sub(r"\s+", " ", atom).strip().rstrip(".").lower()

    @staticmethod
    def _atom_similarity(left: str, right: str) -> float:
        # SequenceMatcher gives a dependency-free 0..1 similarity over normalized atom text.
        return SequenceMatcher(None, left, right, autojunk=False).ratio()

    def _is_near_duplicate(self, key: str, kept_keys: list[str]) -> bool:
        # Compare each candidate only against atoms already kept, preserving first occurrence.
        if not key:
            return True
        threshold = self.config.veriscore_near_duplicate_similarity_threshold
        if threshold <= 0.0:
            return False
        return any(self._atom_similarity(key, previous) >= threshold for previous in kept_keys)

    def _dedupe_atoms(self, debug: list[dict[str, Any]]) -> list[str]:
        # Apply exact dedup first, then string-similarity filtering across atomized sentences.
        deduped: list[str] = []
        exact_seen: set[str] = set()
        kept_keys: list[str] = []
        for item in debug:
            for atom in item["atoms"]:
                key = self._atom_key(atom)
                if not key or key in exact_seen or self._is_near_duplicate(key, kept_keys):
                    continue
                exact_seen.add(key)
                kept_keys.append(key)
                deduped.append(atom)
        return deduped

    async def _atomize(self, response_text: str) -> tuple[list[str], list[dict[str, Any]]]:
        # Atomize sentence prompts concurrently, but keep debug output sorted by sentence order.
        sentences = self._split_sentences(response_text)[: self.config.veriscore_max_sentences]
        semaphore = asyncio.Semaphore(max(1, self.config.veriscore_atomization_concurrency))

        async def atomize_sentence(idx: int, sentence: str) -> dict[str, Any]:
            # Surround the focused sentence with SOS/EOS while leaving the full response as context.
            snippet = response_text.replace(sentence, f"<SOS>{sentence}<EOS>", 1)
            prompt = self.config.atomization_prompt_template.format(snippet=snippet, sentence=sentence)
            async with semaphore:
                start = time.monotonic()
                raw_output = await self._call_judge_model(
                    prompt,
                    max_output_tokens=self.config.veriscore_atomization_max_output_tokens,
                )
                atomization_ms = (time.monotonic() - start) * 1000
            return {
                "sentence_index": idx,
                "sentence": sentence,
                "prompt": prompt,
                "raw_output": raw_output,
                "atomization_ms": atomization_ms,
                "atoms": self._parse_atoms(raw_output),
            }

        debug = list(await asyncio.gather(*(atomize_sentence(i, s) for i, s in enumerate(sentences))))
        debug.sort(key=lambda item: item["sentence_index"])

        deduped = self._dedupe_atoms(debug)
        max_atoms = self.config.veriscore_max_atoms
        # Sample only when there are more atoms than the configured cap.
        if max_atoms <= 0:
            return [], debug
        if len(deduped) > max_atoms:
            deduped = [deduped[i] for i in sorted(random.sample(range(len(deduped)), max_atoms))]
        return deduped, debug

    def _search(self, query: str) -> list[dict[str, Any]]:
        # VeriScore evidence comes from OpenResearcher-backed retrieval, or none for ablations.
        backend = self.config.retrieval_backend.lower().replace("-", "_")
        if backend == "none":
            return []
        if backend == "tantivy":
            return self._tantivy_search(query, k=self.config.veriscore_evidence_top_k)
        raise RuntimeError(
            f"Unsupported VeriScore retrieval_backend={self.config.retrieval_backend!r}; "
            "use 'tantivy' or 'none'."
        )

    def _format_evidence(self, hits: list[dict[str, Any]]) -> str:
        # Keep evidence packets compact so judge prompts stay bounded and fast.
        if not hits:
            return "No retrieved evidence."
        parts: list[str] = []
        for i, hit in enumerate(hits, start=1):
            text = (hit.get("text") or "")[: self.config.veriscore_evidence_max_chars_per_hit]
            parts.append(
                f"[Result {i}]\n"
                f"URL: {hit.get('url', '')}\n"
                f"Score: {hit.get('score', '')}\n"
                f"Content:\n{text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _parse_verdict(output: str) -> str:
        # Trust only the final hashtag-delimited verdict; anything else is inconclusive.
        matches = re.findall(r"###(.*?)###", output, flags=re.DOTALL)
        if not matches:
            return "inconclusive"
        label = matches[-1].strip().lower().rstrip(".")
        return label if label in {"supported", "contradicted", "inconclusive"} else "inconclusive"

    async def _run_veriscore(self, response_text: str) -> dict[str, Any]:
        # Full VeriScore pass: atomize response, retrieve evidence per atom, judge each atom.
        veriscore_start = time.monotonic()
        atomize_start = time.monotonic()
        atoms, atomization_debug = await self._atomize(response_text)
        atomization_wall_ms = (time.monotonic() - atomize_start) * 1000
        semaphore = asyncio.Semaphore(max(1, self.config.veriscore_judge_concurrency))

        async def judge_atom(atom: str) -> dict[str, Any]:
            # Retrieval is blocking, so run it in a worker thread while judge calls stay async.
            search_start = time.monotonic()
            hits = await asyncio.to_thread(self._search, atom)
            search_ms = (time.monotonic() - search_start) * 1000
            prompt = self.config.judge_prompt_template.format(
                claim=atom,
                evidence=self._format_evidence(hits),
            )
            async with semaphore:
                judge_start = time.monotonic()
                output = await self._call_judge_model(
                    prompt,
                    max_output_tokens=self.config.veriscore_judge_max_output_tokens,
                )
                judge_ms = (time.monotonic() - judge_start) * 1000
            return {
                "atom": atom,
                "verdict": self._parse_verdict(output),
                "search_ms": search_ms,
                "judge_ms": judge_ms,
                "search_query": atom,
                "search_hits": [
                    {
                        "url": hit.get("url"),
                        "score": hit.get("score"),
                        "rerank_score": hit.get("rerank_score"),
                        "strategy": hit.get("strategy"),
                    }
                    for hit in hits
                ],
                "judge_prompt": prompt,
                "judge_output": output,
            }

        decisions = list(await asyncio.gather(*(judge_atom(atom) for atom in atoms)))
        # Factuality rewards supported atoms and penalizes contradicted atoms; inconclusive atoms are neutral.
        supported = sum(1 for item in decisions if item["verdict"] == "supported")
        contradicted = sum(1 for item in decisions if item["verdict"] == "contradicted")
        inconclusive = len(decisions) - supported - contradicted
        return {
            "atoms": atoms,
            "atomization_debug": atomization_debug,
            "decisions": decisions,
            "supported": supported,
            "contradicted": contradicted,
            "inconclusive": inconclusive,
            "factuality_reward": supported - contradicted,
            "timing": {
                "veriscore_wall_ms": (time.monotonic() - veriscore_start) * 1000,
                "atomization_wall_ms": atomization_wall_ms,
                "search_wall_ms": max((item["search_ms"] for item in decisions), default=0.0),
                "judge_wall_ms": max((item["judge_ms"] for item in decisions), default=0.0),
            },
        }

    async def _score_quality(self, prompt: str) -> tuple[str, dict[str, Any]]:
        # Quality RM is a plain no-tool completion; VeriScore handles factual verification.
        start = time.monotonic()
        params = self.config.rm_responses_create_params.model_copy(deep=True)
        params.tools = []
        params.tool_choice = "none"
        params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        try:
            response_obj = await self.server_client.post(
                server_name=self.config.rm_agent_server.name,
                url_path="/v1/responses",
                json=params,
            )
            status_code = getattr(response_obj, "status", None)
            await raise_for_status(response_obj)
            payload = await response_obj.json()
            response = NeMoGymResponse.model_validate(payload)
            debug = response_debug(response)
            debug["status_code"] = status_code
            debug["rm_wall_ms"] = (time.monotonic() - start) * 1000
            return extract_text_from_response(response), debug
        except ClientResponseError as error:
            # Preserve response payloads when possible so rollout logs explain RM failures.
            response_content = getattr(error, "response_content", b"")
            if isinstance(response_content, bytes):
                response_content = response_content.decode(errors="replace")
            debug = error_debug(
                error,
                response_payload=response_content or locals().get("payload"),
                status_code=getattr(error, "status", None),
            )
        except Exception as error:
            debug = error_debug(
                error,
                response_payload=locals().get("payload"),
                status_code=locals().get("status_code"),
            )
        debug["rm_wall_ms"] = (time.monotonic() - start) * 1000
        raw = debug.get("raw_response")
        return raw if isinstance(raw, str) else repr(debug), debug

    @staticmethod
    def _quality_reward(quality_score: Optional[float]) -> float:
        # Map the RM's 1..5 rubric onto 0..1; parse failures contribute zero quality reward.
        return (quality_score - 1.0) / 4.0 if quality_score is not None else 0.0

    def _record_timing(self, timing: dict[str, Any]) -> None:
        # Maintain running timing aggregates without storing every verifier call forever.
        log_every = max(0, self.config.veriscore_timing_log_every)
        if log_every == 0:
            return
        count = getattr(self, "_veriscore_timing_count", 0) + 1
        totals = getattr(self, "_veriscore_timing_totals", {key: 0.0 for key in self._TIMING_KEYS})
        maxes = getattr(self, "_veriscore_timing_maxes", {key: 0.0 for key in self._TIMING_KEYS})
        samples = getattr(self, "_veriscore_timing_samples", {key: deque(maxlen=512) for key in self._TIMING_KEYS})
        atoms_total = getattr(self, "_veriscore_timing_atoms_total", 0.0) + float(timing.get("num_atoms") or 0.0)
        for key in self._TIMING_KEYS:
            value = float(timing.get(key) or 0.0)
            totals[key] += value
            maxes[key] = max(maxes[key], value)
            samples.setdefault(key, deque(maxlen=512)).append(value)
        self._veriscore_timing_count = count
        self._veriscore_timing_totals = totals
        self._veriscore_timing_maxes = maxes
        self._veriscore_timing_samples = samples
        self._veriscore_timing_atoms_total = atoms_total
        if count % log_every:
            return
        # Print compact periodic timing stats for bottleneck diagnosis during long runs.
        parts = [f"count={count}", f"atoms_mean={atoms_total / count:.2f}"]
        for key in self._TIMING_KEYS:
            values = list(samples.get(key, []))
            name = key.removesuffix("_wall_ms").removesuffix("_ms")
            parts.extend(
                [
                    f"{name}_mean_ms={totals[key] / count:.1f}",
                    f"{name}_p50_ms={percentile(values, 0.50):.1f}",
                    f"{name}_p95_ms={percentile(values, 0.95):.1f}",
                    f"{name}_max_ms={maxes[key]:.1f}",
                ]
            )
        print("[veriscore-timing] " + " ".join(parts), flush=True)

    async def verify(
        self, body: VeriScoreRMPolicyOptimizationVerifyRequest
    ) -> VeriScoreRMPolicyOptimizationVerifyResponse:
        # Parse the policy response and input conversation from the Nemo Gym verify request.
        verify_start = time.monotonic()
        response_text = extract_text_from_response(body.response)
        input_conversation = format_input_conversation(body.responses_create_params.input or [])
        if not response_text.strip() or not input_conversation.strip():
            # Empty inputs are infrastructure/parse failures, not factuality judgments.
            return VeriScoreRMPolicyOptimizationVerifyResponse(
                **body.model_dump(),
                reward=self.config.reward_for_parse_failure,
                rm_response_text="Bad input/response",
                rm_prompt="",
                rm_factuality_prediction="",
                factuality_prediction="",
                quality_score=None,
                hallucinated=None,
                rm_num_errors=0,
                factuality_reward=0.0,
                quality_reward=0.0,
                veriscore_atoms=[],
                veriscore_decisions=[],
                veriscore_supported=0,
                veriscore_contradicted=0,
                veriscore_inconclusive=0,
                rm_response_debug={},
            )

        rm_prompt = self.config.prompt_template.format(
            input_conversation=input_conversation,
            response=response_text,
        )
        (rm_text, rm_debug), veriscore = await asyncio.gather(
            # Run quality scoring and VeriScore in parallel; neither depends on the other.
            self._score_quality(rm_prompt),
            self._run_veriscore(response_text),
        )
        quality_score = extract_quality_score(rm_text)
        quality_reward = self._quality_reward(quality_score)
        factuality_reward = float(veriscore["factuality_reward"])
        reward = self.config.quality_weight * quality_reward + self.config.factuality_weight * factuality_reward
        timing = {
            # Keep timing in rm_response_debug so rollout jsonl can be inspected post-hoc.
            **veriscore["timing"],
            "rm_wall_ms": rm_debug.get("rm_wall_ms", 0.0),
            "verify_wall_ms": (time.monotonic() - verify_start) * 1000,
            "num_atoms": len(veriscore["atoms"]),
        }
        self._record_timing(timing)

        contradicted = int(veriscore["contradicted"])
        supported = int(veriscore["supported"])
        total_atoms = len(veriscore["atoms"])
        # Keep legacy factuality fields for downstream consumers while reward uses contradictions.
        factuality_prediction = "YES" if total_atoms == 0 or supported == total_atoms else "NO" if contradicted else ""
        return VeriScoreRMPolicyOptimizationVerifyResponse(
            **body.model_dump(),
            reward=reward,
            rm_response_text=rm_text,
            rm_prompt=rm_prompt,
            rm_factuality_prediction=extract_factuality_prediction(rm_text),
            factuality_prediction=factuality_prediction,
            quality_score=quality_score,
            hallucinated=contradicted > 0,
            rm_num_errors=len(extract_error_lines(rm_text)),
            factuality_reward=factuality_reward,
            quality_reward=quality_reward,
            veriscore_atoms=veriscore["atoms"],
            veriscore_decisions=veriscore["decisions"],
            veriscore_supported=supported,
            veriscore_contradicted=contradicted,
            veriscore_inconclusive=int(veriscore["inconclusive"]),
            rm_response_debug={
                **rm_debug,
                "veriscore_atomization": veriscore["atomization_debug"],
                "veriscore_timing": timing,
            },
        )


RunRequest = VeriScoreRMPolicyOptimizationRunRequest
VerifyRequest = VeriScoreRMPolicyOptimizationVerifyRequest
VerifyResponse = VeriScoreRMPolicyOptimizationVerifyResponse


if __name__ == "__main__":
    VeriScoreRMPolicyOptimizationResourcesServer.run_webserver()

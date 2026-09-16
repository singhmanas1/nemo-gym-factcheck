"""
Self-contained fact-checking reward model resource server.

Uses Tantivy search over the OpenResearcher corpus or semantic search over an
external Milvus collection (no GPU needed by default).
The original user query and policy response can optionally be threaded
to the judge via HTTP cookies for context-aware summarization.
"""
from __future__ import annotations

import asyncio
import fcntl
import io
import json
import math
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Union
from urllib.parse import unquote

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


_SESSION_COOKIE = "fact_checking_rm_dev_session"
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
}

RETRIEVAL_SUMMARY_PROMPT_TEMPLATE = """You are preparing retrieved web evidence for a factuality judge.

You will receive multiple documents retrieved from a web corpus. Your job is to turn them into a compact evidence packet that helps a downstream judge verify claims related to the search query.

Rules:
1. Use ONLY the provided retrieved documents.
2. Do NOT add outside knowledge, assumptions, or speculation.
3. Extract facts that are directly useful for checking claims related to the query.
4. Prefer concrete, atomic facts: names, dates, places, titles, numbers, definitions, and explicit relationships.
5. Remove irrelevant details, repetition, stylistic filler, and unnecessary background.
6. If the documents are weak, tangential, or do not cleanly answer the query, include the most relevant facts found and describe the limitation precisely.
7. Do NOT write a general summary of the documents.
8. Keep the output short and evidence-focused.

Inputs:

[Search Query]
{query}

[Retrieved Documents]
{retrieved_content}

Return output in exactly this format:

[Query-Relevant Evidence]
- [<source url>] <atomic fact from the retrieved documents>
- [<source url>] <atomic fact from the retrieved documents>
- [<source url>] <atomic fact from the retrieved documents>

[Supported By Retrieved Evidence]
- <claim or point that the retrieved evidence clearly supports>

[Search Result Limitations]
- <what the retrieved documents do not establish, leave ambiguous, contradict, or only weakly suggest>

Requirements:
- Include 1 to 5 evidence bullets, ordered from most useful to least useful.
- Each bullet must contain only one atomic fact.
- Always include the source URL.
- Prefer evidence that directly answers the query over broad background facts.
- If there is no clean answer but there is tangential information, include the tangential information and describe what it does and does not establish.

"""

RETRIEVAL_SUMMARY_WITH_CONTEXT_PROMPT_TEMPLATE = """You are a research assistant helping a factuality judge verify whether a model response is accurate.

You will be given:
- The original user request
- The model response being fact-checked
- A search query that was used to retrieve evidence relevant to that response
- Documents retrieved from a web corpus

---

Original user request:
{original_query}

---

Model response being fact-checked:
{original_response}

---

Search query:
{query}

---

Retrieved documents:
{retrieved_content}

---

Your task: read the retrieved documents carefully and extract information that is relevant to the search query and to checking the model response. Then, for each specific factual claim in the model response that the retrieved documents can speak to, clearly state whether the evidence supports, contradicts, or is silent on that claim.

Instructions:
1. Use ONLY information from the retrieved documents. Do not add outside knowledge.
2. Extract detailed facts related to the search query: names, dates, roles, relationships, places, numbers, titles, definitions, and explicit claims.
3. If the retrieved documents are weak, tangential, or do not cleanly answer the query, include the most relevant facts found and describe the limitation precisely.
4. For each claim in the model response that touches on the search query topic, explicitly state what the evidence says about it.
5. If the evidence contradicts a claim in the response, state the contradiction clearly and precisely.
6. If the evidence confirms a claim, state that clearly.
7. If the evidence is silent or ambiguous on a claim, say so.
8. Do not summarize the documents generically. Focus entirely on what helps verify or refute the response.

Return your output in exactly this format:

[Extracted Evidence]
- [<source url>] <detailed fact from the retrieved documents>
- [<source url>] <detailed fact from the retrieved documents>

[Claims Confirmed By Evidence]
- "<exact or paraphrased claim from the response>" - confirmed by: <specific evidence>

[Claims Contradicted By Evidence]
- "<exact or paraphrased claim from the response>" - contradicted by: <specific evidence>

[Claims Not Addressed By Evidence]
- "<exact or paraphrased claim from the response>" - no relevant evidence found, or retrieved evidence is incomplete/ambiguous

[Search Result Limitations]
- <state whether the search results are clear, partial, tangential, conflicting, or unrelated>

"""

# ERROR_JUDGING_JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a model accurately identifies factual errors in a conversation.

# MODEL'S IDENTIFIED FACTUAL ERRORS:
# {predicted_errors}

# GROUND-TRUTH FACTUAL ERROR:
# {expected_error}

# Does the model's identify the factual error correctly in its factual errors?
# The identified factual error must be almost exactly the same as the ground truth error, with only small paraphrases allowed. Note that broad statements such as "Incorrect or misleading facts" should not be counted as matching the ground truth error. The level of precision in the predicted error must be the same as the level of precision in the ground truth error.
# Analyze carefully, then respond with exactly [[YES]] or [[NO]] on the last line."""

ERROR_JUDGING_JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a model accurately identifies factual errors in a conversation.

MODEL'S IDENTIFIED FACTUAL ERRORS:
{predicted_errors}

GROUND-TRUTH FACTUAL ERROR:
{expected_error}

Does the model's identify the factual error correctly in its factual errors?
The predicted error must identify the same incorrect fact as the ground-truth error, even if it quotes only part of the claim or states the correction, with paraphrases allowed. Importantly, the detail level of the predicted error must be the almost the same as the detail level of the ground truth error. Note that broad statements such as "Incorrect or misleading facts" should not be counted as matching the ground truth error.
Analyze carefully, then respond with exactly [[YES]] or [[NO]] on the last line."""

NUM_ERRORS_JUDGE_PROMPT_TEMPLATE = """You are counting how many factual errors attempts to identify in a conversation. The ground-truth factual errors are provided, and should be used as a reference for what constitutes an identified factual error. Note you are not evaluating correctness, but rather the number of errors identified. If a model idenitifies an error that is not in the ground-truth factual errors, still count it as an error.

If the model puts multiple errors in a single line, count each error separately. Use the ground-truth factual errors as a reference for what constitutes an identified factual error.

MODEL'S IDENTIFIED FACTUAL ERRORS:
{predicted_errors}

GROUND-TRUTH FACTUAL ERRORS:
{expected_errors}

How many factual errors does the model identify?
Analyze carefully, then respond with the number of errors identified in between <num_errors> tags (e.g. <num_errors>3</num_errors>)."""


def extract_query_and_response(checker_prompt: str) -> tuple[Optional[str], Optional[str]]:
    """Extract user query and policy response from a checker prompt."""
    conv_match = re.search(
        r"\[Beginning of Input Conversation\](.*?)\[End of Input Conversation\]",
        checker_prompt,
        flags=re.DOTALL,
    )
    query: Optional[str] = None
    conv_block = conv_match.group(1)
    msg_texts = re.findall(
        r"\[Begin of \w+ Message\](.*?)\[End of \w+ Message\]",
        conv_block,
        flags=re.DOTALL,
    )
    query = "\n".join(t.strip() for t in msg_texts).strip() or None

    resp_match = re.search(
        r"\[Beginning of Response 1\](.*?)\[End of Response 1\]",
        checker_prompt,
        flags=re.DOTALL,
    )
    response: Optional[str] = resp_match.group(1).strip() if resp_match else None
    return query, response


class DevSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    responses_create_params: Optional[Any] = None


class RubricEvaluation(BaseModel):
    expected_error: str
    judge_prompt: str
    judge_response: str
    verdict: str
    score: float


class SearchWikiRequest(BaseModel):
    query: str


class SearchWikiResponse(BaseModel):
    content: str


class FactCheckingRewardModelRunRequest(BaseRunRequest):
    id: Union[int, str]
    expected_errors: List[str]
    hallucination_severity: float


class FactCheckingRewardModelVerifyRequest(FactCheckingRewardModelRunRequest, BaseVerifyRequest):
    pass


class FactCheckingRewardModelVerifyResponse(BaseVerifyResponse):
    reward: float
    factuality_f1_score: float
    num_errors: int
    predicted_severity: float
    severity_reward: float


class FactCheckingRewardModelDevConfig(BaseResourcesServerConfig):
    name: str = "fact_checking_reward_model_dev"
    search_top_k: int = 3
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    rubric_parallel_evaluation: bool = True
    rubric_yes_label: str = "[[YES]]"
    rubric_no_label: str = "[[NO]]"
    factuality_weight: float = Field(
        default=1.0,
        description="Multiplicative weight applied to the factuality component of the reward.",
    )
    f_beta: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Beta for the factual-error F-beta reward. 1.0 is F1; values below "
            "1.0 weight precision more than recall."
        ),
    )
    severity_weight: float = Field(
        default=1.0,
        description="Multiplicative weight applied to the severity component of the reward.",
    )
    error_judging_judge_prompt_template: str = Field(
        default=ERROR_JUDGING_JUDGE_PROMPT_TEMPLATE,
        description="Template for the judge evaluation prompt.",
    )
    num_errors_judge_prompt_template: str = Field(
        default=NUM_ERRORS_JUDGE_PROMPT_TEMPLATE,
        description="Template for the error-count judge prompt.",
    )
    corpus_db_path: Optional[str] = Field(
        default=None,
        description="Path to the SQLite corpus DB (columns: docid, url, text).",
    )
    corpus_stage_local: bool = Field(
        default=True,
        description=(
            "If True, copy corpus.db and its WAL/SHM sidecars to node-local "
            "storage before opening SQLite. This must stay enabled for DBs on "
            "shared filesystems because read-only WAL clients still take locks."
        ),
    )
    corpus_local_root: str = Field(
        default="/tmp",
        description="Node-local root used when corpus_stage_local is enabled.",
    )
    max_result_chars: int = Field(
        default=8000,
        description="Max characters of document text returned per result.",
    )
    search_cache_size: int = Field(
        default=10000,
        description="Maximum number of completed searches to keep in an in-process LRU cache.",
    )
    retrieval_backend: str = Field(
        default="tantivy",
        description="Retrieval backend: 'none', 'tantivy', or 'milvus'.",
    )
    tantivy_index_dir: Optional[str] = Field(
        default=None,
        description="Path to the Tantivy OpenResearcher index.",
    )
    tantivy_stage_local: bool = Field(
        default=True,
        description="If True, copy the Tantivy index to node-local storage before opening it.",
    )
    tantivy_local_root: str = Field(
        default="/tmp",
        description="Local root used when tantivy_stage_local is enabled.",
    )
    tantivy_candidate_pool: int = Field(
        default=10,
        description="Number of Tantivy candidates to retrieve before lexical reranking.",
    )
    tantivy_searcher_pool_size: int = Field(
        default=16,
        description="Number of independent Tantivy searchers used for concurrent search requests.",
    )
    milvus_uri: str = Field(
        default="http://127.0.0.1:19530",
        description="Milvus server URI.",
    )
    milvus_collection_name: str = Field(
        default="finewebBrowsecomp322M",
        description="Milvus collection containing the document vectors.",
    )
    milvus_embedding_model: str = Field(
        default="google/embeddinggemma-300m",
        description="Sentence Transformers model used to embed search queries.",
    )
    milvus_embedding_device: str = Field(
        default="cpu",
        description="Device used by the Milvus query encoder. Keep this on CPU during training.",
    )
    milvus_embedding_base_url: Optional[str] = Field(
        default=None,
        description=(
            "If set, embed queries via an OpenAI-compatible /v1/embeddings server "
            "(e.g. http://127.0.0.1:8002/v1) instead of loading SentenceTransformer locally."
        ),
    )
    milvus_anns_field: str = Field(
        default="emb",
        description="Milvus vector field used for ANN search.",
    )
    milvus_search_list: int = Field(
        default=256,
        gt=0,
        description="AISAQ search_list parameter.",
    )
    milvus_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Timeout for Milvus client operations.",
    )
    milvus_candidate_multiplier: int = Field(
        default=4,
        gt=0,
        description="Retrieve this multiple of top-k before exact-text deduplication.",
    )
    use_context_aware_prompt: bool = Field(
        default=False,
        description=(
            "If True, pass the original query+response from the session context to the judge. "
            "Produces better evidence but significantly increases judge input length."
        ),
    )
    summarize_retrieval_results: bool = Field(
        default=True,
        description=(
            "If True, use the judge model to summarize retrieved documents before returning "
            "them from search_wiki. If False, return the raw retrieved documents."
        ),
    )
    search_log_path: Optional[str] = Field(
        default=None,
        description=(
            "If set, every search_wiki call is appended as a JSON line to this file. "
            "Each entry contains: session_id, backend, query, hits, and summarized output."
        ),
    )
    retrieval_summary_prompt_template: str = Field(
        default=RETRIEVAL_SUMMARY_PROMPT_TEMPLATE,
        description="Prompt template used to summarize retrieved evidence before returning it to the model.",
    )


class _OpenAICompatEmbeddingEncoder:
    """Query encoder that calls an OpenAI-compatible embeddings HTTP server."""

    def __init__(self, base_url: str, model: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._dim: Optional[int] = None

    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is None:
            vector = self.encode_query("dimension probe", convert_to_numpy=True)
            self._dim = int(len(vector))
        return self._dim

    def encode_query(self, text: str, convert_to_numpy: bool = True):
        return self.encode(text, convert_to_numpy=convert_to_numpy)

    def encode(self, text: str, convert_to_numpy: bool = True):
        import json
        import urllib.error
        import urllib.request

        import numpy as np

        payload = json.dumps({"input": text, "model": self.model}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Embedding server {self.base_url} is unreachable: {exc}"
            ) from exc
        vector = body["data"][0]["embedding"]
        array = np.asarray(vector, dtype="float32")
        return array if convert_to_numpy else array.tolist()


class FactCheckingRewardModelDevResourcesServer(SimpleResourcesServer):
    config: FactCheckingRewardModelDevConfig
    _corpus_db: Optional[sqlite3.Connection] = None
    _corpus_db_uri: Optional[str] = None
    _corpus_db_thread_local: Any = None
    _search_cache: OrderedDict = OrderedDict()
    _search_cache_lock: Optional[threading.Lock] = None
    _tantivy_index: Any = None
    _tantivy_searcher: Any = None
    _tantivy_searchers: Optional[list] = None
    _tantivy_searcher_locks: Optional[list] = None
    _tantivy_pool_lock: Optional[threading.Lock] = None
    _tantivy_next_searcher: int = 0
    _tantivy_lock: Optional[threading.Lock] = None
    _milvus_client: Any = None
    _milvus_encoder: Any = None
    _milvus_lock: Optional[threading.Lock] = None
    _session_contexts: dict = {}
    _session_sample_ids: dict = {}
    _search_log_file: Optional[io.TextIOWrapper] = None
    _search_log_lock: Optional[threading.Lock] = None

    def setup_webserver(self) -> FastAPI:
        backend = self.config.retrieval_backend.lower().replace("-", "_")
        if backend == "tantivy":
            if not self.config.corpus_db_path:
                raise RuntimeError("corpus_db_path must be set when retrieval_backend='tantivy'.")
            self._setup_tantivy()
            corpus_db_path = self._maybe_stage_corpus_db(
                Path(self.config.corpus_db_path)
            )
            print(f"[dev] Opening corpus DB at {corpus_db_path} ...")
            t0 = time.monotonic()
            db_uri = corpus_db_path.resolve().as_uri() + "?mode=ro"
            self._corpus_db_uri = db_uri
            self._corpus_db_thread_local = threading.local()
            self._corpus_db = self._open_corpus_db_connection()
            self._search_cache_lock = threading.Lock()
            print(f"[dev] Corpus DB opened in {time.monotonic()-t0:.1f}s")
        elif backend == "milvus":
            self._setup_milvus()
            self._search_cache_lock = threading.Lock()
        elif backend == "none":
            pass
        else:
            raise RuntimeError(f"Unsupported retrieval_backend={self.config.retrieval_backend!r}")

        self._search_log_lock = threading.Lock()
        if self.config.search_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.config.search_log_path)), exist_ok=True)
            self._search_log_file = open(self.config.search_log_path, "a", buffering=1)
            print(f"[dev] Search logging enabled → {self.config.search_log_path}")

        app = SimpleResourcesServer.setup_webserver(self)
        app.post("/search_wiki")(self.search_wiki)
        return app

    def _setup_milvus(self) -> None:
        print(
            f"[dev] Connecting to Milvus collection "
            f"{self.config.milvus_collection_name!r} at {self.config.milvus_uri} ..."
        )
        t0 = time.monotonic()
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError(
                "pymilvus is required for Milvus retrieval."
            ) from exc

        self._milvus_client = MilvusClient(
            uri=self.config.milvus_uri,
            timeout=self.config.milvus_timeout_seconds,
        )
        collection_name = self.config.milvus_collection_name
        if not self._milvus_client.has_collection(collection_name=collection_name):
            raise RuntimeError(
                f"Milvus collection {collection_name!r} does not exist at "
                f"{self.config.milvus_uri}."
            )

        schema = self._milvus_client.describe_collection(collection_name=collection_name)
        vector_dim: Optional[int] = None
        for field in schema.get("fields", []):
            if field.get("name") == self.config.milvus_anns_field:
                raw_dim = field.get("params", {}).get("dim")
                vector_dim = int(raw_dim) if raw_dim is not None else None
                break
        if vector_dim is None:
            raise RuntimeError(
                f"Milvus vector field {self.config.milvus_anns_field!r} was not found "
                f"in collection {collection_name!r}."
            )

        try:
            if self.config.milvus_embedding_base_url:
                self._milvus_encoder = _OpenAICompatEmbeddingEncoder(
                    base_url=self.config.milvus_embedding_base_url,
                    model=self.config.milvus_embedding_model,
                    timeout_seconds=self.config.milvus_timeout_seconds,
                )
            else:
                from sentence_transformers import SentenceTransformer

                self._milvus_encoder = SentenceTransformer(
                    self.config.milvus_embedding_model,
                    device=self.config.milvus_embedding_device,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Milvus query encoder "
                f"{self.config.milvus_embedding_model!r}. If this is the gated "
                "EmbeddingGemma model, accept its license and provide HF_TOKEN, "
                "or set milvus_embedding_model to a readable local model path."
            ) from exc
        encoder_dim = self._milvus_encoder.get_sentence_embedding_dimension()
        if encoder_dim is not None and int(encoder_dim) != vector_dim:
            raise RuntimeError(
                f"Milvus field dimension is {vector_dim}, but query encoder "
                f"{self.config.milvus_embedding_model!r} produces {encoder_dim} dimensions."
            )
        self._milvus_lock = threading.Lock()
        print(
            f"[dev] Milvus ready in {time.monotonic()-t0:.1f}s "
            f"(dim={vector_dim}, encoder={self.config.milvus_embedding_base_url or self.config.milvus_embedding_device})"
        )

    def _setup_tantivy(self) -> None:
        if not self.config.tantivy_index_dir:
            raise RuntimeError("tantivy_index_dir must be set when retrieval_backend='tantivy'.")

        index_dir = self._maybe_stage_tantivy_index(Path(self.config.tantivy_index_dir))
        print(f"[dev] Opening Tantivy index from {index_dir} ...")
        t0 = time.monotonic()
        try:
            import tantivy
        except ImportError as exc:
            raise RuntimeError("tantivy is required for Tantivy retrieval.") from exc

        self._tantivy_index = tantivy.Index.open(str(index_dir))
        self._tantivy_index.reload()
        searcher_pool_size = max(1, self.config.tantivy_searcher_pool_size)
        self._tantivy_searchers = [
            self._tantivy_index.searcher() for _ in range(searcher_pool_size)
        ]
        self._tantivy_searcher = self._tantivy_searchers[0]
        self._tantivy_searcher_locks = [
            threading.Lock() for _ in range(searcher_pool_size)
        ]
        self._tantivy_pool_lock = threading.Lock()
        self._tantivy_lock = threading.Lock()
        print(
            f"[dev] Tantivy index ready in {time.monotonic()-t0:.1f}s "
            f"with {searcher_pool_size} searchers"
        )

    def _maybe_stage_tantivy_index(self, source_dir: Path) -> Path:
        if not self.config.tantivy_stage_local:
            return source_dir

        job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        target_root = Path(self.config.tantivy_local_root) / f"tantivy-openresearcher-{job_id}"
        target_dir = target_root / source_dir.name
        done_marker = target_root / ".copy_complete"
        if done_marker.exists() and target_dir.exists():
            print(f"[dev] Reusing staged Tantivy index at {target_dir}")
            return target_dir

        print(f"[dev] Staging Tantivy index from {source_dir} to {target_dir} ...")
        target_root.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir, symlinks=True)
        done_marker.write_text("ok\n")
        print(f"[dev] Staged Tantivy index in {time.monotonic()-t0:.1f}s")
        return target_dir

    def _maybe_stage_corpus_db(self, source_db: Path) -> Path:
        """Stage the complete SQLite WAL file set on node-local storage.

        SQLite read-only connections can still lock and update ``-shm`` when a
        database is in WAL mode. Keeping the main DB on a shared filesystem is
        therefore unsafe at the concurrency used by RL training. The marker
        and local flock let server restarts on the same node reuse one complete
        per-job copy without racing each other.
        """
        source_db = source_db.resolve()
        if not self.config.corpus_stage_local:
            print(
                "[dev] WARNING: corpus_stage_local=false; SQLite may generate "
                f"shared-filesystem lock traffic at {source_db}"
            )
            return source_db
        if not source_db.is_file():
            raise RuntimeError(f"Corpus DB does not exist: {source_db}")

        job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        local_root = Path(self.config.corpus_local_root)
        target_root = local_root / f"fact-checker-corpus-{job_id}"
        target_db = target_root / source_db.name
        done_marker = target_root / ".copy_complete"
        lock_path = local_root / f".fact-checker-corpus-{job_id}.lock"
        source_files = [
            path
            for path in (
                source_db,
                Path(str(source_db) + "-wal"),
                Path(str(source_db) + "-shm"),
            )
            if path.exists()
        ]

        local_root.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if done_marker.exists() and target_db.is_file():
                print(f"[dev] Reusing staged corpus DB at {target_db}")
                return target_db

            required_bytes = sum(path.stat().st_size for path in source_files)
            free_bytes = shutil.disk_usage(local_root).free
            reserve_bytes = 10 * 1024**3
            if free_bytes < required_bytes + reserve_bytes:
                raise RuntimeError(
                    f"Not enough node-local space under {local_root}: need "
                    f"{required_bytes / 1024**3:.1f} GiB plus 10 GiB reserve, "
                    f"have {free_bytes / 1024**3:.1f} GiB free"
                )

            print(
                f"[dev] Staging SQLite corpus ({required_bytes / 1024**3:.1f} GiB) "
                f"from {source_db} to {target_root} ..."
            )
            t0 = time.monotonic()
            if target_root.exists():
                shutil.rmtree(target_root)
            target_root.mkdir(parents=True)
            for source_file in source_files:
                target_file = target_root / source_file.name
                temporary_file = target_file.with_name(
                    target_file.name + f".tmp-{os.getpid()}"
                )
                shutil.copy2(source_file, temporary_file)
                if temporary_file.stat().st_size != source_file.stat().st_size:
                    raise RuntimeError(
                        f"Incomplete local SQLite copy: {source_file} -> "
                        f"{temporary_file}"
                    )
                temporary_file.replace(target_file)
            done_marker.write_text("ok\n")
            print(f"[dev] Staged SQLite corpus in {time.monotonic()-t0:.1f}s")
            return target_db

    def _open_corpus_db_connection(self) -> sqlite3.Connection:
        if not self._corpus_db_uri:
            raise RuntimeError("Corpus DB URI is not initialized.")
        conn = sqlite3.connect(self._corpus_db_uri, uri=True, check_same_thread=False)
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA cache_size=-1000000")
        return conn

    def _get_corpus_db_connection(self) -> sqlite3.Connection:
        if self._corpus_db_thread_local is None:
            if self._corpus_db is None:
                raise RuntimeError("Corpus DB is not initialized.")
            return self._corpus_db
        conn = getattr(self._corpus_db_thread_local, "db", None)
        if conn is None:
            conn = self._open_corpus_db_connection()
            self._corpus_db_thread_local.db = conn
        return conn

    def _next_tantivy_searcher(self) -> tuple[Any, threading.Lock]:
        if not self._tantivy_searchers or not self._tantivy_searcher_locks:
            if self._tantivy_searcher is None or self._tantivy_lock is None:
                raise RuntimeError("Tantivy index is not initialized.")
            return self._tantivy_searcher, self._tantivy_lock
        if self._tantivy_pool_lock is None:
            raise RuntimeError("Tantivy searcher pool lock is not initialized.")
        with self._tantivy_pool_lock:
            idx = self._tantivy_next_searcher % len(self._tantivy_searchers)
            self._tantivy_next_searcher += 1
        return self._tantivy_searchers[idx], self._tantivy_searcher_locks[idx]

    @staticmethod
    def _extract_text_from_response(response: NeMoGymResponse) -> str:
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
                    return full_text.split("</think>")[-1].strip()
        return ""

    @staticmethod
    def _extract_verdict(response_text: str, yes_label: str, no_label: str) -> str:
        yes_pos = response_text.rfind(yes_label)
        no_pos = response_text.rfind(no_label)
        if yes_pos < 0 and no_pos < 0:
            last_line = response_text.strip().split("\n")[-1].upper() if response_text.strip() else ""
            if "YES" in last_line:
                return "YES"
            return "NO"
        return "YES" if yes_pos > no_pos else "NO"

    @staticmethod
    def _aggregate_scores(
        scores: list[float],
        num_predicted_errors: int,
        num_ground_truth_errors: int,
        beta: float = 1.0,
    ) -> float:
        """Compute an F-beta score from soft true-positive judgments."""
        num_predicted_errors = max(0, int(num_predicted_errors))
        num_ground_truth_errors = max(0, int(num_ground_truth_errors))
        beta = float(beta)
        if beta <= 0.0:
            raise ValueError("f_beta must be greater than zero")

        if num_predicted_errors == 0 and num_ground_truth_errors == 0:
            return 1.0

        if not scores or num_predicted_errors == 0 or num_ground_truth_errors == 0:
            return 0.0

        tp = max(0.0, float(sum(scores)))
        tp = min(tp, float(num_predicted_errors), float(num_ground_truth_errors))

        precision = tp / num_predicted_errors if num_predicted_errors > 0 else 0.0
        recall = tp / num_ground_truth_errors if num_ground_truth_errors > 0 else 0.0

        beta_squared = beta * beta
        denominator = beta_squared * precision + recall
        if denominator == 0:
            return 0.0
        return (1.0 + beta_squared) * precision * recall / denominator

    @staticmethod
    def _extract_factual_severity(output: str) -> float:
        matches = re.findall(
            r"\[Beginning of Factual Severity\](.*?)\[End of Factual Severity\]",
            output,
            re.DOTALL,
        )
        # A reasoning trace can quote the response template before the actual
        # verdict, so prefer the last valid tagged severity.
        for match in reversed(matches):
            try:
                score = float(match.strip())
            except (ValueError, TypeError):
                continue
            if 1.0 <= score <= 5.0:
                return score
        return -1

    async def seed_session(
        self,
        body: DevSeedSessionRequest,
        response: Response,
    ) -> BaseSeedSessionResponse:
        session_id = str(uuid.uuid4())
        context = self._extract_checker_context(body)
        if context:
            self._session_contexts[session_id] = context
        self._session_sample_ids[session_id] = str(body.id)
        response.set_cookie(_SESSION_COOKIE, session_id, httponly=True)
        return BaseSeedSessionResponse()

    @staticmethod
    def _extract_checker_context(body: DevSeedSessionRequest) -> Optional[str]:
        try:
            params = body.responses_create_params
            if params is None:
                return None
            inp = params.get("input") if isinstance(params, dict) else getattr(params, "input", None)
            if not inp:
                return None
            first = inp[0] if isinstance(inp, list) else None
            if first is None:
                return None
            content = first.get("content") if isinstance(first, dict) else getattr(first, "content", None)
            return content if isinstance(content, str) else None
        except Exception:
            return None

    def _log_search(
        self,
        session_id: Optional[str],
        query: str,
        hits: list,
        summarized: str,
        retrieval_ms: float = 0.0,
        summarize_ms: float = 0.0,
        backend: Optional[str] = None,
        error: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> None:
        if self._search_log_file is None:
            return
        entry = {
            "session_id": session_id,
            "sample_id": sample_id or (self._session_sample_ids.get(session_id) if session_id else None),
            "backend": backend or self.config.retrieval_backend,
            "query": query,
            "retrieval_ms": round(retrieval_ms, 1),
            "summarize_ms": round(summarize_ms, 1),
            "hits": [
                {
                    "url": h.get("url"),
                    "title": h.get("title"),
                    "score": h.get("score"),
                    "distance": h.get("distance"),
                    "rerank_score": h.get("rerank_score"),
                }
                for h in hits
            ],
            "summarized": summarized,
        }
        if error:
            entry["error"] = error
        with self._search_log_lock:
            self._search_log_file.write(json.dumps(entry) + "\n")

    @staticmethod
    def _search_tokens(text: str) -> list[str]:
        text = re.sub(r"\bu\.s\.c\.?\b", " usc ", text, flags=re.IGNORECASE)
        tokens = _TOKEN_RE.findall(text.lower())
        return [t for t in tokens if len(t) > 1 and t not in _QUERY_STOPWORDS]

    @classmethod
    def _token_coverage(cls, query_tokens: list[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        haystack = set(cls._search_tokens(text))
        if not haystack:
            return 0.0
        return sum(1 for token in query_tokens if token in haystack) / len(query_tokens)

    @classmethod
    def _rerank_hit(cls, query: str, hit: dict) -> float:
        query_tokens = cls._search_tokens(query)
        normalized_url = unquote(hit["url"]).replace("-", " ").replace("_", " ")
        head_text = hit["text"][:6000]
        combined = f"{normalized_url}\n{head_text}"

        url_coverage = cls._token_coverage(query_tokens, normalized_url)
        text_coverage = cls._token_coverage(query_tokens, head_text)
        combined_coverage = cls._token_coverage(query_tokens, combined)

        query_l = query.lower().strip()
        combined_l = combined.lower()
        phrase_bonus = 0.0
        if query_l and query_l in combined_l:
            phrase_bonus += 12.0

        legal_markers = ("u.s.c", "usc", "§", "gen stat", "bankr", "f.3d", "b.r.", "in re")
        if any(marker in query_l for marker in legal_markers):
            legal_terms = [t for t in query_tokens if t.isdigit() or t in {"usc", "bankr", "stat", "re"}]
            legal_coverage = cls._token_coverage(legal_terms, combined) if legal_terms else 0.0
            phrase_bonus += 10.0 * legal_coverage
            if legal_coverage < 0.5:
                phrase_bonus -= 8.0

        return (
            float(hit["score"])
            + 14.0 * url_coverage
            + 10.0 * combined_coverage
            + 4.0 * text_coverage
            + phrase_bonus
        )

    @staticmethod
    def _tantivy_doc_value(doc: dict, key: str) -> str:
        value = doc.get(key)
        if isinstance(value, list):
            return str(value[0]) if value else ""
        return str(value) if value is not None else ""

    def _tantivy_search(self, query: str, k: int) -> list[dict]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        candidate_k = max(k, self.config.tantivy_candidate_pool)
        fetch_chars = max(self.config.max_result_chars, 6000)
        cache_key = ("tantivy", normalized_query, k, candidate_k, fetch_chars)
        cached = self._search_cache_get(cache_key)
        if cached is not None:
            return cached

        if self._tantivy_index is None:
            raise RuntimeError("Tantivy index is not initialized.")

        parsed_query, parse_errors = self._tantivy_index.parse_query_lenient(
            normalized_query,
            ["text", "url"],
        )
        searcher, searcher_lock = self._next_tantivy_searcher()
        with searcher_lock:
            results = searcher.search(parsed_query, candidate_k)
            docs = []
            for score, addr in results.hits:
                doc = searcher.doc(addr).to_dict()
                docs.append((float(score), doc))

        docids = [self._tantivy_doc_value(doc, "docid") for _, doc in docs]
        rows_by_docid = self._fetch_corpus_rows(docids, fetch_chars)

        hits = []
        parse_note = f" parse_errors={len(parse_errors)}" if parse_errors else ""
        for score, doc in docs:
            docid = self._tantivy_doc_value(doc, "docid")
            url = self._tantivy_doc_value(doc, "url")
            row = rows_by_docid.get(str(docid))
            if row is None:
                continue
            db_url, text = row
            hit = {
                "url": url or db_url,
                "text": text,
                "score": score,
                "strategy": f"tantivy_raw_lenient{parse_note}",
            }
            hit["rerank_score"] = self._rerank_hit(query, hit)
            hits.append(hit)

        hits.sort(key=lambda h: h["rerank_score"], reverse=True)
        hits = hits[:k]
        self._search_cache_put(cache_key, hits)
        return [dict(hit) for hit in hits]

    def _milvus_search(self, query: str, k: int) -> list[dict]:
        normalized_query = query.strip()
        if not normalized_query or k <= 0:
            return []

        candidate_k = max(k, k * self.config.milvus_candidate_multiplier)
        cache_key = (
            "milvus",
            normalized_query,
            k,
            candidate_k,
            self.config.milvus_collection_name,
            self.config.milvus_embedding_model,
            self.config.milvus_search_list,
        )
        cached = self._search_cache_get(cache_key)
        if cached is not None:
            return cached

        if (
            self._milvus_client is None
            or self._milvus_encoder is None
            or self._milvus_lock is None
        ):
            raise RuntimeError("Milvus retrieval is not initialized.")

        with self._milvus_lock:
            encode_query = getattr(self._milvus_encoder, "encode_query", None)
            if encode_query is not None:
                query_vector = encode_query(normalized_query, convert_to_numpy=True)
            else:
                query_vector = self._milvus_encoder.encode(
                    normalized_query,
                    convert_to_numpy=True,
                )
            vector = query_vector.tolist()
            if not vector or not all(math.isfinite(float(value)) for value in vector):
                raise RuntimeError("Milvus query encoder returned a non-finite vector.")

        results = self._milvus_client.search(
            collection_name=self.config.milvus_collection_name,
            data=[vector],
            limit=candidate_k,
            anns_field=self.config.milvus_anns_field,
            output_fields=["orig_id", "text"],
            search_params={"search_list": self.config.milvus_search_list},
        )

        raw_hits = results[0] if results else []
        hits: list[dict] = []
        seen_texts: set[str] = set()
        for raw_hit in raw_hits:
            entity = raw_hit.get("entity") or {}
            text = str(entity.get("text") or "").strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            result_id = raw_hit.get("id")
            orig_id = str(entity.get("orig_id") or result_id)
            distance = float(raw_hit.get("distance", 0.0))
            hits.append(
                {
                    "url": f"milvus://{self.config.milvus_collection_name}/{orig_id}",
                    "text": text,
                    "score": -distance,
                    "distance": distance,
                    "strategy": "milvus_embeddinggemma_aisaq",
                }
            )
            if len(hits) >= k:
                break

        self._search_cache_put(cache_key, hits)
        return [dict(hit) for hit in hits]

    def _fetch_corpus_rows(self, docids: list[str], fetch_chars: int) -> dict[str, tuple[str, str]]:
        if not docids:
            return {}
        placeholders = ",".join("?" * len(docids))
        rows = self._get_corpus_db_connection().execute(
            f"""
            SELECT docid, url, substr(text, 1, ?)
            FROM corpus
            WHERE docid IN ({placeholders})
            """,
            [fetch_chars, *docids],
        ).fetchall()
        return {str(docid): (url, text) for docid, url, text in rows}

    def _search_cache_get(self, key: tuple) -> Optional[list[dict]]:
        if self.config.search_cache_size <= 0:
            return None
        with self._search_cache_lock:
            cached = self._search_cache.get(key)
            if cached is None:
                return None
            self._search_cache.move_to_end(key)
            return [dict(hit) for hit in cached]

    def _search_cache_put(self, key: tuple, hits: list[dict]) -> None:
        if self.config.search_cache_size <= 0:
            return
        with self._search_cache_lock:
            self._search_cache[key] = [dict(hit) for hit in hits]
            self._search_cache.move_to_end(key)
            while len(self._search_cache) > self.config.search_cache_size:
                self._search_cache.popitem(last=False)

    async def _summarize_retrieved_content(
        self, query: str, retrieved_content: str, context: Optional[str] = None
    ) -> str:
        if not self.config.summarize_retrieval_results:
            return retrieved_content
        if not self.config.judge_model_server or not self.config.judge_responses_create_params:
            return retrieved_content

        if self.config.use_context_aware_prompt and context:
            original_query, original_response = extract_query_and_response(context)
            if original_query and original_response:
                judge_prompt = RETRIEVAL_SUMMARY_WITH_CONTEXT_PROMPT_TEMPLATE.format(
                    original_query=original_query,
                    original_response=original_response,
                    query=query,
                    retrieved_content=retrieved_content,
                )
            else:
                judge_prompt = self.config.retrieval_summary_prompt_template.format(
                    query=query,
                    retrieved_content=retrieved_content,
                )
        else:
            judge_prompt = self.config.retrieval_summary_prompt_template.format(
                query=query,
                retrieved_content=retrieved_content,
            )

        request_params = self.config.judge_responses_create_params.model_copy(deep=True)
        request_params.input = [NeMoGymEasyInputMessage(role="user", content=judge_prompt)]
        response_obj = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=request_params,
        )
        judge_response_obj = NeMoGymResponse.model_validate(await response_obj.json())
        summary = self._extract_text_from_response(judge_response_obj)
        return summary or retrieved_content

    async def search_wiki(self, request: Request, body: SearchWikiRequest) -> SearchWikiResponse:
        session_id = request.cookies.get(_SESSION_COOKIE)
        sample_id = request.headers.get("x-fact-checking-sample-id")
        context = self._session_contexts.get(session_id) if session_id else None

        try:
            t0 = time.monotonic()
            backend = self.config.retrieval_backend.lower().replace("-", "_")
            if backend == "none":
                content = (
                    "Search is disabled for this run. No retrieved evidence is available "
                    f"for query: {body.query}"
                )
                self._log_search(session_id, body.query, [], content, retrieval_ms=0.0, backend=backend, sample_id=sample_id)
                return SearchWikiResponse(content=content)
            if backend == "tantivy":
                hits = await asyncio.to_thread(
                    self._tantivy_search,
                    body.query,
                    self.config.search_top_k,
                )
            elif backend == "milvus":
                hits = await asyncio.to_thread(
                    self._milvus_search,
                    body.query,
                    self.config.search_top_k,
                )
            else:
                raise RuntimeError(f"Unsupported retrieval_backend={self.config.retrieval_backend!r}")
            retrieval_ms = (time.monotonic() - t0) * 1000

            if not hits:
                self._log_search(
                    session_id,
                    body.query,
                    [],
                    "No results found",
                    retrieval_ms=retrieval_ms,
                    backend=backend,
                    sample_id=sample_id,
                )
                return SearchWikiResponse(content=f"No results found for: {body.query}")

            parts = []
            for hit in hits:
                text = hit["text"][: self.config.max_result_chars]
                parts.append(f"=== {hit['url']} ===\n\n{text}")
            web_content = "\n\n".join(parts)

            t1 = time.monotonic()
            summarized = await self._summarize_retrieved_content(
                query=body.query,
                retrieved_content=web_content,
                context=context,
            )
            summarize_ms = (time.monotonic() - t1) * 1000

            self._log_search(
                session_id,
                body.query,
                hits,
                summarized,
                retrieval_ms=retrieval_ms,
                summarize_ms=summarize_ms,
                backend=backend,
                sample_id=sample_id,
            )
            return SearchWikiResponse(content=summarized)
        except Exception as e:
            self._log_search(
                session_id,
                body.query,
                [],
                f"Search error ({self.config.retrieval_backend}): {e}. Query was: {body.query}",
                backend=self.config.retrieval_backend,
                error=str(e),
                sample_id=sample_id,
            )
            return SearchWikiResponse(
                content=f"Search error ({self.config.retrieval_backend}): {e}. Query was: {body.query}"
            )

    async def _evaluate_single_error(
        self, expected_error: str, predicted_factual_errors: str
    ) -> RubricEvaluation:
        judge_prompt = self.config.error_judging_judge_prompt_template.format(
            expected_error=expected_error,
            predicted_errors=predicted_factual_errors,
        )
        request_params = self.config.judge_responses_create_params.model_copy(deep=True)
        request_params.input = [NeMoGymEasyInputMessage(role="user", content=judge_prompt)]
        response_obj = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=request_params,
        )
        judge_response_obj = NeMoGymResponse.model_validate(await response_obj.json())
        judge_response = self._extract_text_from_response(judge_response_obj)
        verdict = self._extract_verdict(
            judge_response,
            self.config.rubric_yes_label,
            self.config.rubric_no_label,
        )
        score = 1.0 if verdict == "YES" else 0.0
        return RubricEvaluation(
            expected_error=expected_error,
            judge_prompt=judge_prompt,
            judge_response=judge_response,
            verdict=verdict,
            score=score,
        )

    async def _evaluate_num_errors(
        self, expected_errors: str, predicted_factual_errors: str
    ) -> int:
        judge_prompt = self.config.num_errors_judge_prompt_template.format(
            expected_errors=expected_errors,
            predicted_errors=predicted_factual_errors,
        )
        request_params = self.config.judge_responses_create_params.model_copy(deep=True)
        request_params.input = [NeMoGymEasyInputMessage(role="user", content=judge_prompt)]
        response_obj = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=request_params,
        )
        judge_response_obj = NeMoGymResponse.model_validate(await response_obj.json())
        judge_response = self._extract_text_from_response(judge_response_obj)
        try:
            return int(judge_response.split("<num_errors>")[-1].split("</num_errors>")[0].strip())
        except Exception:
            return 20

    async def verify(
        self, body: FactCheckingRewardModelVerifyRequest
    ) -> FactCheckingRewardModelVerifyResponse:
        output = self._extract_text_from_response(body.response)

        expected_errors = [x for x in body.expected_errors if x.strip()]

        m = re.search(
            r"\[Beginning of Factual Errors\](.*?)\[End of Factual Errors\]",
            output,
            re.DOTALL,
        )
        if not m:
            # An incomplete agent trajectory made no final factual-error claim.
            # Score it as predicting an empty error list (severity 1), rather
            # than conflating a tool-loop/format failure with an asserted error.
            factual_severity = 1.0
            f1_score = 1.0 if not expected_errors else 0.0
            severity_reward = (
                1.0 if body.hallucination_severity == factual_severity else 0.0
            )
            result = FactCheckingRewardModelVerifyResponse(
                **body.model_dump(),
                reward=float(
                    self.config.factuality_weight * f1_score
                    + self.config.severity_weight * severity_reward
                ),
                factuality_f1_score=f1_score,
                predicted_severity=factual_severity,
                severity_reward=severity_reward,
                num_errors=0,
            )
        else:
            predicted_factual_errors = m.group(1).strip()
            if not predicted_factual_errors and len(expected_errors) > 0:
                f1_score = 0.0
                num_errors = 0
            elif not predicted_factual_errors and len(expected_errors) == 0:
                f1_score = 1.0
                num_errors = 0
            else:
                if self.config.rubric_parallel_evaluation and len(expected_errors) > 1:
                    evaluations = await asyncio.gather(
                        *[
                            self._evaluate_single_error(err, predicted_factual_errors)
                            for err in expected_errors
                        ]
                    )
                else:
                    evaluations = []
                    for err in expected_errors:
                        evaluations.append(
                            await self._evaluate_single_error(err, predicted_factual_errors)
                        )

                scores = [e.score for e in evaluations]
                num_errors = await self._evaluate_num_errors(
                    "\n\n".join(expected_errors), predicted_factual_errors
                )
                
                if len(expected_errors) > 0:
                    f1_score = self._aggregate_scores(
                        scores=scores,
                        num_predicted_errors=num_errors,
                        num_ground_truth_errors=len(expected_errors),
                        beta=self.config.f_beta,
                    )
                else:
                    f1_score = 0.0

            factual_severity = self._extract_factual_severity(output)
            expected_severity = body.hallucination_severity
            if expected_severity == factual_severity:
                severity_reward = 1.0
            else:
                severity_reward = 0.0
            reward = self.config.factuality_weight * f1_score + self.config.severity_weight * severity_reward
            result = FactCheckingRewardModelVerifyResponse(
                **body.model_dump(),
                reward=float(reward),
                factuality_f1_score=f1_score,
                num_errors=num_errors,
                predicted_severity=factual_severity,
                severity_reward=severity_reward,
            )

        if len(self._session_contexts) > 10_000:
            oldest = list(self._session_contexts.keys())[: len(self._session_contexts) - 10_000]
            for k in oldest:
                self._session_contexts.pop(k, None)
                self._session_sample_ids.pop(k, None)
        return result


RunRequest = FactCheckingRewardModelRunRequest
VerifyRequest = FactCheckingRewardModelVerifyRequest
VerifyResponse = FactCheckingRewardModelVerifyResponse


if __name__ == "__main__":
    FactCheckingRewardModelDevResourcesServer.run_webserver()

"""
Dev variant of FactCheckerPolicyOptimizationResourcesServer.

Uses a local BM25s index over the OpenResearcher web corpus (no GPU needed)
instead of Kiwix Wikipedia search.

Context (the prompt/response pair being judged) is automatically threaded
to the summarization judge via HTTP cookies — no agent cooperation required.
The simple_agent framework already propagates cookies from seed_session
through every tool call, so we mint a session ID in seed_session, store the
checker prompt context keyed by it, and read it back in search_wiki.
"""
from __future__ import annotations

import io
import json
import sqlite3
import threading
import uuid
from typing import Any, Optional

import bm25s
from fastapi import FastAPI, Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse
from resources_servers.fact_checker_policy_optimization.app import (
    FactCheckerPolicyOptimizationConfig,
    FactCheckerPolicyOptimizationResourcesServer,
    FactCheckerPolicyOptimizationRunRequest,
    FactCheckerPolicyOptimizationVerifyRequest,
    FactCheckerPolicyOptimizationVerifyResponse,
)
from resources_servers.factual_reward_model_with_wiki.app import SearchWikiResponse
from resources_servers.fact_checker_policy_optimization_dev.prompts import (
    FACT_CHECKER_POLICY_PROMPT_TEMPLATE,
    SEARCH_WIKI_TOOL,
    WIKI_SUMMARY_WITH_CONTEXT_PROMPT_TEMPLATE,
    extract_query_and_response,
)

_SESSION_COOKIE = "fact_checker_dev_session"


class DevSeedSessionRequest(BaseSeedSessionRequest):
    """Accepts the full run-request body so we can extract checker context."""
    model_config = ConfigDict(extra="allow")
    responses_create_params: Optional[Any] = None


class DevSearchWikiRequest(BaseSeedSessionRequest):
    query: str


class FactCheckerPolicyOptimizationDevConfig(FactCheckerPolicyOptimizationConfig):
    name: str = "fact_checker_policy_optimization_dev"
    prompt_template: str = Field(default=FACT_CHECKER_POLICY_PROMPT_TEMPLATE)
    bm25s_index_dir: Optional[str] = Field(
        default=None,
        description="Path to the bm25s index directory (contains docids.json).",
    )
    bm25s_db_path: Optional[str] = Field(
        default=None,
        description="Path to the SQLite corpus DB (columns: docid, url, text).",
    )
    bm25s_max_chars: int = Field(
        default=18000,
        description="Max characters of document text returned per result.",
    )
    search_log_path: Optional[str] = Field(
        default=None,
        description=(
            "If set, every search_wiki call is appended as a JSON line to this file. "
            "Each entry contains: session_id, query, bm25_hits (url + score), and summarized output. "
            "Useful for analysing retrieval quality post-hoc."
        ),
    )


class FactCheckerPolicyOptimizationDevResourcesServer(
    FactCheckerPolicyOptimizationResourcesServer
):
    config: FactCheckerPolicyOptimizationDevConfig
    _bm25_retriever: Optional[bm25s.BM25] = None
    _bm25_docids: Optional[list] = None
    _bm25_db: Optional[sqlite3.Connection] = None
    _session_contexts: dict = {}  # session_id -> checker prompt text
    _search_log_file: Optional[io.TextIOWrapper] = None
    _search_log_lock: Optional[threading.Lock] = None

    def setup_webserver(self) -> FastAPI:
        if not self.config.bm25s_index_dir or not self.config.bm25s_db_path:
            raise RuntimeError(
                "bm25s_index_dir and bm25s_db_path must both be set for the dev server. "
                "Use the production fact_checker_policy_optimization server if you want Kiwix."
            )
        print(f"[dev] Loading BM25s index from {self.config.bm25s_index_dir} ...")
        self._bm25_retriever = bm25s.BM25.load(
            self.config.bm25s_index_dir, load_corpus=False
        )
        with open(f"{self.config.bm25s_index_dir}/docids.json") as f:
            self._bm25_docids = json.load(f)
        self._bm25_db = sqlite3.connect(
            self.config.bm25s_db_path, check_same_thread=False
        )
        self._bm25_db.execute("PRAGMA journal_mode=WAL")
        print(f"[dev] BM25s index ready ({len(self._bm25_docids):,} docs)")

        self._search_log_lock = threading.Lock()
        if self.config.search_log_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(self.config.search_log_path)), exist_ok=True)
            self._search_log_file = open(self.config.search_log_path, "a", buffering=1)
            print(f"[dev] Search logging enabled → {self.config.search_log_path}")

        app = SimpleResourcesServer.setup_webserver(self)
        app.post("/search_wiki")(self.search_wiki)
        return app

    # ------------------------------------------------------------------
    # Session management — mint a session ID and store checker context
    # ------------------------------------------------------------------

    async def seed_session(
        self,
        body: DevSeedSessionRequest,
        response: Response,
    ) -> BaseSeedSessionResponse:
        session_id = str(uuid.uuid4())
        context = self._extract_checker_context(body)
        if context:
            self._session_contexts[session_id] = context
        response.set_cookie(_SESSION_COOKIE, session_id, httponly=True)
        return BaseSeedSessionResponse()

    @staticmethod
    def _extract_checker_context(body: DevSeedSessionRequest) -> Optional[str]:
        """Pull the checker prompt text out of the seed_session request body."""
        try:
            params = body.responses_create_params
            if params is None:
                return None
            # responses_create_params.input is a list of messages; the checker
            # prompt is the first user message.
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

    # ------------------------------------------------------------------
    # Search result logging
    # ------------------------------------------------------------------

    def _log_search(
        self,
        session_id: Optional[str],
        query: str,
        hits: list,
        summarized: str,
    ) -> None:
        if self._search_log_file is None:
            return
        entry = {
            "session_id": session_id,
            "query": query,
            "bm25_hits": [
                {"url": h["url"], "score": h["score"]} for h in hits
            ],
            "summarized": summarized,
        }
        with self._search_log_lock:
            self._search_log_file.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # BM25 retrieval
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, k: int) -> list[dict]:
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        results, scores = self._bm25_retriever.retrieve(tokens, k=k)
        hits = []
        for idx, score in zip(results[0], scores[0]):
            docid = self._bm25_docids[idx]
            row = self._bm25_db.execute(
                "SELECT url, text FROM corpus WHERE docid=?", (str(docid),)
            ).fetchone()
            if row:
                hits.append({"url": row[0], "text": row[1], "score": float(score)})
        return hits

    # ------------------------------------------------------------------
    # Context-aware summarization
    # ------------------------------------------------------------------

    async def _summarize_wiki_content(
        self, query: str, wiki_content: str, context: Optional[str] = None
    ) -> str:
        if (
            not self.config.judge_model_server
            or not self.config.judge_responses_create_params
        ):
            return wiki_content

        if context:
            original_query, original_response = extract_query_and_response(context)
            if original_query and original_response:
                judge_prompt = WIKI_SUMMARY_WITH_CONTEXT_PROMPT_TEMPLATE.format(
                    original_query=original_query,
                    original_response=original_response,
                    query=query,
                    wiki_content=wiki_content,
                )
            else:
                # Extraction failed — fall back to query-only prompt
                judge_prompt = self.config.wiki_summary_prompt_template.format(
                    query=query,
                    wiki_content=wiki_content,
                )
        else:
            judge_prompt = self.config.wiki_summary_prompt_template.format(
                query=query,
                wiki_content=wiki_content,
            )

        request_params = self.config.judge_responses_create_params.model_copy(deep=True)
        request_params.input = [
            NeMoGymEasyInputMessage(role="user", content=judge_prompt)
        ]
        response_obj = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=request_params,
        )
        judge_response_obj = NeMoGymResponse.model_validate(await response_obj.json())
        summary = self._extract_text_from_response(judge_response_obj)
        return summary or wiki_content

    # ------------------------------------------------------------------
    # search_wiki endpoint — reads session cookie for automatic context
    # ------------------------------------------------------------------

    async def search_wiki(
        self, request: Request, body: DevSearchWikiRequest
    ) -> SearchWikiResponse:
        session_id = request.cookies.get(_SESSION_COOKIE)
        context = self._session_contexts.get(session_id) if session_id else None

        try:
            hits = self._bm25_search(body.query, k=self.config.kiwix_top_k)
            if not hits:
                self._log_search(session_id, body.query, [], "No results found")
                return SearchWikiResponse(content=f"No results found for: {body.query}")
            parts = []
            for hit in hits:
                text = hit["text"][: self.config.bm25s_max_chars]
                parts.append(f"=== {hit['url']} ===\n\n{text}")
            web_content = "\n\n".join(parts)
            summarized = await self._summarize_wiki_content(
                query=body.query,
                wiki_content=web_content,
                context=context,
            )
            self._log_search(session_id, body.query, hits, summarized)
            return SearchWikiResponse(content=summarized)
        except Exception as e:
            return SearchWikiResponse(
                content=f"BM25 search error: {e}. Query was: {body.query}"
            )

    # ------------------------------------------------------------------
    # Clean up session context after verify() completes
    # ------------------------------------------------------------------

    async def verify(
        self, body: FactCheckerPolicyOptimizationVerifyRequest
    ) -> FactCheckerPolicyOptimizationVerifyResponse:
        result = await super().verify(body)
        # Clean up stored context (cookies aren't visible here but we can
        # prune old sessions to avoid unbounded growth — keep last 10k)
        if len(self._session_contexts) > 10_000:
            oldest = list(self._session_contexts.keys())[: len(self._session_contexts) - 10_000]
            for k in oldest:
                self._session_contexts.pop(k, None)
        return result


RunRequest = FactCheckerPolicyOptimizationRunRequest
VerifyRequest = FactCheckerPolicyOptimizationVerifyRequest
VerifyResponse = FactCheckerPolicyOptimizationVerifyResponse


if __name__ == "__main__":
    FactCheckerPolicyOptimizationDevResourcesServer.run_webserver()

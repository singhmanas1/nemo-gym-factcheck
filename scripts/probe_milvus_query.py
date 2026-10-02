#!/usr/bin/env python3
"""Probe one query against Milvus or Exa and flag matching passages.

Milvus: the harness asks for search_top_k * milvus_candidate_multiplier
(3 * 4 = 12) and then keeps 3. search_list is the ANN walk width.

Exa: the harness POSTs type=auto with numResults=search_top_k (3) and up to
8000 characters of page text. --limit is numResults for that call.

--hyde asks the policy model to turn --query into one short passage before
either endpoint is called. The passage may only restate clues from the
question. Unknowns stay as placeholders.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


# HyDE_1 style: prose from the query only. The unknown year stays a
# placeholder. No award category, album title, or calendar year is added.
CHRIS_ROCK_HYDE = (
    "Chris Rock won his first Grammy Award in [YEAR]. "
    "The honor is recorded as the first Grammy Award he received."
)

HYDE_INSTRUCTION = (
    "Turn the question into one short document passage for retrieval.\n"
    "Restate only clues already written in the question.\n"
    "Leave each unknown as a bracket placeholder such as [NAME], [YEAR], "
    "[METHOD], or [CHANNEL].\n"
    "Do not guess the answer. Do not add dates, titles, mechanisms, "
    "or other specific facts that the question does not already state.\n"
    "Output only the passage."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def embed(base_url: str, model: str, text: str, timeout: float) -> list[float]:
    payload = json.dumps({"input": text, "model": model}).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    return body["data"][0]["embedding"]


def policy_model_id(base_url: str, timeout: float, explicit: str | None) -> str:
    if explicit:
        return explicit
    request = urllib.request.Request(f"{base_url.rstrip('/')}/models", method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    data = body.get("data") or []
    if not data or not data[0].get("id"):
        raise SystemExit(f"No model id at {base_url}/models")
    return str(data[0]["id"])


def write_hyde(base_url: str, model: str, question: str, timeout: float) -> str:
    # The Lightning chat template starts the assistant inside <think> unless
    # enable_thinking is false. An unclosed think block is the whole reply,
    # and that trace must not be embedded.
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": HYDE_INSTRUCTION},
                {"role": "user", "content": question},
            ],
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise SystemExit(f"Policy HyDE request failed: {exc.code} {detail}") from exc
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    print(f"hyde_finish={choice.get('finish_reason')}")
    text = _THINK_RE.sub("", str(message.get("content") or "")).strip()
    lowered = text.lower()
    if (
        not text
        or "<think>" in lowered
        or "thinking process" in lowered
        or len(text) > 500
    ):
        raise SystemExit(
            "Policy model did not return a short passage. "
            "Refusing to search with the reasoning trace:\n"
            + text[:400]
        )
    return text


def snippet_around(text: str, needle: str) -> str:
    low = text.lower()
    idx = low.find(needle.lower())
    if idx < 0:
        return " ".join(text.split())[:140]
    start = max(0, idx - 180)
    end = min(len(text), idx + 280)
    return " ".join(text[start:end].split())


def print_exa(query: str, k: int, timeout: float, key_file: Path, needles: list[str]) -> None:
    results = exa_search(query, k, timeout, key_file)
    print(
        f"query={query!r} backend=exa type=auto "
        f"returned={len(results)} numResults={k}"
    )
    matched = 0
    for rank, result in enumerate(results, 1):
        text = str(result.get("text") or "")
        low = text.lower()
        found = [needle for needle in needles if needle.lower() in low]
        mark = ",".join(found) if found else "-"
        title = " ".join(str(result.get("title") or "").split())
        print(
            f"{rank:3d}  score={result.get('score')}  match={mark}  "
            f"{result.get('url')}"
        )
        print(f"     {title}")
        if not found:
            print(f"     {' '.join(text.split())[:140]}")
            continue
        matched += 1
        print(f"     SNIPPET: {snippet_around(text, found[0])}")
    print(f"matches={matched} of {len(results)}")


def print_milvus(args: argparse.Namespace, query: str) -> None:
    from pymilvus import MilvusClient

    vector = embed(args.embed_url, args.embed_model, query, args.timeout)
    client = MilvusClient(uri=args.uri, timeout=args.timeout)
    results = client.search(
        collection_name=args.collection,
        data=[vector],
        limit=args.limit,
        anns_field="emb",
        output_fields=["orig_id", "text"],
        search_params={"search_list": args.search_list},
    )
    hits = results[0] if results else []
    print(
        f"query={query!r} backend=milvus dim={len(vector)} "
        f"returned={len(hits)} limit={args.limit} search_list={args.search_list}"
    )
    matched = 0
    focused = 0
    focus = args.focus.lower() if args.focus else None
    for rank, hit in enumerate(hits, 1):
        entity = hit.get("entity") or {}
        text = str(entity.get("text") or "")
        low = text.lower()
        if focus is not None and focus not in low:
            continue
        focused += 1
        found = [needle for needle in args.needle if needle.lower() in low]
        orig = entity.get("orig_id") or hit.get("id")
        head = " ".join(text.split())[:140]
        mark = ",".join(found) if found else "-"
        print(f"{rank:3d}  dist={hit.get('distance')}  match={mark}  id={orig}")
        print(f"     {head}")
        if not found:
            continue
        matched += 1
        print(f"     SNIPPET: {snippet_around(text, found[0])}")
    if focus is None:
        print(f"matches={matched} of {len(hits)}")
    else:
        print(f"focus={args.focus!r} shown={focused} matches={matched} of {len(hits)}")


def exa_search(query: str, k: int, timeout: float, key_file: Path) -> list[dict]:
    api_key = os.environ.get("EXA_API_KEY", "").strip()
    if not api_key and key_file.is_file():
        api_key = key_file.read_text().strip()
    if not api_key:
        raise SystemExit(f"EXA_API_KEY is empty and {key_file} is missing")
    payload = json.dumps(
        {
            "query": query,
            "type": "auto",
            "numResults": k,
            "contents": {
                "text": {"maxCharacters": 8000, "includeHtmlTags": False}
            },
        }
    ).encode()
    request = urllib.request.Request(
        "https://api.exa.ai/search",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "User-Agent": "nemo-gym-factcheck",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    return body.get("results") or []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="Chris Rock first Grammy Award year",
    )
    parser.add_argument(
        "--preset",
        choices=("chris-rock-hyde",),
        default=None,
        help="Replace --query with a hypothetical answer passage and embed that.",
    )
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Ask the policy model to rewrite --query into a HyDE passage first.",
    )
    parser.add_argument(
        "--policy-url",
        default="http://127.0.0.1:8000/v1",
        help="Policy chat-completions base URL used by --hyde.",
    )
    parser.add_argument(
        "--policy-model",
        default=None,
        help="Policy model id. Default is the first id from --policy-url/models.",
    )
    parser.add_argument(
        "--backend",
        choices=("milvus", "exa", "both"),
        default="milvus",
    )
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument(
        "--exa-limit",
        type=int,
        default=10,
        help="Exa numResults. Separate from --limit so a wide Milvus probe stays small.",
    )
    parser.add_argument("--search-list", type=int, default=256)
    parser.add_argument("--uri", default="http://10.185.120.81:19530")
    parser.add_argument("--collection", default="finewebBrowsecomp322M")
    parser.add_argument("--embed-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--embed-model", default="google/embeddinggemma-300m")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--exa-key-file",
        type=Path,
        default=Path("/home/nvidia/nemo-gym-factcheck/.exa_api_key"),
    )
    parser.add_argument(
        "--needle",
        action="append",
        default=["chris rock", "roll with the new"],
        help="Case-insensitive substring. Repeat to add more.",
    )
    parser.add_argument(
        "--focus",
        default=None,
        help="Milvus only: print hits whose text contains this substring.",
    )
    args = parser.parse_args()
    if args.preset == "chris-rock-hyde":
        args.query = CHRIS_ROCK_HYDE
        print("preset=chris-rock-hyde")
        print(args.query)
        print()
    if args.hyde:
        model = policy_model_id(args.policy_url, args.timeout, args.policy_model)
        print(f"hyde_model={model}")
        print("hyde_question=" + args.query)
        args.query = write_hyde(args.policy_url, model, args.query, args.timeout)
        print("hyde_passage=" + args.query)
        print()

    if args.backend in ("milvus", "both"):
        print_milvus(args, args.query)
        if args.backend == "both":
            print()
    if args.backend in ("exa", "both"):
        print_exa(
            args.query,
            args.exa_limit,
            args.timeout,
            args.exa_key_file,
            args.needle,
        )


if __name__ == "__main__":
    main()

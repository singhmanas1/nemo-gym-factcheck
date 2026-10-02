#!/usr/bin/env python3
"""Print gold vs the [Factual Errors] blocks verify() can score."""
import json
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from collect_timing import timing_from_jsonl  # noqa: E402

ERR_RE = re.compile(
    r"\[Beginning of Factual Errors\](.*?)\[End of Factual Errors\]",
    re.DOTALL,
)
SEV_RE = re.compile(
    r"\[Beginning of Factual Severity\](.*?)\[End of Factual Severity\]",
    re.DOTALL,
)


def item_texts(item: dict) -> str:
    texts = []
    content = item.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif isinstance(part.get("summary"), str):
                    texts.append(part["summary"])
    elif isinstance(content, str):
        texts = [content]
    return "\n".join(texts).strip()


def last_assistant_text(row: dict) -> str:
    output = (row.get("response") or {}).get("output") or []
    for item in reversed(output):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        full = item_texts(item)
        if full:
            return full.split("</think>")[-1].strip()
    return ""


def gather_by_type(row: dict) -> dict[str, str]:
    buckets: dict[str, list[str]] = {}
    for item in (row.get("response") or {}).get("output") or []:
        kind = item.get("type") or "?"
        buckets.setdefault(kind, []).append(item_texts(item) or json.dumps(item.get("name") or item.get("arguments") or "")[:200])
    return {k: "\n---\n".join(v) for k, v in buckets.items()}


def main() -> None:
    path = sys.argv[1]
    reports = timing_from_jsonl(path)
    print("timing")
    print(json.dumps(reports if len(reports) != 1 else reports[0], indent=4))

    with open(path) as f:
        first = None
        for line in f:
            line = line.strip()
            if line:
                first = json.loads(line)
                break
    if first is None:
        print("empty jsonl")
        return
    row = first
    gold = row.get("expected_errors") or []
    resp = row.get("response") or {}
    scored = last_assistant_text(row)
    blocks = [b.strip() for b in ERR_RE.findall(scored)]
    sevs = [s.strip() for s in SEV_RE.findall(scored)]
    by_type = gather_by_type(row)
    print("GOLD")
    for g in gold:
        print(repr(g))
    print("incomplete_details", resp.get("incomplete_details"))
    print("output_types", [o.get("type") for o in (resp.get("output") or [])])
    timings = row.get("timings")
    breakdown = row.get("timing_breakdown") if isinstance(row.get("timing_breakdown"), dict) else None
    if breakdown and breakdown.get("steps"):
        print("\n----- step table -----")
        print(f"{'step':<36} {'s':>8} {'share':>8}")
        for step in breakdown["steps"]:
            share_pct = 100.0 * float(step.get("share") or 0.0)
            print(f"{step.get('label', step.get('step', '')):<36} {float(step.get('s') or 0.0):8.3f} {share_pct:7.1f}%")
        print(f"{'total':<36} {float(breakdown.get('total_s') or 0.0):8.3f}")
    if isinstance(timings, dict):
        print("\n----- step timings -----")
        compact = {k: v for k, v in timings.items() if k not in {"policy_steps", "searches", "judge_yes_no"}}
        print(json.dumps(compact, indent=2))
        if timings.get("policy_steps"):
            print("policy_steps", json.dumps(timings["policy_steps"], indent=2))
        if timings.get("searches"):
            print("searches", json.dumps(timings["searches"], indent=2))
        if timings.get("judge_yes_no"):
            print("judge_yes_no", json.dumps(timings["judge_yes_no"], indent=2))
    else:
        print("\n----- step timings -----\n<missing>")
    print(f"\nscored_message_chars={len(scored)} error_blocks={len(blocks)} severity_blocks={sevs}")
    for kind in ("reasoning", "message", "function_call"):
        blob = by_type.get(kind, "")
        err_n = len(ERR_RE.findall(blob)) if blob else 0
        print(f"{kind}_chars={len(blob)} {kind}_error_blocks={err_n}")
    print("\nALL error blocks in scored MESSAGE (verify() uses FIRST)")
    if not blocks:
        print("-----")
        print("<NO BLOCK>")
        print("-----")
    for i, block in enumerate(blocks):
        print(f"----- block {i} -----")
        print(block)
        print("-----")
    reason = by_type.get("reasoning", "")
    rblocks = [b.strip() for b in ERR_RE.findall(reason)]
    if rblocks:
        print("\nLAST error block inside REASONING (not scored by verify() today)")
        print("-----")
        print(rblocks[-1][:1500])
        print("-----")
    evals = row.get("judge_evaluations") or []
    if evals:
        print(f"\njudge_evaluations={len(evals)}")
        for i, ev in enumerate(evals):
            print(f"----- judge {i} verdict={ev.get('verdict')} score={ev.get('score')} -----")
            gold_err = ev.get("expected_error")
            if gold_err:
                print("GOLD_ERROR", repr(gold_err))
            reasoning = (ev.get("judge_reasoning") or "").strip()
            print("----- judge reasoning -----")
            print(reasoning[-3000:] if reasoning else "<EMPTY>")
            print("----- judge scored message -----")
            print((ev.get("judge_response") or "")[-2000:])
            print("-----")
    else:
        print("\njudge_evaluations=<missing> (not stored on this collect)")

    print("\n----- search_wiki -----")
    pending_query = None
    n_search = 0
    for item in resp.get("output") or []:
        kind = item.get("type")
        if kind == "function_call":
            n_search += 1
            raw_args = item.get("arguments") or "{}"
            try:
                query = json.loads(raw_args).get("query", raw_args)
            except Exception:
                query = raw_args
            pending_query = query
            print(f"----- search {n_search} query -----")
            print(query)
        elif kind == "function_call_output":
            blob = item.get("output") or ""
            print(f"----- search {n_search} hits ({len(blob)} chars) -----")
            print(blob[:2000])
            if len(blob) > 2000:
                print(f"... [{len(blob) - 2000} more chars]")
            print(f"(query was: {pending_query!r})")
    if n_search == 0:
        print("<no search_wiki calls>")

    print("\n----- count judge -----")
    print("response:", (row.get("count_judge_response") or "<missing>")[-1500:])
    print("reasoning:", (row.get("count_judge_reasoning") or "<missing>")[-1500:])

    print("\nscored message tail")
    print("-----")
    print(scored[-800:] if scored else "<EMPTY>")
    print("-----")


if __name__ == "__main__":
    main()

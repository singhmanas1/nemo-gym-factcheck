#!/usr/bin/env python3
"""Split a collect JSONL into empty vs non-empty gold and report F1.

Joins on the inner Input Conversation + Response 1 text, not a raw hash of
``responses_create_params`` (Gym rewrites that object: max_output_tokens, pydantic).
"""
from __future__ import annotations

import json
import sys
from collections import Counter


_MARKERS = (
    ("[Beginning of Response 1]", "[End of Response 1]"),
    ("[Beginning of Input Conversation]", "[End of Input Conversation]"),
)


def _walk_strings(obj, out: list[str]) -> None:
    if isinstance(obj, str):
        if obj:
            out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _walk_strings(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _walk_strings(value, out)


def _fingerprint(params) -> str | None:
    blobs: list[str] = []
    _walk_strings(params, blobs)
    text = "\n".join(blobs)
    if not text.strip():
        return None
    parts = []
    for start, end in _MARKERS:
        i = text.find(start)
        j = text.find(end)
        if i != -1 and j != -1 and j > i:
            parts.append(text[i : j + len(end)])
    if parts:
        return "\n".join(parts)
    return text


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _summarize(rows: list[dict]) -> dict:
    f1 = [r["factuality_f1_score"] for r in rows]
    rew = [r["reward"] for r in rows]
    sev = [r["severity_reward"] for r in rows]
    nerr = [r["num_errors"] for r in rows]
    lat = [r["collect_latency_s"] for r in rows if r.get("collect_latency_s") is not None]
    gold_n = [len(r["expected_errors"]) for r in rows]
    return {
        "n": len(rows),
        "f1_mean": round(sum(f1) / len(f1), 4) if f1 else None,
        "f1_p50": round(_pct(f1, 0.5), 4) if f1 else None,
        "f1_eq_1": sum(1 for x in f1 if x == 1.0),
        "f1_eq_0": sum(1 for x in f1 if x == 0.0),
        "f1_partial": sum(1 for x in f1 if 0.0 < x < 1.0),
        "reward_mean": round(sum(rew) / len(rew), 4) if rew else None,
        "severity_match": sum(1 for x in sev if x == 1.0),
        "severity_match_rate": round(sum(sev) / len(sev), 4) if sev else None,
        "num_errors_mean": round(sum(nerr) / len(nerr), 3) if nerr else None,
        "gold_errors_mean": round(sum(gold_n) / len(gold_n), 3) if gold_n else None,
        "latency_mean_s": round(sum(lat) / len(lat), 2) if lat else None,
        "latency_p50_s": round(_pct(lat, 0.5), 2) if lat else None,
        "latency_p95_s": round(_pct(lat, 0.95), 2) if lat else None,
        "gold_n_hist": dict(sorted(Counter(gold_n).items())),
    }


def _load_gold(path: str) -> tuple[dict[str, dict], list[str]]:
    gold: dict[str, dict] = {}
    collisions: list[str] = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = _fingerprint(row.get("responses_create_params"))
            if not key:
                collisions.append(f"gold line {i}: empty fingerprint")
                continue
            rec = {
                "line": i,
                "id": row.get("id"),
                "expected_errors": row.get("expected_errors") or [],
                "hallucination_severity": row.get("hallucination_severity"),
            }
            if key in gold:
                collisions.append(f"gold line {i} collides with line {gold[key]['line']}")
            gold[key] = rec
    return gold, collisions


def _debug_unmatched(row: dict, gold: dict[str, dict]) -> dict:
    params = row.get("responses_create_params")
    fp = _fingerprint(params)
    sample = next(iter(gold), None)
    return {
        "output_keys": sorted(row.keys()),
        "has_responses_create_params": isinstance(params, dict),
        "output_fp_chars": len(fp) if fp else 0,
        "output_fp_preview": (fp or "")[:240],
        "gold_fp_count": len(gold),
        "gold_fp_preview": (sample or "")[:240],
    }


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "usage: python3 scripts/summarize_gold_split.py <output.jsonl> <gold.jsonl>",
            file=sys.stderr,
        )
        sys.exit(2)
    out_path = sys.argv[1]
    gold_path = sys.argv[2]
    gold, collisions = _load_gold(gold_path)
    empty, filled, unmatched = [], [], []
    debug = None
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = _fingerprint(row.get("responses_create_params"))
            g = gold.get(key) if key else None
            if g is None:
                unmatched.append(row)
                if debug is None:
                    debug = _debug_unmatched(row, gold)
                continue
            rec = {
                "factuality_f1_score": float(row.get("factuality_f1_score") or 0.0),
                "reward": float(row.get("reward") or 0.0),
                "severity_reward": float(row.get("severity_reward") or 0.0),
                "num_errors": float(row.get("num_errors") or 0.0),
                "collect_latency_s": row.get("collect_latency_s"),
                "expected_errors": g["expected_errors"],
            }
            if g["expected_errors"]:
                filled.append(rec)
            else:
                empty.append(rec)
    report = {
        "matched": len(empty) + len(filled),
        "unmatched": len(unmatched),
        "empty_gold": _summarize(empty) if empty else {"n": 0},
        "filled_gold": _summarize(filled) if filled else {"n": 0},
        "all_matched": _summarize(empty + filled) if empty or filled else {"n": 0},
    }
    if collisions:
        report["gold_fingerprint_notes"] = collisions[:10]
    if unmatched:
        report["join_debug"] = debug
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

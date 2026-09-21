#!/usr/bin/env python3
"""Report empty vs filled F1 and per-step timings from a collect metrics sidecar."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def _nums(rows: list[dict], key: str) -> list[float]:
    out = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)):
            out.append(float(val))
    return out


def _group_summary(rows: list[dict]) -> dict:
    f1 = _nums(rows, "factuality_f1_score")
    return {
        "n": len(rows),
        "f1_mean": _mean(f1),
        "f1_eq_1": sum(1 for x in f1 if x == 1.0),
        "f1_eq_0": sum(1 for x in f1 if x == 0.0),
        "f1_partial": sum(1 for x in f1 if 0.0 < x < 1.0),
        "reward_mean": _mean(_nums(rows, "reward")),
        "latency_mean_s": _mean(_nums(rows, "collect_latency_s")),
        "t_judges_wall_s": _mean(_nums(rows, "t_judges_wall_s")),
        "t_count_judge_s": _mean(_nums(rows, "t_count_judge_s")),
        "t_yes_no_matcher_s": _mean(_nums(rows, "t_yes_no_matcher_s")),
        "t_policy_generate_s": _mean(_nums(rows, "t_policy_generate_s")),
        "t_search_s": _mean(_nums(rows, "t_search_s")),
        "n_policy_steps_mean": _mean(_nums(rows, "n_policy_steps")),
        "n_searches_mean": _mean(_nums(rows, "n_searches")),
        "format_miss_filled": sum(
            1
            for r in rows
            if r.get("gold_empty") is False
            and (r.get("factuality_f1_score") or 0) == 0
            and not r.get("t_yes_no_matcher_s")
        ),
    }


def main() -> None:
    path = Path(sys.argv[1])
    if path.name.endswith(".jsonl") and not path.name.endswith(".metrics.jsonl"):
        metrics = path.with_name(path.name[: -len(".jsonl")] + ".metrics.jsonl")
        if metrics.exists():
            path = metrics
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    empty = [r for r in rows if r.get("gold_empty") is True]
    filled = [r for r in rows if r.get("gold_empty") is False]
    unknown = [r for r in rows if r.get("gold_empty") is None]
    report = {
        "file": str(path),
        "n": len(rows),
        "all": _group_summary(rows),
        "empty_gold": _group_summary(empty),
        "filled_gold": _group_summary(filled),
        "unknown_gold": len(unknown),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

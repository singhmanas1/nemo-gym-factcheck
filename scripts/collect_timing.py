#!/usr/bin/env python3
"""Report per-sample p95 latency and batch throughput from a collect JSONL.

After ``ng_collect_rollouts`` (with the timing patch), each row stores
``collect_latency_s`` (e2e ``/run`` wall time), ``collect_job_t0``, and
``collect_finished_at``. Throughput is job wall time, not 60 / mean latency —
concurrent samples overlap.

Usage:
  python3 scripts/collect_timing.py /path/to/factcheck_output.jsonl
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from typing import Any, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. ``q`` is in ``[0, 1]`` (p95 → 0.95)."""
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    weight = pos - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def summarize_collect_timing(latencies: Sequence[float], wall_s: float) -> dict[str, float]:
    n = len(latencies)
    wall = float(wall_s)
    if n == 0:
        return {
            "n": 0.0,
            "latency_mean_s": 0.0,
            "latency_p50_s": 0.0,
            "latency_p95_s": 0.0,
            "latency_max_s": 0.0,
            "wall_s": wall,
            "samples_per_min": 0.0,
        }
    total = float(sum(latencies))
    return {
        "n": float(n),
        "latency_mean_s": total / n,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
        "latency_max_s": float(max(latencies)),
        "wall_s": wall,
        "samples_per_min": (n / wall * 60.0) if wall > 0 else 0.0,
    }


def format_collect_timing(timing: dict[str, float]) -> dict[str, Any]:
    return {
        "n": int(timing["n"]),
        "latency_mean_s": round(timing["latency_mean_s"], 3),
        "latency_p50_s": round(timing["latency_p50_s"], 3),
        "latency_p95_s": round(timing["latency_p95_s"], 3),
        "latency_max_s": round(timing["latency_max_s"], 3),
        "wall_s": round(timing["wall_s"], 3),
        "samples_per_min": round(timing["samples_per_min"], 2),
    }


def _job_wall_s(rows: list[dict]) -> float | None:
    t0s = [float(r["collect_job_t0"]) for r in rows if isinstance(r.get("collect_job_t0"), (int, float))]
    finished = [
        float(r["collect_finished_at"]) for r in rows if isinstance(r.get("collect_finished_at"), (int, float))
    ]
    latencies = [
        float(r["collect_latency_s"]) for r in rows if isinstance(r.get("collect_latency_s"), (int, float))
    ]
    if t0s and finished:
        wall = max(finished) - min(t0s)
        if wall > 0:
            return wall
    if latencies:
        # Same job, missing clocks: fully serial lower bound is sum; overlap unknown.
        return None
    return None


def timing_from_jsonl(path: str) -> list[dict[str, Any]]:
    """One timing dict per collect job (grouped by ``collect_job_t0``)."""
    jobs: dict[Any, list[dict]] = defaultdict(list)
    untimed = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row.get("collect_latency_s"), (int, float)):
                untimed += 1
                continue
            key = row.get("collect_job_t0")
            jobs[key].append(row)

    reports = []
    for job_t0, rows in jobs.items():
        latencies = [float(r["collect_latency_s"]) for r in rows]
        wall = _job_wall_s(rows)
        if wall is None:
            # Percentiles still valid; throughput would be a guess.
            summary = summarize_collect_timing(latencies, wall_s=max(latencies) if latencies else 0.0)
            formatted = format_collect_timing(summary)
            formatted["samples_per_min"] = None
            formatted["wall_s"] = None
            formatted["wall_note"] = "missing collect_job_t0/collect_finished_at; p95 only"
        else:
            formatted = format_collect_timing(summarize_collect_timing(latencies, wall))
        formatted["collect_job_t0"] = job_t0
        reports.append(formatted)
    if untimed and not reports:
        reports.append(
            {
                "n": untimed,
                "note": "no collect_latency_s on these rows; re-run ng_collect_rollouts",
            }
        )
    elif untimed:
        reports.append({"untimed_rows": untimed})
    return reports


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python3 scripts/collect_timing.py <output.jsonl>", file=sys.stderr)
        sys.exit(2)
    reports = timing_from_jsonl(sys.argv[1])
    print(json.dumps({"timing": reports if len(reports) != 1 else reports[0]}, indent=4))


if __name__ == "__main__":
    main()

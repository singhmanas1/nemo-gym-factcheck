#!/usr/bin/env python3
"""Convert an audited RLHF JSONL into NeMo Gym collect_rollouts input.

Keeps responses_create_params (including search_wiki tools) and writes gold
labels to a sidecar file that is not sent to Gym.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="Audited dataset JSONL")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("factcheck_input.jsonl"),
        help="Gym input JSONL",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("factcheck_labels.jsonl"),
        help="Sidecar labels JSONL",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    args = parser.parse_args()

    n = 0
    with args.src.open() as fin, args.out.open("w") as fout, args.labels.open("w") as lout:
        for line in fin:
            if args.limit and n >= args.limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            params = obj.get("responses_create_params")
            if not params:
                raise SystemExit(f"row {n} missing responses_create_params")
            row = {
                "id": n,
                "source_id": obj.get("id", n),
                "responses_create_params": params,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            label = {
                "id": n,
                "source_id": obj.get("id", n),
                "hallucination_severity": obj.get("hallucination_severity"),
                "expected_errors": obj.get("expected_errors")
                or obj.get("factual_errors"),
            }
            lout.write(json.dumps(label, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows -> {args.out} and {args.labels}")


if __name__ == "__main__":
    main()

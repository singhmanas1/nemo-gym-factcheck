#!/usr/bin/env python3
"""Write a small empty+filled gold JSONL, skipping the Chris Rock crossword."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SKIP_NEEDLES = (
    "rock on stage (pattern: c??is)",
    "won his first grammy award in 1995",
)


def _blob(row: dict) -> str:
    parts = [json.dumps(row.get("expected_errors") or []), str(row.get("id") or "")]
    params = row.get("responses_create_params") or {}
    inp = params.get("input") if isinstance(params, dict) else None
    if isinstance(inp, list) and inp:
        content = inp[0].get("content") if isinstance(inp[0], dict) else None
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts).lower()


def _is_chris_rock(row: dict) -> bool:
    blob = _blob(row)
    return any(needle in blob for needle in SKIP_NEEDLES)


def _gold_n(row: dict) -> int:
    gold = row.get("expected_errors")
    if not isinstance(gold, list):
        return 0
    return sum(1 for x in gold if isinstance(x, str) and x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--n-empty", type=int, default=4)
    parser.add_argument("--n-filled", type=int, default=6)
    args = parser.parse_args()

    empty: list[str] = []
    filled: list[str] = []
    skipped = 0
    with args.src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if _is_chris_rock(row):
                skipped += 1
                continue
            target = empty if _gold_n(row) == 0 else filled
            want = args.n_empty if target is empty else args.n_filled
            if len(target) >= want:
                if len(empty) >= args.n_empty and len(filled) >= args.n_filled:
                    break
                continue
            target.append(line if line.endswith("\n") else line + "\n")

    if len(empty) < args.n_empty or len(filled) < args.n_filled:
        raise SystemExit(
            f"not enough rows: empty {len(empty)}/{args.n_empty} "
            f"filled {len(filled)}/{args.n_filled} (skipped chris-rock={skipped})"
        )

    rows = empty + filled
    args.dst.write_text("".join(line if line.endswith("\n") else line + "\n" for line in rows))
    print(
        f"wrote {len(rows)} rows -> {args.dst} "
        f"(empty={len(empty)} filled={len(filled)} skipped_chris_rock={skipped})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Copy one JSONL row by 1-based line number."""
import sys
from pathlib import Path

src = Path(sys.argv[1])
line_no = int(sys.argv[2])
dst = Path(sys.argv[3])
with src.open() as f:
    for i, line in enumerate(f, 1):
        if i == line_no:
            dst.write_text(line if line.endswith("\n") else line + "\n")
            print(f"wrote line {line_no} -> {dst}")
            raise SystemExit(0)
raise SystemExit(f"no line {line_no} in {src}")

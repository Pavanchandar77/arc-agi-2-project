"""Turn a directory of ARC task files into the two files the runners expect.

The ARC repositories store one JSON per task. Both runners expect the Kaggle
layout instead: a single challenges file keyed by task id with test outputs
withheld, and a matching solutions file holding those outputs for local
scoring. Pointing a runner at the directory fails, so this bridges the two.

Withholding the test outputs is the point, not a formality. Scoring a
submission against answers the solver could have read is not scoring.

    python scripts/make_challenges.py --split evaluation --out data/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def find_split_dir(split: str, data_root: Optional[str]) -> Path:
    if data_root:
        candidate = Path(data_root) / split
        if candidate.is_dir():
            return candidate
        raise SystemExit(f"no such directory: {candidate}")
    for base in (REPO / "ARC-AGI-2" / "data", REPO / "ARC-AGI" / "data"):
        candidate = base / split
        if candidate.is_dir():
            return candidate
    raise SystemExit(
        f"could not find a '{split}' split under ARC-AGI-2/data or ARC-AGI/data. "
        f"Clone the data first, or pass --data-root."
    )


def build(split_dir: Path) -> tuple[dict, dict]:
    """Return (challenges, solutions). Test outputs appear only in solutions."""
    challenges: dict[str, dict] = {}
    solutions: dict[str, list] = {}
    for path in sorted(split_dir.glob("*.json")):
        task_id = path.stem
        payload = json.loads(path.read_text(encoding="utf-8"))
        tests = payload.get("test") or []
        challenges[task_id] = {
            "train": payload.get("train") or [],
            # The solver must never be handed the answer it is scored against.
            "test": [{"input": t["input"]} for t in tests],
        }
        if all("output" in t for t in tests):
            solutions[task_id] = [t["output"] for t in tests]
    return challenges, solutions


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="build Kaggle-format ARC files")
    p.add_argument("--split", default="evaluation")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="data/eval")
    args = p.parse_args(argv)

    split_dir = find_split_dir(args.split, args.data_root)
    challenges, solutions = build(split_dir)
    if not challenges:
        raise SystemExit(f"no task files found in {split_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ch_path = out_dir / "arc-agi_test_challenges.json"
    sol_path = out_dir / "arc-agi_test_solutions.json"
    ch_path.write_text(json.dumps(challenges), encoding="utf-8")
    if solutions:
        sol_path.write_text(json.dumps(solutions), encoding="utf-8")

    print(json.dumps({
        "split": args.split,
        "source": str(split_dir),
        "tasks": len(challenges),
        "with_answers": len(solutions),
        "challenges": str(ch_path),
        "solutions": str(sol_path) if solutions else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

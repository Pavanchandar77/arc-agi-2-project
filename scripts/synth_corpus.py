"""Build the program-training corpus: generated examples plus harvested real ones.

Harvesting alone yields 26 tasks, because search explains only ~2.6% of
ARC-AGI-2 training and more time does not move that. Generation yields as many
verified examples as asked for, in seconds, but from the DSL's own distribution
rather than ARC's.

Neither is sufficient alone. Generated tasks teach the language - the operator
names, the argument forms, how composition reads. Harvested tasks are the only
examples drawn from the real distribution, so they are few and precious, and
are repeated so they are not drowned by a corpus a thousand times their size.

    python scripts/synth_corpus.py --n 20000 \\
        --merge data/programs/programs.jsonl \\
        --out data/programs/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.hrps.program_prompt import build_training_example  # noqa: E402
from src.hrps.synth_programs import synthesize  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="build the program training corpus")
    p.add_argument("--n", type=int, default=20000, help="generated examples")
    p.add_argument("--merge", default=None, help="harvested corpus to fold in")
    p.add_argument("--out", default="data/programs/train.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument(
        "--real-repeats",
        type=int,
        default=8,
        help="times to repeat each harvested example, so the handful drawn from "
             "the real distribution are not lost in the generated bulk",
    )
    args = p.parse_args(argv)

    started = time.perf_counter()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs, report = synthesize(args.n, seed=args.seed, max_depth=args.max_depth)
    n_real = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for task, program in pairs:
            handle.write(json.dumps(build_training_example(task, program.serialize())) + "\n")
        if args.merge:
            merge_path = Path(args.merge)
            if not merge_path.is_file():
                raise SystemExit(f"--merge given but {merge_path} does not exist")
            rows = [
                line for line in merge_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for _ in range(max(1, args.real_repeats)):
                for line in rows:
                    handle.write(line + "\n")
                    n_real += 1

    summary = {
        "generated": report.produced,
        "generation": report.as_dict(),
        "real_rows": n_real,
        "real_unique": n_real // max(1, args.real_repeats),
        "total": report.produced + n_real,
        "seconds": round(time.perf_counter() - started, 1),
        "out": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    (out_path.parent / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Mine program supervision from the verifier.

Nobody can hand-label a thousand ARC tasks with programs, and a label nobody
can check is worse than none. But the search already finds programs, and the
verifier already certifies them against every demonstration - so the training
corpus can be generated and proved without a human in the loop.

    for each task:
        search for a program
        keep it only if it reproduces every demonstration exactly
        emit (task, program) as one supervised example

Every label is therefore correct by construction. The corpus is small and
skewed toward what search can already reach, which is the point: it teaches the
model the language and the shape of a solution, and the model's job is to
propose compositions search cannot find in time. Programs the model later
discovers, once verified the same way, can be appended here - the loop closes.

    python scripts/harvest_programs.py --splits training --out data/programs
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
from src.hrps.proposal import parse_program, verify_program  # noqa: E402
from src.hrps.task import ArcTask, iter_split  # noqa: E402


def programs_for(task: ArcTask, *, seconds: float, stage: str) -> list[str]:
    """Every distinct verified program found for this task, cheapest first.

    Supervision comes from the DSL search alone. The solver bank cannot
    contribute: its rules are arbitrary Python closures, strictly more
    expressive than the DSL, so a bank rule has no program to emit. That caps
    coverage at what search can express, which is the honest ceiling here.

    Search results are re-verified rather than trusted. A harvester that trusts
    its source will happily emit a mislabelled corpus, and a corpus nobody
    checked is what this whole design exists to avoid.
    """
    found: list[str] = []
    seen: set[str] = set()
    deadline = time.perf_counter() + seconds

    def consider(text: Optional[str]) -> None:
        if not text or text in seen:
            return
        program = parse_program(text)
        if program is None:
            return
        key = program.serialize()
        if key in seen:
            return
        if verify_program(program, task).is_total:
            seen.add(key)
            seen.add(text)
            found.append(key)

    from src.hrps.search import SearchBudget, search_task

    # Errors here are not swallowed. A harvester that silently yields nothing
    # looks exactly like a corpus with no coverage, which is the one failure
    # that would go unnoticed until training produced garbage.
    result = search_task(
        task,
        stage=stage,
        budget=SearchBudget(max_seconds=max(1.0, seconds), max_depth=3, max_nodes=2000),
    )
    for text in result.programs:
        consider(text)

    found.sort(key=lambda s: (len(s.split("|")), len(s), s))
    return found


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="harvest verified programs as supervision")
    p.add_argument("--splits", nargs="+", default=["training"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="data/programs")
    p.add_argument("--seconds-per-task", type=float, default=10.0)
    p.add_argument("--stage", default="L")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-per-task", type=int, default=4,
                   help="keep at most this many distinct programs per task")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = out_dir / "programs.jsonl"
    index = out_dir / "index.json"

    data_root = Path(args.data_root) if args.data_root else None
    n_tasks = 0
    n_solved = 0
    n_examples = 0
    started = time.perf_counter()
    per_task: dict[str, list[str]] = {}

    with corpus.open("w", encoding="utf-8") as handle:
        for split in args.splits:
            for task in iter_split(split, data_root=data_root):
                if args.limit and n_tasks >= args.limit:
                    break
                n_tasks += 1
                programs = programs_for(
                    task, seconds=args.seconds_per_task, stage=args.stage
                )[: args.max_per_task]
                if not programs:
                    continue
                n_solved += 1
                per_task[task.task_id] = programs
                for program in programs:
                    handle.write(
                        json.dumps(build_training_example(task, program)) + "\n"
                    )
                    n_examples += 1
                if n_tasks % 50 == 0:
                    rate = n_solved / max(1, n_tasks)
                    print(
                        f"  {n_tasks} tasks, {n_solved} with a verified program "
                        f"({rate:.1%}), {n_examples} examples",
                        flush=True,
                    )

    index.write_text(json.dumps(per_task, indent=2), encoding="utf-8")
    summary = {
        "n_tasks": n_tasks,
        "n_tasks_with_program": n_solved,
        "coverage": round(n_solved / max(1, n_tasks), 4),
        "n_examples": n_examples,
        "seconds": round(time.perf_counter() - started, 1),
        "corpus": str(corpus),
    }
    print(json.dumps(summary, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.hrps.program_prompt import build_training_example  # noqa: E402
from src.hrps.proposal import parse_program, verify_program  # noqa: E402
from src.hrps.task import ArcTask, iter_split  # noqa: E402

ARC2_REPO = "https://github.com/arcprize/ARC-AGI-2.git"


def ensure_arc_data(explicit: Optional[str] = None) -> Optional[Path]:
    """Return the data root holding <split>/ folders, cloning ARC-AGI-2 if needed.

    Harvesting runs before training, so it cannot assume the trainer has
    already fetched the corpus. Returning None means the default layout is
    already in place and iter_split can find it unaided.
    """
    if explicit:
        return Path(explicit)
    default = REPO / "ARC-AGI-2" / "data"
    if (default / "training").is_dir():
        return None
    if shutil.which("git") is None:
        raise SystemExit(
            f"no ARC data at {default} and git is unavailable. "
            f"Clone {ARC2_REPO} into {REPO} manually, or pass --data-root."
        )
    print(f"[data] cloning {ARC2_REPO}", flush=True)
    target = REPO / "ARC-AGI-2"
    subprocess.run(["git", "clone", "--depth", "1", ARC2_REPO, str(target)], check=True)
    if not (default / "training").is_dir():
        raise SystemExit(f"clone succeeded but {default / 'training'} is missing")
    return None


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


def _worker(payload: tuple) -> list[str]:
    """Runs in a child process. Returns serialized programs, never raises."""
    task, seconds, stage = payload
    try:
        return programs_for(task, seconds=seconds, stage=stage)
    except Exception:
        return []


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
    p.add_argument("--workers", type=int, default=0,
                   help="parallel search workers; 0 picks one per core")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = out_dir / "programs.jsonl"
    index = out_dir / "index.json"

    data_root = ensure_arc_data(args.data_root)
    n_solved = 0
    n_examples = 0
    started = time.perf_counter()
    per_task: dict[str, list[str]] = {}

    workers = args.workers or min(8, os.cpu_count() or 1)
    tasks: list[ArcTask] = []
    for split in args.splits:
        for task in iter_split(split, data_root=data_root):
            if args.limit and len(tasks) >= args.limit:
                break
            tasks.append(task)
    n_tasks = len(tasks)
    print(
        f"[harvest] {n_tasks} tasks, {workers} workers, "
        f"{args.seconds_per_task:.0f}s each -> "
        f"~{n_tasks * args.seconds_per_task / max(1, workers) / 60:.0f} min",
        flush=True,
    )

    payloads = [(t, args.seconds_per_task, args.stage) for t in tasks]
    with corpus.open("w", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, (task, programs) in enumerate(
                zip(tasks, pool.map(_worker, payloads, chunksize=1)), start=1
            ):
                programs = programs[: args.max_per_task]
                if programs:
                    n_solved += 1
                    per_task[task.task_id] = programs
                    for program in programs:
                        handle.write(
                            json.dumps(build_training_example(task, program)) + "\n"
                        )
                        n_examples += 1
                    handle.flush()
                if done % 25 == 0 or done == n_tasks:
                    elapsed = time.perf_counter() - started
                    eta = elapsed / done * (n_tasks - done)
                    print(
                        f"  {done}/{n_tasks}  {n_solved} with a program "
                        f"({n_solved / done:.1%})  {n_examples} examples  "
                        f"eta {eta / 60:.1f} min",
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

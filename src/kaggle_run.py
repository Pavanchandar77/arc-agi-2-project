"""Kaggle submission runner for the ARC Prize competition.

Design constraints, all of which the Kaggle environment actually imposes:

* **No internet.** Nothing here downloads, and nothing imports torch,
  transformers, or any optional dependency at module scope.
* **A hard wall clock.** The notebook is killed at the limit, so the runner
  owns a global deadline, budgets each task against the tasks still to come,
  and writes ``submission.json`` incrementally. A kill at any point leaves a
  complete, schema-valid file on disk.
* **Every task must appear.** A crash, a timeout, or a hung worker yields a
  placeholder entry rather than a missing key, because a missing key scores the
  whole submission zero.

Run it as ``python -m src.kaggle_run`` or call :func:`run` from a notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.arc_solve import blank_entry, solve_task, submission_entry
from src.hrps.task import ArcTask, parse_task

# Kaggle mounts the competition data read-only under /kaggle/input/<slug>/.
CHALLENGE_BASENAMES = (
    "arc-agi_test_challenges.json",
    "arc-agi_evaluation_challenges.json",
)
SOLUTION_BASENAMES = (
    "arc-agi_test_solutions.json",
    "arc-agi_evaluation_solutions.json",
)
SEARCH_ROOTS = ("/kaggle/input", "input", "data")


@dataclass
class RunConfig:
    challenges: Path
    solutions: Optional[Path]
    output: Path
    total_seconds: float
    per_task_seconds: float
    min_task_seconds: float
    workers: int
    use_search: bool
    stage: str
    limit: Optional[int]
    verbose: bool


def find_challenges(explicit: Optional[str] = None) -> Path:
    """Locate the challenge file without any assumption about the dataset slug."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"challenge file not found: {p}")
        return p
    for root in SEARCH_ROOTS:
        base = Path(root)
        if not base.is_dir():
            continue
        for name in CHALLENGE_BASENAMES:
            hits = sorted(base.glob(f"*/{name}")) + sorted(base.glob(name))
            if hits:
                return hits[0]
    # Last resort: any json whose name mentions test/evaluation challenges.
    for root in SEARCH_ROOTS:
        base = Path(root)
        if not base.is_dir():
            continue
        hits = sorted(base.glob("**/*challenges*.json"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "no ARC challenge file found. Pass --challenges explicitly. "
        f"Looked under {', '.join(SEARCH_ROOTS)} for {', '.join(CHALLENGE_BASENAMES)}."
    )


def find_solutions(challenges: Path, explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for name in SOLUTION_BASENAMES:
        cand = challenges.parent / name
        if cand.is_file():
            return cand
    return None


def load_tasks(path: Path) -> list[ArcTask]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[ArcTask] = []
    for task_id in sorted(payload):
        try:
            tasks.append(parse_task(task_id, payload[task_id], "test"))
        except Exception:
            # Keep the id; the runner will emit a placeholder for it.
            tasks.append(ArcTask(task_id=task_id, train=(), test=(), split="test"))
    return tasks


def _n_test(task: ArcTask, raw: dict[str, Any]) -> int:
    if task.test:
        return len(task.test)
    try:
        return max(1, len(raw["test"]))
    except Exception:
        return 1


def validate_submission(submission: dict[str, Any], expected_ids: list[str]) -> list[str]:
    """Return a list of schema problems. Empty list means the file is submittable."""
    problems: list[str] = []
    missing = [t for t in expected_ids if t not in submission]
    if missing:
        problems.append(f"{len(missing)} task ids missing (e.g. {missing[:3]})")
    extra = [t for t in submission if t not in set(expected_ids)]
    if extra:
        problems.append(f"{len(extra)} unexpected task ids (e.g. {extra[:3]})")
    for task_id, entry in submission.items():
        if not isinstance(entry, list) or not entry:
            problems.append(f"{task_id}: entry must be a non-empty list")
            continue
        for i, item in enumerate(entry):
            if not isinstance(item, dict):
                problems.append(f"{task_id}[{i}]: not an object")
                continue
            for key in ("attempt_1", "attempt_2"):
                grid = item.get(key)
                if not isinstance(grid, list) or not grid:
                    problems.append(f"{task_id}[{i}].{key}: not a non-empty list")
                    continue
                if len(grid) > 30:
                    problems.append(f"{task_id}[{i}].{key}: {len(grid)} rows > 30")
                width = None
                for row in grid:
                    if not isinstance(row, list) or not row:
                        problems.append(f"{task_id}[{i}].{key}: bad row")
                        break
                    if width is None:
                        width = len(row)
                        if width > 30:
                            problems.append(f"{task_id}[{i}].{key}: {width} cols > 30")
                    elif len(row) != width:
                        problems.append(f"{task_id}[{i}].{key}: ragged rows")
                        break
                    if any(not isinstance(v, int) or v < 0 or v > 9 for v in row):
                        problems.append(f"{task_id}[{i}].{key}: cell outside 0..9")
                        break
    return problems[:50]


def _write(output: Path, submission: dict[str, Any]) -> None:
    """Atomic-ish write so a kill mid-flush cannot truncate the real file."""
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(submission), encoding="utf-8")
    os.replace(tmp, output)


class _Overrun(Exception):
    pass


def _on_alarm(signum, frame):  # pragma: no cover - signal path
    raise _Overrun("task exceeded its hard budget")


def _worker(payload: tuple) -> dict[str, Any]:
    """Runs in a child process. Returns a plain dict so nothing exotic is pickled.

    The solver's own deadline is cooperative, so a SIGALRM backs it up: a task
    that wedges inside a single long operation still returns a fallback rather
    than holding a pool slot forever.
    """
    import signal

    task_id, raw, seconds, use_search, stage = payload
    armed = False
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(1.0, seconds) + 15.0)
        armed = True
    except Exception:
        pass
    try:
        task = parse_task(task_id, raw, "test")
        outcome = solve_task(task, seconds=seconds, use_search=use_search, search_stage=stage)
        return {
            "task_id": task_id,
            "entry": submission_entry(outcome),
            "meta": outcome.as_dict(),
        }
    except BaseException as exc:
        n = 1
        try:
            n = max(1, len(raw["test"]))
        except Exception:
            pass
        return {
            "task_id": task_id,
            "entry": blank_entry(n),
            "meta": {"task_id": task_id, "source": "worker_error", "error": repr(exc)[:200]},
        }
    finally:
        if armed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass


def run(config: RunConfig) -> dict[str, Any]:
    started = time.perf_counter()
    hard_deadline = started + config.total_seconds
    raw_payload = json.loads(config.challenges.read_text(encoding="utf-8"))
    task_ids = sorted(raw_payload)
    if config.limit is not None:
        task_ids = task_ids[: config.limit]

    # Seed every id up front. From this line on, the file on disk is always
    # complete and always submittable, whatever happens next.
    submission: dict[str, Any] = {}
    for task_id in task_ids:
        try:
            n = max(1, len(raw_payload[task_id]["test"]))
        except Exception:
            n = 1
        submission[task_id] = blank_entry(n)
    _write(config.output, submission)

    metas: list[dict[str, Any]] = []
    done = 0
    n_total = len(task_ids)

    def budget_for(index: int) -> float:
        left = hard_deadline - time.perf_counter()
        remaining = max(1, n_total - index)
        # Reserve a slice for the final write and validation.
        fair = (left - 5.0) / remaining * max(1, config.workers)
        return max(config.min_task_seconds, min(config.per_task_seconds, fair))

    if config.workers > 1:
        import multiprocessing as mp

        try:
            ctx = mp.get_context("fork")
        except ValueError:  # pragma: no cover - non-fork platforms
            ctx = mp.get_context()
        # A Pool, not a ProcessPoolExecutor: terminate() actually kills the
        # workers. ProcessPoolExecutor.shutdown(wait=False) leaves them running
        # and its atexit hook then blocks the interpreter past the deadline.
        pool = ctx.Pool(processes=config.workers)
        try:
            inflight: dict[Any, str] = {}
            cursor = 0
            window = config.workers * 2
            while cursor < n_total or inflight:
                if time.perf_counter() > hard_deadline:
                    if config.verbose:
                        print("[kaggle_run] global deadline reached; stopping", flush=True)
                    break
                while (
                    cursor < n_total
                    and len(inflight) < window
                    and time.perf_counter() < hard_deadline
                ):
                    task_id = task_ids[cursor]
                    handle = pool.apply_async(
                        _worker,
                        (
                            (
                                task_id,
                                raw_payload[task_id],
                                budget_for(cursor),
                                config.use_search,
                                config.stage,
                            ),
                        ),
                    )
                    inflight[handle] = task_id
                    cursor += 1
                if not inflight:
                    break
                progressed = False
                for handle in list(inflight):
                    if not handle.ready():
                        continue
                    task_id = inflight.pop(handle)
                    progressed = True
                    done += 1
                    try:
                        result = handle.get(timeout=5.0)
                    except Exception as exc:
                        metas.append(
                            {"task_id": task_id, "source": "pool_error", "error": repr(exc)[:200]}
                        )
                        continue
                    submission[result["task_id"]] = result["entry"]
                    metas.append(result["meta"])
                    if done % 10 == 0:
                        _write(config.output, submission)
                        if config.verbose:
                            _progress(done, n_total, metas, started)
                if not progressed:
                    time.sleep(0.05)
        finally:
            _write(config.output, submission)
            pool.terminate()
            pool.join()
    else:
        for idx, task_id in enumerate(task_ids):
            if time.perf_counter() > hard_deadline:
                if config.verbose:
                    print("[kaggle_run] global deadline reached; stopping", flush=True)
                break
            result = _worker(
                (task_id, raw_payload[task_id], budget_for(idx), config.use_search, config.stage)
            )
            submission[result["task_id"]] = result["entry"]
            metas.append(result["meta"])
            done += 1
            if done % 10 == 0:
                _write(config.output, submission)
                if config.verbose:
                    _progress(done, n_total, metas, started)
        _write(config.output, submission)

    problems = validate_submission(submission, task_ids)
    if problems:
        # Never ship a malformed file: replace the offending entries.
        for task_id in task_ids:
            entry = submission.get(task_id)
            if not isinstance(entry, list) or not entry:
                submission[task_id] = blank_entry(1)
        problems = validate_submission(submission, task_ids)
        _write(config.output, submission)

    report: dict[str, Any] = {
        "challenges": str(config.challenges),
        "output": str(config.output),
        "n_tasks": n_total,
        "n_attempted": done,
        "wall_clock_sec": round(time.perf_counter() - started, 2),
        "schema_problems": problems,
        "sources": _count(metas, "source"),
        "n_verified": sum(1 for m in metas if m.get("verified")),
    }
    if config.solutions is not None and config.solutions.is_file():
        report["score"] = score_submission(submission, config.solutions)
    return report


def _count(metas: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in metas:
        out[str(m.get(key, "?"))] = out.get(str(m.get(key, "?")), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _progress(done: int, total: int, metas: list[dict[str, Any]], started: float) -> None:
    ver = sum(1 for m in metas if m.get("verified"))
    print(
        f"[kaggle_run] {done}/{total} verified={ver} "
        f"elapsed={time.perf_counter() - started:.0f}s",
        flush=True,
    )


def score_submission(submission: dict[str, Any], solutions_path: Path) -> dict[str, Any]:
    """Official pass@2: a test input counts if either attempt matches exactly."""
    truth = json.loads(solutions_path.read_text(encoding="utf-8"))
    # Score only what was actually attempted, so a --limit run reports honestly.
    truth = {k: v for k, v in truth.items() if k in submission}
    n_inputs = n_correct = 0
    solved_tasks = 0
    for task_id, gts in truth.items():
        entry = submission.get(task_id) or []
        task_ok = bool(gts)
        for i, gt in enumerate(gts):
            n_inputs += 1
            item = entry[i] if i < len(entry) else {}
            hit = item.get("attempt_1") == gt or item.get("attempt_2") == gt
            if hit:
                n_correct += 1
            else:
                task_ok = False
        if task_ok:
            solved_tasks += 1
    return {
        "tasks": len(truth),
        "tasks_solved": solved_tasks,
        "task_solve_rate": round(solved_tasks / len(truth), 6) if truth else 0.0,
        "test_inputs": n_inputs,
        "test_inputs_correct": n_correct,
        "pass_at_2": round(n_correct / n_inputs, 6) if n_inputs else 0.0,
    }


def build_config(argv: Optional[list[str]] = None) -> RunConfig:
    p = argparse.ArgumentParser(description="ARC Prize Kaggle submission runner")
    p.add_argument("--challenges", default=None, help="path to *_challenges.json")
    p.add_argument("--solutions", default=None, help="path to *_solutions.json (local scoring)")
    p.add_argument("--output", default="submission.json")
    p.add_argument("--total-seconds", type=float, default=None, help="global wall-clock budget")
    p.add_argument("--per-task-seconds", type=float, default=60.0)
    p.add_argument("--min-task-seconds", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=0, help="0 = one per CPU")
    p.add_argument("--no-search", action="store_true", help="solver bank only")
    p.add_argument("--stage", default="L")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    challenges = find_challenges(args.challenges)
    workers = args.workers or max(1, (os.cpu_count() or 2))
    total = args.total_seconds
    if total is None:
        # Kaggle allows 12h; stop an hour early so commit + save always fits.
        total = float(os.environ.get("ARC_TOTAL_SECONDS", 11 * 3600))
    return RunConfig(
        challenges=challenges,
        solutions=find_solutions(challenges, args.solutions),
        output=Path(args.output),
        total_seconds=total,
        per_task_seconds=args.per_task_seconds,
        min_task_seconds=args.min_task_seconds,
        workers=workers,
        use_search=not args.no_search,
        stage=args.stage,
        limit=args.limit,
        verbose=not args.quiet,
    )


def main(argv: Optional[list[str]] = None) -> int:
    try:
        config = build_config(argv)
    except Exception:
        traceback.print_exc()
        # Even a config failure must leave a file behind if we know the ids.
        Path("submission.json").write_text("{}", encoding="utf-8")
        return 1
    report = run(config)
    print(json.dumps(report, indent=2))
    return 0 if not report["schema_problems"] else 1


if __name__ == "__main__":
    sys.exit(main())

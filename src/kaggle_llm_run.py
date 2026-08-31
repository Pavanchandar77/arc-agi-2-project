"""Two-phase Kaggle runner: symbolic pass on CPU, then the LLM on what is left.

The phases are separated because they want opposite things. The symbolic layer
is CPU-bound, embarrassingly parallel, and cheap; the model is GPU-bound and
must be loaded exactly once. Running them in one worker pool would load the
model N times and blow the memory budget.

    phase 1  all tasks, multiprocess, exact solver bank + DSL search
    phase 2  single process, model loaded once, only the tasks phase 1 missed
    phase 3  validate and write

Phase 1's answers are never overwritten by phase 2. A bank rule has replayed
every demonstration exactly; a sampled generation has not. When they disagree,
the verified answer wins.

`submission.json` is complete after phase 1, so losing the GPU, running out of
time, or failing to load the model costs the model's contribution and nothing
else.
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

from src.arc_solve import blank_entry
from src.hrps.grid import to_lists
from src.hrps.llm_solver import LlmConfig, LlmSolver, find_model_dir
from src.kaggle_run import (
    RunConfig,
    _write,
    build_config,
    find_challenges,
    run as run_symbolic,
    score_submission,
    validate_submission,
)


@dataclass
class LlmPhaseConfig:
    model_path: str
    adapter_path: Optional[str]
    seconds: float
    per_task_seconds: float
    ttt_steps: int
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    propose_programs: bool = True
    n_proposals: int = 8
    proposal_tokens: int = 256
    execution_guided: bool = True
    beam_width: int = 3
    max_depth: int = 4


def _unsolved_ids(report: dict[str, Any], submission: dict[str, Any]) -> list[str]:
    """Task ids whose entries came from the fallback layer, not a verified rule."""
    verified = set(report.get("verified_ids") or [])
    return [t for t in sorted(submission) if t not in verified]


def run_llm_phase(
    submission: dict[str, Any],
    raw_payload: dict[str, Any],
    targets: list[str],
    cfg: LlmPhaseConfig,
    output: Path,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fill `targets` with model predictions. Returns stats; never raises."""
    started = time.perf_counter()
    deadline = started + cfg.seconds
    solver = LlmSolver(
        LlmConfig(
            model_path=cfg.model_path,
            adapter_path=cfg.adapter_path,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=cfg.seed,
            ttt_steps=cfg.ttt_steps,
        )
    )
    if not solver.load(deadline=deadline):
        if verbose:
            print(f"[llm] load failed: {solver.stats.load_error}", flush=True)
        return {"skipped": "load_failed", **solver.stats.as_dict()}

    n_filled = 0
    n_program_solved = 0
    for i, task_id in enumerate(targets):
        left = deadline - time.perf_counter()
        if left <= 1.0:
            if verbose:
                print(f"[llm] deadline reached after {i}/{len(targets)} tasks", flush=True)
            break
        remaining_tasks = max(1, len(targets) - i)
        budget = min(cfg.per_task_seconds, max(1.0, (left - 1.0) / remaining_tasks))
        task = raw_payload.get(task_id)
        if not isinstance(task, dict) or not task.get("test"):
            continue
        entry = list(submission.get(task_id) or blank_entry(len(task["test"])))
        changed = False
        task_deadline = time.perf_counter() + budget

        # Programs first. A verified proposal reproduces every demonstration
        # exactly, so it outranks any ungrounded grid the model might emit -
        # and when none survives, nothing is written and the grid path still
        # gets its turn on the remaining budget.
        filled_by_program: set[int] = set()
        if cfg.propose_programs:
            filled_by_program = _program_pass(
                solver, task, task_id, entry, cfg, task_deadline
            )
            if filled_by_program:
                n_program_solved += 1
                changed = True

        # Only the test inputs a verified program could not answer fall through.
        # A program can verify on the demonstrations and still fail a
        # precondition on one test input, and that input still deserves a guess.
        for test_idx in range(len(task["test"])):
            if test_idx in filled_by_program:
                continue
            grids = solver.predict(task, test_idx, n_attempts=2, deadline=task_deadline)
            if test_idx >= len(entry):
                continue
            slot = dict(entry[test_idx])
            for key, grid in zip(("attempt_1", "attempt_2"), grids):
                if grid is not None:
                    slot[key] = to_lists(grid)
                    changed = True
            entry[test_idx] = slot
        if changed:
            submission[task_id] = entry
            n_filled += 1
        if (i + 1) % 5 == 0:
            _write(output, submission)
            if verbose:
                print(
                    f"[llm] {i + 1}/{len(targets)} filled={n_filled} "
                    f"elapsed={time.perf_counter() - started:.0f}s",
                    flush=True,
                )
    _write(output, submission)
    return {
        "n_targets": len(targets),
        "n_filled": n_filled,
        "n_program_verified": n_program_solved,
        **solver.stats.as_dict(),
    }


def _program_pass(solver, raw_task, task_id, entry, cfg, deadline) -> set[int]:
    """Sample programs, keep only what the demonstrations prove.

    Writes into `entry` in place and returns the test indices it answered, so
    the caller knows which inputs still need the grid path. Failure here is
    never fatal: a task the proposer cannot explain falls through untouched.
    """
    try:
        from src.hrps.program_solver import solve_by_proposal, solve_step_by_step
        from src.hrps.task import parse_task

        task = parse_task(task_id, raw_task, "test")
        result = None
        if cfg.execution_guided:
            # Run each operator before writing the next, so composition is
            # against what happened rather than what was assumed.
            result = solve_step_by_step(
                solver,
                task,
                beam_width=cfg.beam_width,
                max_depth=cfg.max_depth,
                temperature=cfg.temperature,
                max_new_tokens=cfg.proposal_tokens,
                deadline=deadline,
            )
        if result is None or not result.solved:
            # Blind whole-program proposal still gets its turn: a model may
            # name a pipeline it cannot assemble one operator at a time.
            result = solve_by_proposal(
                solver,
                task,
                n_samples=cfg.n_proposals,
                temperature=cfg.temperature,
                max_new_tokens=cfg.proposal_tokens,
                deadline=deadline,
            )
        if not result.solved:
            return set()
        wrote: set[int] = set()
        for test_idx in range(min(len(entry), len(result.attempts[0]))):
            slot = dict(entry[test_idx])
            for key, attempt in zip(("attempt_1", "attempt_2"), result.attempts):
                grid = attempt[test_idx] if test_idx < len(attempt) else None
                if grid is not None:
                    slot[key] = to_lists(grid)
                    wrote.add(test_idx)
            entry[test_idx] = slot
        return wrote
    except Exception:
        return set()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="ARC Prize runner: symbolic pass then LLM pass")
    p.add_argument("--challenges", default=None)
    p.add_argument("--solutions", default=None)
    p.add_argument("--output", default="submission.json")
    p.add_argument("--total-seconds", type=float, default=None, help="global budget for both phases")
    p.add_argument("--symbolic-fraction", type=float, default=0.25, help="share of the budget for phase 1")
    p.add_argument("--symbolic-per-task", type=float, default=20.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--stage", default="L")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model-path", default=None, help="local model dir; auto-discovered if omitted")
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--llm-per-task", type=float, default=90.0)
    p.add_argument("--ttt-steps", type=int, default=0, help="0 disables test-time training")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-llm", action="store_true", help="phase 1 only")
    p.add_argument(
        "--no-programs",
        action="store_true",
        help="skip the program-proposal pass and ask the model for grids directly",
    )
    p.add_argument(
        "--n-proposals",
        type=int,
        default=8,
        help="program samples per task; each is verified against the demonstrations",
    )
    p.add_argument("--proposal-tokens", type=int, default=256)
    p.add_argument(
        "--no-execution-guided",
        action="store_true",
        help="skip stepwise decoding and only propose whole programs blind",
    )
    p.add_argument("--beam-width", type=int, default=3)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    verbose = not args.quiet
    total = args.total_seconds
    if total is None:
        total = float(os.environ.get("ARC_TOTAL_SECONDS", 11 * 3600))
    started = time.perf_counter()

    try:
        challenges = find_challenges(args.challenges)
    except Exception:
        traceback.print_exc()
        Path(args.output).write_text("{}", encoding="utf-8")
        return 1

    symbolic_seconds = max(60.0, total * max(0.05, min(0.95, args.symbolic_fraction)))
    sym_argv = [
        "--challenges", str(challenges),
        "--output", args.output,
        "--total-seconds", str(symbolic_seconds),
        "--per-task-seconds", str(args.symbolic_per_task),
        "--workers", str(args.workers or (os.cpu_count() or 4)),
        "--stage", args.stage,
    ]
    if args.solutions:
        sym_argv += ["--solutions", args.solutions]
    if args.limit is not None:
        sym_argv += ["--limit", str(args.limit)]
    if args.quiet:
        sym_argv.append("--quiet")

    sym_report = run_symbolic(build_config(sym_argv))
    if verbose:
        print(json.dumps({"phase_1_symbolic": sym_report}, indent=2), flush=True)

    output = Path(args.output)
    submission = json.loads(output.read_text(encoding="utf-8"))
    raw_payload = json.loads(challenges.read_text(encoding="utf-8"))

    llm_report: dict[str, Any] = {"skipped": "disabled"}
    if not args.no_llm:
        model_path = args.model_path or find_model_dir()
        if not model_path:
            llm_report = {"skipped": "no_model_found"}
            if verbose:
                print(
                    "[llm] no model directory found. Pass --model-path, or attach the "
                    "weights as a Kaggle Dataset so a config.json sits beside the "
                    "safetensors under /kaggle/input/.",
                    flush=True,
                )
        else:
            targets = _unsolved_ids(sym_report, submission)
            if args.limit is not None:
                targets = targets[: args.limit]
            llm_seconds = max(0.0, total - (time.perf_counter() - started) - 60.0)
            if verbose:
                print(
                    f"[llm] model={model_path} targets={len(targets)} "
                    f"budget={llm_seconds:.0f}s ttt_steps={args.ttt_steps}",
                    flush=True,
                )
            llm_report = run_llm_phase(
                submission,
                raw_payload,
                targets,
                LlmPhaseConfig(
                    model_path=model_path,
                    adapter_path=args.adapter_path,
                    seconds=llm_seconds,
                    per_task_seconds=args.llm_per_task,
                    ttt_steps=args.ttt_steps,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                    propose_programs=not args.no_programs,
                    n_proposals=args.n_proposals,
                    proposal_tokens=args.proposal_tokens,
                    execution_guided=not args.no_execution_guided,
                    beam_width=args.beam_width,
                    max_depth=args.max_depth,
                ),
                output,
                verbose=verbose,
            )

    expected = sorted(raw_payload)
    if args.limit is not None:
        expected = expected[: args.limit]
    problems = validate_submission(submission, expected)
    if problems:
        for task_id in expected:
            entry = submission.get(task_id)
            if not isinstance(entry, list) or not entry:
                submission[task_id] = blank_entry(1)
        problems = validate_submission(submission, expected)
    _write(output, submission)

    report: dict[str, Any] = {
        "phase_1_symbolic": sym_report,
        "phase_2_llm": llm_report,
        "wall_clock_sec": round(time.perf_counter() - started, 2),
        "schema_problems": problems,
    }
    if args.solutions and Path(args.solutions).is_file():
        report["final_score"] = score_submission(submission, Path(args.solutions))
    print(json.dumps(report, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

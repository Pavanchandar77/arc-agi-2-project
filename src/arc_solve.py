"""One task in, two attempts out. Never raises, never returns a malformed grid.

This is the single seam every runner goes through, so the submission schema is
guaranteed in exactly one place. The layers, cheapest first:

  1. exact train-verified solver bank (src.hrps.solvers)
  2. budgeted finite-DSL search   (src.hrps.search)
  3. structural fallbacks         (identity, train-output shape priors)

Anything that predicts must first reproduce every demonstration exactly, except
the fallbacks, which exist only so that a submission is always well formed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.hrps.grid import Grid, as_grid, is_valid_grid, majority_color, shape, to_lists
from src.hrps.solvers import fit_rules
from src.hrps.task import ArcTask

GridList = list[list[int]]


@dataclass
class SolveOutcome:
    task_id: str
    attempts: list[list[GridList]]  # [attempt_1_per_test, attempt_2_per_test]
    source: str
    rules: list[str] = field(default_factory=list)
    program: str = ""
    verified: bool = False
    seconds: float = 0.0
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "rules": self.rules[:8],
            "program": self.program,
            "verified": self.verified,
            "seconds": round(self.seconds, 4),
            "error": self.error,
        }


def _fallback_grid(task: ArcTask, test_input: Grid) -> Grid:
    """A well-formed guess for when nothing verified. Prefers a constant output
    shape seen in training, else echoes the input."""
    outs = task.train_outputs()
    if outs:
        oshapes = {shape(o) for o in outs}
        if len(oshapes) == 1 and len({o for o in outs}) == 1:
            return outs[0]
        if len(oshapes) == 1:
            h, w = next(iter(oshapes))
            fill = majority_color(outs[0])
            return tuple((fill,) * w for _ in range(h))
    return test_input


def _safe(grid: object, fallback: Grid) -> GridList:
    if is_valid_grid(grid):
        return to_lists(as_grid(grid))  # type: ignore[arg-type]
    return to_lists(fallback)


def solve_task(
    task: ArcTask,
    *,
    seconds: float = 30.0,
    use_search: bool = True,
    search_stage: str = "L",
    vote_frames: int = 0,
) -> SolveOutcome:
    """Solve one task within `seconds`. Total: any exception becomes a fallback.

    `vote_frames > 1` solves the task under that many D8/colour frames and votes
    on the back-transformed predictions. Frame 0 is the identity, so voting is a
    superset of the plain bank pass and replaces it rather than adding to it.
    """
    started = time.perf_counter()
    deadline = started + max(0.5, seconds)
    test_inputs = task.test_inputs()
    fallbacks = [_fallback_grid(task, g) for g in test_inputs]
    per_test: list[list[Grid]] = [[] for _ in test_inputs]
    rules_used: list[str] = []
    program = ""
    source = "fallback"
    verified = False
    error = ""

    # Layer 1: exact solver bank. Cheap, so it always runs first.
    bank_deadline = min(deadline, started + max(1.0, seconds * 0.5))
    if vote_frames > 1:
        try:
            from src.hrps.voting import solve_with_voting

            (a1, a2), report = solve_with_voting(
                task, n_frames=vote_frames, deadline=bank_deadline
            )
            for i in range(len(test_inputs)):
                for grid in (a1[i], a2[i]):
                    if grid is not None and is_valid_grid(grid) and grid not in per_test[i]:
                        per_test[i].append(grid)
            if any(per_test):
                source = "voting"
                verified = True
                rules_used.append(
                    f"frames={report.n_frames_with_prediction}/{report.n_frames_run}"
                    f" agreement={report.agreement:.2f}"
                )
        except Exception as exc:  # pragma: no cover - defensive
            error = f"voting:{type(exc).__name__}"
    else:
        try:
            for rule in fit_rules(task, deadline=bank_deadline):
                for i, inp in enumerate(test_inputs):
                    if len(per_test[i]) >= 2:
                        continue
                    try:
                        pred = rule.predict(inp)
                    except Exception:
                        continue
                    if pred is None or not is_valid_grid(pred):
                        continue
                    if pred not in per_test[i]:
                        per_test[i].append(pred)
                        if rule.name not in rules_used:
                            rules_used.append(rule.name)
                if all(len(p) >= 2 for p in per_test):
                    break
            if any(per_test):
                source = "solver_bank"
                verified = True
        except Exception as exc:  # pragma: no cover - defensive
            error = f"bank:{type(exc).__name__}"

    # Layer 2: DSL search, only for the slots the bank left empty.
    if use_search and not all(len(p) >= 2 for p in per_test):
        remaining = deadline - time.perf_counter()
        if remaining > 0.25:
            try:
                from src.hrps.search import SearchBudget, search_task

                res = search_task(
                    task,
                    stage=search_stage,
                    budget=SearchBudget(
                        max_depth=3,
                        max_nodes=100_000,
                        max_seconds=remaining,
                        # Frontier nodes hold materialised prediction grids, so
                        # this cap is a memory bound, not just a search bound.
                        max_frontier=10_000,
                        max_ops_per_node=60,
                    ),
                )
                if res.telemetry.get("joint_verified"):
                    program = res.programs[0] if res.programs else ""
                    verified = True
                    if source == "fallback":
                        source = "dsl_search"
                    else:
                        source = f"{source}+dsl_search"
                    for attempt in res.attempts[:2]:
                        for i, grid_l in enumerate(attempt):
                            if i >= len(per_test) or len(per_test[i]) >= 2:
                                continue
                            if not is_valid_grid(grid_l):
                                continue
                            g = as_grid(grid_l)
                            if g not in per_test[i]:
                                per_test[i].append(g)
            except Exception as exc:  # pragma: no cover - defensive
                error = (error + " " if error else "") + f"search:{type(exc).__name__}"

    a1 = [_safe(p[0] if p else None, fb) for p, fb in zip(per_test, fallbacks)]
    a2 = [
        _safe(p[1] if len(p) > 1 else (p[0] if p else None), fb)
        for p, fb in zip(per_test, fallbacks)
    ]
    return SolveOutcome(
        task_id=task.task_id,
        attempts=[a1, a2],
        source=source,
        rules=rules_used,
        program=program,
        verified=verified,
        seconds=time.perf_counter() - started,
        error=error,
    )


def submission_entry(outcome: SolveOutcome) -> list[dict[str, GridList]]:
    """Kaggle's per-task shape: one object per test input, both attempts present."""
    a1, a2 = outcome.attempts
    return [{"attempt_1": a1[i], "attempt_2": a2[i]} for i in range(len(a1))]


def blank_entry(n_test: int) -> list[dict[str, GridList]]:
    """A schema-valid placeholder used when a task could not be solved at all."""
    return [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(max(1, n_test))]

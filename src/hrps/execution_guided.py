"""Write one operator, run it, look at what happened, write the next.

Blind generation asks the model for a whole program and only then finds out
whether it was nonsense. Every wrong guess costs a full generation, the model
never learns within the attempt, and an operator that was invalid at step one
is discovered after eight more were written on top of it.

Execution-guided decoding closes the loop instead. After each operator the
executor runs it on every demonstration, and the resulting grids go back into
the prompt. The model writes the second operator while looking at what the
first actually did - not at what it hoped it did.

    original input   ->  [op1]  ->  current state  ->  [op2]  ->  ...
                                         |
                              shown to the model, next to the target

Three things follow, and they are the point:

* **A dead branch dies at depth one.** An operator whose execution fails, or
  whose result cannot reach the target shape, is discarded before anything is
  built on it.
* **Success is detected, not hoped for.** The moment the current state equals
  every demonstration target, the program is verified and the search stops. No
  separate checking pass, no wasted remaining depth.
* **The search is over states, not strings.** Two different operator sequences
  reaching the same grids are the same node, so the beam does not fill up with
  spellings of one idea.

The test input rides through the identical operators, so a solved search hands
back the answer as a side effect. Its output is never read - only the
demonstrations decide, exactly as everywhere else in this system.

Kind: exact execution and verification, heuristic beam ordering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.hrps.dsl import Op, Program, execute_op
from src.hrps.grid import Grid, is_valid_grid, shape
from src.hrps.proposal import parse_program, verify_program
from src.hrps.task import ArcTask


def _grid_lines(grid: Grid) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in grid)


@dataclass(frozen=True)
class State:
    """A partial program and what it has done to every grid so far."""

    ops: tuple[Op, ...]
    demo_grids: tuple[Grid, ...]          # current state of each demonstration input
    test_grids: tuple[Optional[Grid], ...]  # same operators applied to the test inputs

    @property
    def program(self) -> Program:
        return Program(self.ops)

    @property
    def depth(self) -> int:
        return len(self.ops)

    def key(self) -> tuple:
        """States are identified by what the grids ARE, not how they got there."""
        return self.demo_grids


def initial_state(task: ArcTask) -> State:
    demos = tuple(p.input for p in task.train if p.output is not None)
    return State(ops=(), demo_grids=demos, test_grids=tuple(task.test_inputs()))


def targets(task: ArcTask) -> tuple[Grid, ...]:
    return tuple(p.output for p in task.train if p.output is not None)  # type: ignore[misc]


def is_solved(state: State, task: ArcTask) -> bool:
    """Every demonstration reproduced exactly. This is the whole acceptance test."""
    goal = targets(task)
    return bool(goal) and len(state.demo_grids) == len(goal) and state.demo_grids == goal


def apply_op(state: State, op: Op) -> Optional[State]:
    """Run one operator on every grid. None if it fails on any demonstration.

    A failure on a *test* input is survivable - that input simply has no answer
    from this branch - but a failure on a demonstration kills the branch, since
    a program that cannot run on the examples can never be verified by them.
    """
    demos: list[Grid] = []
    for grid in state.demo_grids:
        out = execute_op(op, grid)
        if out is None or not is_valid_grid(out):
            return None
        demos.append(out)
    tests: list[Optional[Grid]] = []
    for grid in state.test_grids:
        if grid is None:
            tests.append(None)
            continue
        out = execute_op(op, grid)
        tests.append(out if out is not None and is_valid_grid(out) else None)
    return State(ops=state.ops + (op,), demo_grids=tuple(demos), test_grids=tuple(tests))


def _shape_distance(state: State, task: ArcTask) -> int:
    """How far the current shapes are from the targets. Zero is necessary, not
    sufficient - but a branch whose shapes are diverging is worth ranking below
    one whose shapes already line up."""
    total = 0
    for got, want in zip(state.demo_grids, targets(task)):
        gh, gw = shape(got)
        wh, ww = shape(want)
        total += abs(gh - wh) + abs(gw - ww)
    return total


def _cell_distance(state: State, task: ArcTask) -> int:
    """Mismatched cells where the shapes already agree; a large number where
    they do not, so same-shape branches always sort first."""
    total = 0
    for got, want in zip(state.demo_grids, targets(task)):
        if shape(got) != shape(want):
            total += 10_000
            continue
        total += sum(
            1 for r, row in enumerate(got) for c, v in enumerate(row) if v != want[r][c]
        )
    return total


def build_step_prompt(task: ArcTask, state: State) -> str:
    """Show the model what its program has done so far, beside the target."""
    parts: list[str] = []
    if state.ops:
        parts.append(f"Program so far: {state.program.serialize()}\n")
    else:
        parts.append("No operators applied yet.\n")
    goal = targets(task)
    for i, (current, want) in enumerate(zip(state.demo_grids, goal), 1):
        parts.append(
            f"Demonstration {i}\n"
            f"current state:\n{_grid_lines(current)}\n"
            f"target output:\n{_grid_lines(want)}\n"
        )
    parts.append(
        "Give the single next operator to apply, on one line, with no prose. "
        "Offer several alternatives on separate lines, best first."
    )
    return "\n".join(parts)


STEP_SYSTEM_PROMPT = (
    "You are transforming grids one operator at a time using a fixed DSL.\n\n"
    "You are shown the CURRENT state of each demonstration after the operators "
    "applied so far, alongside the target output. Your job is to name the next "
    "single operator that moves the current state closer to the target.\n\n"
    "Reply with one operator per line, best first, and nothing else. No prose, "
    "no grids, no explanation."
)


@dataclass
class ExecReport:
    n_model_calls: int = 0
    n_ops_proposed: int = 0
    n_ops_rejected: int = 0
    n_states_expanded: int = 0
    max_depth_reached: int = 0
    solved: bool = False
    program: str = ""
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n_model_calls": self.n_model_calls,
            "n_ops_proposed": self.n_ops_proposed,
            "n_ops_rejected": self.n_ops_rejected,
            "n_states_expanded": self.n_states_expanded,
            "max_depth_reached": self.max_depth_reached,
            "solved": self.solved,
            "program": self.program,
            "seconds": round(self.seconds, 2),
        }


@dataclass
class ExecResult:
    state: Optional[State]
    report: ExecReport
    candidates: list[str] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return self.report.solved

    def answers(self) -> list[Optional[Grid]]:
        return list(self.state.test_grids) if self.state is not None else []


def parse_ops(text: str) -> list[Op]:
    """Read single operators out of the model's reply, dropping anything else."""
    if not text:
        return []
    ops: list[Op] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip().strip("`").strip()
        if not line or line.startswith(("#", "//")):
            continue
        # Reuse the proposal parser so an operator accepted here is one the
        # executor will certainly accept: same names, same arity, same ranges.
        program = parse_program(line)
        if program is None or len(program.ops) != 1:
            continue
        key = program.ops[0].serialize()
        if key not in seen:
            seen.add(key)
            ops.append(program.ops[0])
    return ops


def search_execution_guided(
    solver,
    task: ArcTask,
    *,
    beam_width: int = 3,
    max_depth: int = 4,
    temperature: float = 0.7,
    max_new_tokens: int = 96,
    deadline: Optional[float] = None,
) -> ExecResult:
    """Grow programs one executed operator at a time, keeping the best `beam_width`.

    Returns as soon as a state reproduces every demonstration exactly.
    """
    started = time.perf_counter()
    report = ExecReport()
    root = initial_state(task)
    if not targets(task):
        report.seconds = time.perf_counter() - started
        return ExecResult(None, report)
    if is_solved(root, task):  # the identity program already explains it
        report.solved = True
        report.program = "identity"
        report.seconds = time.perf_counter() - started
        return ExecResult(root, report)

    beam: list[State] = [root]
    visited: set[tuple] = {root.key()}

    for depth in range(max_depth):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        report.max_depth_reached = depth + 1
        children: list[State] = []
        for state in beam:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            messages = [
                {"role": "system", "content": STEP_SYSTEM_PROMPT},
                {"role": "user", "content": build_step_prompt(task, state)},
            ]
            text = solver.complete(
                messages,
                temperature=temperature if depth or state.ops else 0.0,
                attempt=depth,
                max_new_tokens=max_new_tokens,
            )
            report.n_model_calls += 1
            ops = parse_ops(text or "")
            report.n_ops_proposed += len(ops)
            for op in ops:
                child = apply_op(state, op)
                if child is None:
                    report.n_ops_rejected += 1
                    continue
                if child.key() in visited:
                    continue
                visited.add(child.key())
                report.n_states_expanded += 1
                if is_solved(child, task):
                    # Belt and braces: the branch says it matches, so make the
                    # standalone verifier agree before anyone believes it.
                    if verify_program(child.program, task).is_total:
                        report.solved = True
                        report.program = child.program.serialize()
                        report.seconds = time.perf_counter() - started
                        return ExecResult(child, report)
                children.append(child)
        if not children:
            break
        children.sort(key=lambda s: (_cell_distance(s, task), _shape_distance(s, task), s.depth))
        beam = children[:beam_width]

    report.seconds = time.perf_counter() - started
    return ExecResult(None, report, candidates=[s.program.serialize() for s in beam])

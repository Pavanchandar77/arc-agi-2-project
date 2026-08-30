"""The prompt that asks for a program, and the target it is trained against.

Asking a model for a grid gives you an answer you cannot check. Asking it for a
program gives you a hypothesis the demonstrations can settle. The prompt is
built so that a proposal is cheap to emit and cheap to reject: one program per
line, no prose, the operator catalog inline so the model is never guessing at
names it half-remembers.

The catalog is generated from the DSL itself rather than written out here, so a
new operator becomes proposable the moment it is registered, and a removed one
stops being advertised. A prompt that names operators the executor does not have
would train the model to hallucinate.
"""

from __future__ import annotations

from typing import Optional

from src.hrps.dsl import OP_DEFS
from src.hrps.grid import Grid
from src.hrps.proposal import MAX_OPS, UNPROPOSABLE
from src.hrps.task import ArcTask

SYSTEM_PROMPT = (
    "You solve ARC puzzles by writing a short program in a fixed DSL, never by "
    "writing the answer grid directly.\n\n"
    "A program is a pipeline of operators separated by '|', applied left to "
    "right to the input grid. Your program must reproduce EVERY demonstration "
    "output exactly; programs that miss even one cell are discarded.\n\n"
    "Reply with one program per line and nothing else. No prose, no "
    "explanation, no grids. Order your lines best guess first. Offer several "
    "genuinely different hypotheses rather than variations of one."
)


def _grid_to_lines(grid: Grid) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in grid)


def operator_reference() -> str:
    """One line per proposable operator: name, arguments, what it needs."""
    lines: list[str] = []
    for name in sorted(OP_DEFS):
        if name in UNPROPOSABLE:
            continue
        defn = OP_DEFS[name]
        arg_types = [t.value for t in defn.in_types[1:]]
        signature = f"{name}:{','.join(arg_types)}" if arg_types else name
        lines.append(f"  {signature:<52} {defn.preconditions}")
    return "\n".join(lines)


ARG_LEGEND = (
    "Argument types: Color/Bg are 0-9 (Bg is the background colour, usually 0); "
    "Connectivity is 4 or 8; Bool is t or f; Int is a small integer; "
    "ColorMap is written src-dst;src-dst (e.g. apply_colormap:1-2;3-4).\n"
    "'identity' is a valid program meaning 'output equals input'."
)


def build_prompt(task: ArcTask, *, include_catalog: bool = True) -> str:
    """The user turn: demonstrations, the test input, and the operator catalog."""
    parts: list[str] = []
    if include_catalog:
        parts.append(f"Operators available:\n{operator_reference()}\n\n{ARG_LEGEND}\n")
    parts.append("Demonstrations:")
    for i, pair in enumerate(task.train, 1):
        if pair.output is None:
            continue
        parts.append(
            f"\nExample {i} input:\n{_grid_to_lines(pair.input)}"
            f"\nExample {i} output:\n{_grid_to_lines(pair.output)}"
        )
    test_inputs = task.test_inputs()
    if test_inputs:
        parts.append(f"\nTest input:\n{_grid_to_lines(test_inputs[0])}")
    parts.append(
        f"\nWrite up to {MAX_OPS} operators per program, one program per line."
    )
    return "\n".join(parts)


def build_training_example(task: ArcTask, program: str) -> dict:
    """One supervised example: the task, and a program known to explain it.

    The completion is a program the verifier has already certified against
    every demonstration, so the label is correct by construction rather than by
    anyone's judgement.
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(task)},
            {"role": "assistant", "content": program},
        ],
        "task_id": task.task_id,
        "program": program,
    }


def build_inference_messages(task: ArcTask) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(task)},
    ]

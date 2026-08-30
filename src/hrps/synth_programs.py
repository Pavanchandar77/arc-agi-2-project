"""Unlimited verified supervision, by running the DSL backwards.

Harvesting real tasks hit a hard ceiling: search explains 2.6% of ARC-AGI-2
training, which is 26 tasks and 31 examples. That is not enough to teach a
model a language, and more search time does not help, because the limit is what
the DSL can express rather than how long it is given to look.

Generation has no such ceiling. Sample a program, sample inputs, apply it, and
the result is a task whose correct program is known by construction - the
labelling problem disappears because the label came first. The distribution is
not ARC's, and that is understood: this teaches the language and the shape of
composition, and the harvested real tasks anchor it to the real thing.

Two filters do the real work, because a generated pair is only supervision if
it is unambiguous:

* **A task must be informative.** If every output equals its input, the label
  might as well be 'identity'; if the program collapses everything to a
  constant, nothing about the transformation is recoverable from the examples.
* **A task must be verified.** Every emitted pair is replayed through the same
  gate that judges the model at inference. A generator that trusted itself
  would quietly emit mislabelled data, which is worse than none.

Kind: exact generation, heuristic sampling.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from src.hrps.dsl import OP_DEFS, DslType, Op, Program, replay
from src.hrps.grid import Grid, is_valid_grid
from src.hrps.proposal import UNPROPOSABLE, verify_program
from src.hrps.task import ArcTask, Pair

PROPOSABLE = tuple(sorted(n for n in OP_DEFS if n not in UNPROPOSABLE))


def _sample_arg(rng: random.Random, dtype: DslType, palette: tuple[int, ...]) -> object:
    if dtype is DslType.COLOR:
        return rng.choice(palette) if palette else rng.randint(1, 9)
    if dtype is DslType.BG:
        # 0 is the usual background, but not always; a model that only ever
        # sees bg=0 will not generalise to tasks where it is not.
        return 0 if rng.random() < 0.7 else rng.randint(0, 9)
    if dtype is DslType.CONNECTIVITY:
        return rng.choice((4, 8))
    if dtype is DslType.BOOL:
        return rng.random() < 0.5
    if dtype is DslType.INT:
        return rng.choice((1, 2, 3))
    if dtype is DslType.COLORMAP:
        colors = list(palette) if palette else list(range(1, 6))
        rng.shuffle(colors)
        rotated = colors[1:] + colors[:1]
        return tuple((a, b) for a, b in zip(colors, rotated) if a != b) or ((1, 2),)
    return 0


def sample_op(rng: random.Random, palette: tuple[int, ...]) -> Op:
    name = rng.choice(PROPOSABLE)
    defn = OP_DEFS[name]
    args = tuple(_sample_arg(rng, t, palette) for t in defn.in_types[1:])
    return Op(name, args)


def sample_program(rng: random.Random, palette: tuple[int, ...], *, max_depth: int = 3) -> Program:
    depth = rng.choices((1, 2, 3, 4), weights=(35, 35, 20, 10))[0]
    depth = min(depth, max_depth)
    return Program(tuple(sample_op(rng, palette) for _ in range(depth)))


def random_grid(rng: random.Random, palette: tuple[int, ...]) -> Grid:
    """A grid with some structure: a background plus scattered blobs."""
    h = rng.randint(3, 14)
    w = rng.randint(3, 14)
    bg = 0
    cells = [[bg] * w for _ in range(h)]
    for _ in range(rng.randint(1, 5)):
        colour = rng.choice(palette)
        bh, bw = rng.randint(1, max(1, h // 2)), rng.randint(1, max(1, w // 2))
        r0, c0 = rng.randint(0, h - bh), rng.randint(0, w - bw)
        for r in range(r0, r0 + bh):
            for c in range(c0, c0 + bw):
                if rng.random() < 0.85:
                    cells[r][c] = colour
    return tuple(tuple(row) for row in cells)


def _informative(inputs: list[Grid], outputs: list[Grid]) -> bool:
    """Reject pairs from which no transformation could be inferred."""
    if any(o is None or not is_valid_grid(o) for o in outputs):
        return False
    if all(i == o for i, o in zip(inputs, outputs)):
        return False  # indistinguishable from identity
    if len({o for o in outputs}) == 1 and len(outputs) > 1:
        return False  # collapses to a constant, telling nothing about the rule
    if any(len(o) * len(o[0]) <= 1 for o in outputs):
        return False  # a single cell carries almost no signal
    return True


@dataclass
class SynthReport:
    attempted: int = 0
    rejected_execution: int = 0
    rejected_uninformative: int = 0
    rejected_unverified: int = 0
    produced: int = 0

    def as_dict(self) -> dict:
        return {
            "attempted": self.attempted,
            "rejected_execution": self.rejected_execution,
            "rejected_uninformative": self.rejected_uninformative,
            "rejected_unverified": self.rejected_unverified,
            "produced": self.produced,
            "yield": round(self.produced / max(1, self.attempted), 4),
        }


def synthesize(
    n: int,
    *,
    seed: int = 0,
    n_demos: int = 3,
    max_attempts_per: int = 40,
    max_depth: int = 3,
) -> tuple[list[tuple[ArcTask, Program]], SynthReport]:
    """`n` verified (task, program) pairs, or as many as the budget allows."""
    rng = random.Random(seed)
    out: list[tuple[ArcTask, Program]] = []
    report = SynthReport()
    budget = n * max_attempts_per

    while len(out) < n and report.attempted < budget:
        report.attempted += 1
        palette = tuple(rng.sample(range(1, 10), rng.randint(2, 5)))
        program = sample_program(rng, palette, max_depth=max_depth)
        grids = [random_grid(rng, palette) for _ in range(n_demos + 1)]
        try:
            outputs = [replay(program, g) for g in grids]
        except Exception:
            report.rejected_execution += 1
            continue
        if any(o is None for o in outputs):
            report.rejected_execution += 1
            continue
        if not _informative(grids, outputs):  # type: ignore[arg-type]
            report.rejected_uninformative += 1
            continue

        train = tuple(Pair(input=g, output=o) for g, o in zip(grids[:-1], outputs[:-1]))
        test = (Pair(input=grids[-1], output=outputs[-1]),)
        task = ArcTask(
            task_id=f"synth_{len(out):06d}", train=train, test=test, split="synthetic"
        )
        # The generator does not get to vouch for itself.
        if not verify_program(program, task).is_total:
            report.rejected_unverified += 1
            continue
        out.append((task, program))
        report.produced += 1
    return out, report

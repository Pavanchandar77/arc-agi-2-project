"""Verifier-closed synthetic ARC tasks from Bond-L1 programs.

A program is sampled from the closed language, applied to random grids,
and kept only if every pair is valid and not a no-op. Hold-out official
IDs are never used. Public evaluation is never read.

Kind: exact (program generates labels).
"""

from __future__ import annotations

import random
from typing import Optional

from src.hrps.dsl import Op, Program, replay
from src.hrps.grid import Grid, as_grid, shape
from src.hrps.task import ArcTask, Pair

BOND_UNARY = (
    Op("rot90", ()),
    Op("rot180", ()),
    Op("rot270", ()),
    Op("flip_h", ()),
    Op("flip_v", ()),
    Op("transpose", ()),
    Op("anti_transpose", ()),
)

BOND_PARAM = (
    lambda rng: Op("recolor_nonsingleton_to_singleton_color", (4, False, 0)),
    lambda rng: Op("recolor_largest_to_smallest_color", (4, False, 0)),
    lambda rng: Op("recolor_smallest_to_largest_color", (4, False, 0)),
    lambda rng: Op("erase_smallest", (4, False, 0)),
    lambda rng: Op("recolor_all_fg_to_smallest_color", (4, False, 0)),
    lambda rng: Op("keep_least_frequent_color", (0,)),
    lambda rng: Op("keep_most_frequent_nonbg", (0,)),
    lambda rng: Op("recolor_least_frequent_to_most_frequent", (0,)),
    lambda rng: Op("translate_fg", (rng.randrange(4), 0)),
    lambda rng: Op("center_fg", (0,)),
    lambda rng: Op("fill_holes", (0,)),
    lambda rng: Op("outline", (0,)),
    lambda rng: Op("upscale", (2,)),
)


def _random_grid(rng: random.Random, *, h: int, w: int, palette: tuple[int, ...], density: float) -> Grid:
    rows = []
    for _ in range(h):
        row = []
        for _ in range(w):
            row.append(rng.choice(palette) if rng.random() < density else 0)
        rows.append(tuple(row))
    g = as_grid(rows)
    if all(v == 0 for row in g for v in row):
        rr, cc = rng.randrange(h), rng.randrange(w)
        rows[rr] = tuple(palette[0] if c == cc else rows[rr][c] for c in range(w))
        g = as_grid(rows)
    return g


def _marker_blob_grid(rng: random.Random, *, n: int = 7) -> Grid:
    """Structural singleton + blob. Not a named ARC task."""
    g = [[0] * n for _ in range(n)]
    blob = rng.choice([2, 3, 4, 6, 8])
    marker = rng.choice([c for c in range(1, 10) if c != blob])
    r0, c0 = rng.randint(1, n - 4), rng.randint(1, n - 4)
    bh, bw = rng.randint(2, 3), rng.randint(2, 3)
    for r in range(r0, r0 + bh):
        for c in range(c0, c0 + bw):
            g[r][c] = blob
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    mr, mc = rng.choice(corners)
    g[mr][mc] = marker
    return as_grid(g)


def sample_program(rng: random.Random, *, depth: int = 1) -> Program:
    ops: list[Op] = []
    for _ in range(depth):
        if rng.random() < 0.45:
            ops.append(rng.choice(BOND_UNARY))
        else:
            ops.append(rng.choice(BOND_PARAM)(rng))
    return Program(tuple(ops))


def _apply_task(program: Program, inputs: list[Grid]) -> Optional[list[Grid]]:
    outs: list[Grid] = []
    for g in inputs:
        pred = replay(program, g)
        if pred is None or pred == g:
            return None
        outs.append(pred)
    if len({shape(o) for o in outs}) > 1 and program.names()[0] in {"upscale", "tile"}:
        pass
    return outs


def synthesize_task(rng: random.Random, *, kind: str = "auto") -> Optional[tuple[ArcTask, Program]]:
    if kind == "auto":
        kind = rng.choice(["geom", "role", "count", "marker"])
    if kind == "geom":
        program = Program((rng.choice(BOND_UNARY),))
        h = rng.choice([3, 4, 5])
        inputs = [_random_grid(rng, h=h, w=h, palette=(1, 2, 3, 4), density=0.6) for _ in range(3)]
    elif kind == "marker":
        program = Program(
            (
                Op("recolor_nonsingleton_to_singleton_color", (4, False, 0)),
                Op("erase_smallest", (4, False, 0)),
            )
        )
        inputs = [_marker_blob_grid(rng) for _ in range(3)]
    elif kind == "count":
        program = Program((Op("keep_most_frequent_nonbg", (0,)),))
        inputs = [_random_grid(rng, h=5, w=5, palette=(1, 2, 3), density=0.7) for _ in range(3)]
    else:
        program = sample_program(rng, depth=1)
        inputs = [_random_grid(rng, h=6, w=6, palette=(1, 2, 3, 4, 5), density=0.45) for _ in range(3)]
    outs = _apply_task(program, inputs)
    if outs is None:
        return None
    pairs_train = [Pair(inputs[i], outs[i]) for i in range(2)]
    pairs_test = [Pair(inputs[2], outs[2])]
    digest = abs(hash((program.serialize(), tuple(inputs[0][0])))) % 10_000_000
    task = ArcTask(
        task_id=f"synth_{kind}_{digest:07d}",
        split="training",
        train=tuple(pairs_train),
        test=tuple(pairs_test),
    )
    return task, program


def synthesize_batch(n: int, *, seed: int = 0, kinds: Optional[tuple[str, ...]] = None) -> list[tuple[ArcTask, Program]]:
    rng = random.Random(seed)
    kinds = kinds or ("geom", "role", "count", "marker")
    out: list[tuple[ArcTask, Program]] = []
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        kind = kinds[len(out) % len(kinds)]
        rec = synthesize_task(rng, kind=kind)
        if rec is None:
            continue
        task, program = rec
        if any(t.task_id == task.task_id for t, _ in out):
            continue
        out.append((task, program))
    return out

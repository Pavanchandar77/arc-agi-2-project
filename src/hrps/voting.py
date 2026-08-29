"""Augmented inference with voting.

A solver rule fires on the frame it happens to be written for. A rule keyed on
"background is colour 0" misses a task whose background is 3; a rule keyed on
"read the columns" misses its own transpose. Rather than write every rule eight
ways, solve the task eight ways.

For each of a set of task-level transformations T (a D8 symmetry composed with
a colour bijection), solve T(task), map the prediction back through T-inverse,
and tally votes across frames. The same underlying rule then gets many chances
to fire, and agreement across independent frames is evidence the answer is
right rather than an artefact of one framing.

Two properties matter and are tested:

* T must be a true task transformation - applied to every demonstration input,
  every demonstration output, and every test input alike, so the augmented task
  poses the same problem in a different frame.
* T-inverse must exactly undo T, or every vote is cast in the wrong frame.

Kind: exact transformations, heuristic vote weighting.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from src.hrps.grid import (
    Grid,
    anti_transpose,
    flip_h,
    flip_v,
    is_valid_grid,
    rot90,
    rot180,
    rot270,
    transpose,
)
from src.hrps.solvers import fit_rules
from src.hrps.task import ArcTask, Pair

# Name -> (forward, inverse name). Reflections are their own inverse; the
# quarter turns pair up. test_voting.py checks every one of these round-trips
# on a non-square grid, where a wrong pairing cannot hide.
D8_FORWARD = {
    "identity": lambda g: g,
    "rot90": rot90,
    "rot180": rot180,
    "rot270": rot270,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
    "anti_transpose": anti_transpose,
}
D8_INVERSE_NAME = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_h": "flip_h",
    "flip_v": "flip_v",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}
D8_NAMES = tuple(D8_FORWARD)

ColorPerm = tuple[int, ...]  # index = source colour, value = destination
IDENTITY_PERM: ColorPerm = tuple(range(10))


@dataclass(frozen=True)
class Frame:
    """One task-level transformation, and how to undo it."""

    d8: str
    perm: ColorPerm

    def apply(self, grid: Grid) -> Grid:
        out = D8_FORWARD[self.d8](grid)
        if self.perm != IDENTITY_PERM:
            out = tuple(tuple(self.perm[v] for v in row) for row in out)
        return out

    def invert(self, grid: Grid) -> Grid:
        # Colour and position commute, so either order undoes the pair.
        if self.perm != IDENTITY_PERM:
            back = invert_perm(self.perm)
            grid = tuple(tuple(back[v] for v in row) for row in grid)
        return D8_FORWARD[D8_INVERSE_NAME[self.d8]](grid)

    @property
    def is_identity(self) -> bool:
        return self.d8 == "identity" and self.perm == IDENTITY_PERM


def invert_perm(perm: ColorPerm) -> ColorPerm:
    back = [0] * 10
    for src, dst in enumerate(perm):
        back[dst] = src
    return tuple(back)


def _random_perm(rng: random.Random, *, fix_zero: bool) -> ColorPerm:
    if fix_zero:
        rest = list(range(1, 10))
        rng.shuffle(rest)
        return (0, *rest)
    everything = list(range(10))
    rng.shuffle(everything)
    return tuple(everything)


def build_frames(n: int, *, seed: int = 0) -> list[Frame]:
    """`n` distinct frames, identity first, deterministic for a given seed.

    The eight D8 symmetries come first because they are free and exhaustive;
    colour permutations only start once those run out. Permutations that fix
    colour 0 come before ones that move it, since most rules treat 0 as
    background and moving it is the more violent change.
    """
    if n < 1:
        return [Frame("identity", IDENTITY_PERM)]
    rng = random.Random(seed)
    frames: list[Frame] = []
    seen: set[tuple[str, ColorPerm]] = set()

    def add(frame: Frame) -> bool:
        key = (frame.d8, frame.perm)
        if key in seen:
            return False
        seen.add(key)
        frames.append(frame)
        return len(frames) >= n

    for name in D8_NAMES:
        if add(Frame(name, IDENTITY_PERM)):
            return frames
    for round_idx in range(64):
        perm = _random_perm(rng, fix_zero=round_idx < 32)
        if perm == IDENTITY_PERM:
            continue
        for name in D8_NAMES:
            if add(Frame(name, perm)):
                return frames
    return frames


def transform_task(task: ArcTask, frame: Frame) -> ArcTask:
    """Apply the frame to every grid in the task, demonstrations and tests."""
    train = tuple(
        Pair(
            input=frame.apply(p.input),
            output=frame.apply(p.output) if p.output is not None else None,
        )
        for p in task.train
    )
    test = tuple(
        Pair(
            input=frame.apply(p.input),
            output=frame.apply(p.output) if p.output is not None else None,
        )
        for p in task.test
    )
    return ArcTask(task_id=task.task_id, train=train, test=test, split=task.split)


# Weight of the k-th distinct prediction a frame produces. The best rule in a
# frame should count for more than that frame's third-choice rule, but a
# third choice agreed on by many frames should still be able to win.
_RANK_WEIGHTS = (1.0, 0.5, 0.25)
_MAX_PER_FRAME = len(_RANK_WEIGHTS)


@dataclass
class VoteReport:
    n_frames_run: int = 0
    n_frames_with_prediction: int = 0
    n_candidates: int = 0
    top_votes: float = 0.0
    agreement: float = 0.0  # share of predicting frames backing the winner

    def as_dict(self) -> dict:
        return {
            "n_frames_run": self.n_frames_run,
            "n_frames_with_prediction": self.n_frames_with_prediction,
            "n_candidates": self.n_candidates,
            "top_votes": round(self.top_votes, 3),
            "agreement": round(self.agreement, 3),
        }


def solve_with_voting(
    task: ArcTask,
    *,
    n_frames: int = 8,
    deadline: Optional[float] = None,
    seed: int = 0,
    frames: Optional[Iterable[Frame]] = None,
) -> tuple[list[list[Grid]], VoteReport]:
    """Solve the task in several frames and vote.

    Returns ([attempt_1_per_test, attempt_2_per_test], report). A test input
    with no verified prediction in any frame gets an empty candidate list, and
    the caller supplies its own fallback.
    """
    frame_list = list(frames) if frames is not None else build_frames(n_frames, seed=seed)
    n_test = task.n_test
    tally: list[defaultdict[Grid, float]] = [defaultdict(float) for _ in range(n_test)]
    # Predictions from the untransformed task break ties: it is the frame the
    # task was actually written in.
    identity_pick: list[Optional[Grid]] = [None] * n_test
    report = VoteReport()
    frames_that_predicted = 0

    for frame in frame_list:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        report.n_frames_run += 1
        try:
            variant = transform_task(task, frame)
            rules = fit_rules(variant, deadline=deadline)
        except Exception:
            continue
        if not rules:
            continue
        predicted_here = False
        for i, inp in enumerate(variant.test_inputs()):
            distinct: list[Grid] = []
            for rule in rules:
                try:
                    pred = rule.predict(inp)
                except Exception:
                    continue
                if pred is None or not is_valid_grid(pred):
                    continue
                try:
                    back = frame.invert(pred)
                except Exception:
                    continue
                if not is_valid_grid(back) or back in distinct:
                    continue
                distinct.append(back)
                if len(distinct) >= _MAX_PER_FRAME:
                    break
            for k, grid in enumerate(distinct):
                tally[i][grid] += _RANK_WEIGHTS[k]
                predicted_here = True
                if frame.is_identity and identity_pick[i] is None and k == 0:
                    identity_pick[i] = grid
        if predicted_here:
            frames_that_predicted += 1

    report.n_frames_with_prediction = frames_that_predicted
    per_test: list[list[Grid]] = []
    for i in range(n_test):
        ranked = sorted(
            tally[i].items(),
            key=lambda kv: (-kv[1], kv[0] != identity_pick[i], kv[0]),
        )
        report.n_candidates += len(ranked)
        if ranked:
            report.top_votes = max(report.top_votes, ranked[0][1])
            total = sum(v for _, v in ranked)
            if total > 0:
                report.agreement = max(report.agreement, ranked[0][1] / total)
        per_test.append([g for g, _ in ranked[:2]])
    # Entries stay aligned with the test index; None means no frame predicted.
    a1: list[Optional[Grid]] = [p[0] if p else None for p in per_test]
    a2: list[Optional[Grid]] = [
        (p[1] if len(p) > 1 else (p[0] if p else None)) for p in per_test
    ]
    return [a1, a2], report  # type: ignore[return-value]

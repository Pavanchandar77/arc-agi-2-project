"""Augmented voting. The transformations must be exact or every vote is wrong."""

from __future__ import annotations

import time

import pytest

from src.hrps.grid import as_grid
from src.hrps.solvers import fit_rules
from src.hrps.task import parse_task
from src.hrps.voting import (
    D8_FORWARD,
    D8_INVERSE_NAME,
    D8_NAMES,
    IDENTITY_PERM,
    Frame,
    build_frames,
    invert_perm,
    solve_with_voting,
    transform_task,
)

# Non-square and asymmetric: a wrong inverse cannot round-trip by accident.
ASYM = as_grid([[1, 2, 3], [4, 5, 6]])


def mk(task_id, train, test):
    payload = {
        "train": [{"input": i, "output": o} for i, o in train],
        "test": [({"input": i, "output": o} if o is not None else {"input": i}) for i, o in test],
    }
    return parse_task(task_id, payload, "test")


# --------------------------------------------------------------------------
# Exactness of the transformations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", D8_NAMES)
def test_every_d8_inverse_round_trips_on_an_asymmetric_grid(name):
    forward = D8_FORWARD[name]
    inverse = D8_FORWARD[D8_INVERSE_NAME[name]]
    assert inverse(forward(ASYM)) == ASYM
    assert forward(inverse(ASYM)) == ASYM


def test_d8_names_are_eight_distinct_transformations():
    assert len(D8_NAMES) == 8
    images = {D8_FORWARD[n](as_grid([[1, 2], [3, 4]])) for n in D8_NAMES}
    assert len(images) == 8


@pytest.mark.parametrize("name", D8_NAMES)
def test_frame_invert_undoes_frame_apply(name):
    perm = (0, 3, 1, 2, 5, 4, 7, 6, 9, 8)
    frame = Frame(name, perm)
    assert frame.invert(frame.apply(ASYM)) == ASYM


def test_colour_permutation_inverse_is_exact():
    perm = (0, 3, 1, 2, 5, 4, 7, 6, 9, 8)
    back = invert_perm(perm)
    assert all(back[perm[c]] == c for c in range(10))


def test_transform_task_moves_every_grid_including_test_inputs():
    task = mk("t", [([[1, 2]], [[2, 1]])], [([[3, 4]], [[4, 3]])])
    frame = Frame("flip_h", IDENTITY_PERM)
    moved = transform_task(task, frame)
    assert moved.train[0].input == ((2, 1),)
    assert moved.train[0].output == ((1, 2),)
    assert moved.test[0].input == ((4, 3),)


def test_transform_task_keeps_hidden_test_outputs_absent():
    task = mk("t", [([[1, 2]], [[2, 1]])], [([[3, 4]], None)])
    moved = transform_task(task, Frame("rot180", IDENTITY_PERM))
    assert moved.test[0].output is None


def test_a_solvable_task_stays_solvable_in_every_frame():
    # If a frame broke the task, no rule would verify on the transformed copy.
    task = mk(
        "rot",
        [([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]),
         ([[7, 0, 8], [0, 9, 0]], [[0, 9, 0], [8, 0, 7]])],
        [([[1, 1, 2], [3, 4, 4]], None)],
    )
    for frame in build_frames(8):
        assert fit_rules(transform_task(task, frame)), f"no rule verified under {frame}"


# --------------------------------------------------------------------------
# Frame selection
# --------------------------------------------------------------------------


def test_frames_are_distinct_and_start_with_the_identity():
    frames = build_frames(16)
    assert len(frames) == 16
    assert frames[0].is_identity
    assert len({(f.d8, f.perm) for f in frames}) == 16


def test_first_eight_frames_are_the_plain_symmetries():
    assert all(f.perm == IDENTITY_PERM for f in build_frames(8))


def test_frame_selection_is_deterministic_for_a_seed():
    a = [(f.d8, f.perm) for f in build_frames(24, seed=3)]
    b = [(f.d8, f.perm) for f in build_frames(24, seed=3)]
    assert a == b
    assert a != [(f.d8, f.perm) for f in build_frames(24, seed=4)]


def test_build_frames_handles_degenerate_counts():
    assert len(build_frames(0)) == 1
    assert len(build_frames(1)) == 1


# --------------------------------------------------------------------------
# Voting
# --------------------------------------------------------------------------


ROT_TASK = mk(
    "rot180",
    [([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]),
     ([[7, 0, 8], [0, 9, 0]], [[0, 9, 0], [8, 0, 7]])],
    [([[1, 1, 2], [3, 4, 4]], [[4, 4, 3], [2, 1, 1]])],
)


def test_votes_are_cast_in_the_original_frame():
    (a1, _), report = solve_with_voting(ROT_TASK, n_frames=8)
    assert a1[0] == ((4, 4, 3), (2, 1, 1))
    assert report.n_frames_with_prediction >= 1


def test_frames_agree_on_an_unambiguous_task():
    _, report = solve_with_voting(ROT_TASK, n_frames=8)
    assert report.agreement > 0.5


def test_unsolvable_task_yields_no_candidate_rather_than_a_guess():
    task = mk("bad", [([[1]], [[2]]), ([[1]], [[3]])], [([[1]], None)])
    (a1, a2), report = solve_with_voting(task, n_frames=4)
    assert a1 == [None] and a2 == [None]
    assert report.n_candidates == 0


def test_multiple_test_inputs_stay_aligned():
    task = mk(
        "rot",
        [([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]),
         ([[7, 0, 8], [0, 9, 0]], [[0, 9, 0], [8, 0, 7]])],
        [([[1, 1, 2], [3, 4, 4]], None), ([[5, 5, 5], [6, 7, 8]], None)],
    )
    (a1, a2), _ = solve_with_voting(task, n_frames=4)
    assert len(a1) == len(a2) == 2
    assert a1[0] == ((4, 4, 3), (2, 1, 1))
    assert a1[1] == ((8, 7, 6), (5, 5, 5))


def test_expired_deadline_stops_before_running_frames():
    _, report = solve_with_voting(ROT_TASK, n_frames=8, deadline=time.perf_counter() - 1)
    assert report.n_frames_run == 0


def test_voting_is_deterministic():
    first, _ = solve_with_voting(ROT_TASK, n_frames=12, seed=1)
    second, _ = solve_with_voting(ROT_TASK, n_frames=12, seed=1)
    assert first == second

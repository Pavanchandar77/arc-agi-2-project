"""Generating verified supervision by running the DSL backwards.

The generator writes the labels a model will be trained to believe, so the
claims that matter are that nothing unverified escapes, that uninformative
tasks are rejected rather than emitted, and that the output is diverse enough
to teach a language rather than a handful of phrases.
"""

from __future__ import annotations

import collections
import random

import pytest

from src.hrps.dsl import OP_DEFS, Program
from src.hrps.proposal import UNPROPOSABLE, parse_program, verify_program
from src.hrps.synth_programs import (
    PROPOSABLE,
    _informative,
    random_grid,
    sample_op,
    sample_program,
    synthesize,
)

PALETTE = (1, 2, 3, 4)


# --------------------------------------------------------------------------
# Nothing unverified escapes
# --------------------------------------------------------------------------


def test_every_generated_pair_verifies_against_its_own_task():
    pairs, report = synthesize(120, seed=1)
    assert pairs
    for task, program in pairs:
        assert verify_program(program, task).is_total, program.serialize()
    assert report.rejected_unverified == 0


def test_every_generated_program_survives_the_proposal_parser():
    # The corpus teaches the model to emit these strings; a string the parser
    # would reject at inference is a label that trains a wasted proposal.
    pairs, _ = synthesize(120, seed=2)
    for _, program in pairs:
        text = program.serialize()
        assert parse_program(text) is not None, text


def test_generated_tasks_carry_a_hidden_test_pair():
    pairs, _ = synthesize(20, seed=3)
    for task, _ in pairs:
        assert task.train and task.test
        assert task.test[0].output is not None


# --------------------------------------------------------------------------
# Informativeness
# --------------------------------------------------------------------------


def test_an_identity_task_is_rejected():
    grids = [((1, 2), (3, 4)), ((5, 6), (7, 8))]
    assert not _informative(grids, list(grids))


def test_a_task_collapsing_to_one_constant_output_is_rejected():
    grids = [((1, 2), (3, 4)), ((5, 6), (7, 8))]
    same = [((9, 9), (9, 9)), ((9, 9), (9, 9))]
    assert not _informative(grids, same)


def test_a_single_cell_output_is_rejected():
    assert not _informative([((1, 2), (3, 4))], [((7,),)])


def test_a_failed_execution_is_rejected():
    assert not _informative([((1, 2),)], [None])  # type: ignore[list-item]


def test_a_genuine_transformation_is_kept():
    grids = [((1, 2), (3, 4)), ((5, 6), (7, 8))]
    flipped = [((2, 1), (4, 3)), ((6, 5), (8, 7))]
    assert _informative(grids, flipped)


# --------------------------------------------------------------------------
# Diversity: a language, not a phrasebook
# --------------------------------------------------------------------------


def test_the_corpus_covers_most_of_the_operator_catalog():
    pairs, _ = synthesize(600, seed=4)
    used = {op.name for _, program in pairs for op in program.ops}
    assert len(used) >= 20, f"only {len(used)} operators appear"


def test_programs_are_mostly_distinct():
    pairs, _ = synthesize(600, seed=5)
    programs = [p.serialize() for _, p in pairs]
    assert len(set(programs)) > len(programs) * 0.3


def test_no_single_program_dominates():
    pairs, _ = synthesize(600, seed=6)
    counts = collections.Counter(p.serialize() for _, p in pairs)
    assert counts.most_common(1)[0][1] < len(pairs) * 0.1


def test_compositions_are_generated_not_just_single_operators():
    pairs, _ = synthesize(400, seed=7)
    depths = collections.Counter(len(p.ops) for _, p in pairs)
    assert sum(v for k, v in depths.items() if k >= 2) > len(pairs) * 0.2


def test_max_depth_is_respected():
    pairs, _ = synthesize(200, seed=8, max_depth=1)
    assert all(len(p.ops) == 1 for _, p in pairs)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def test_unproposable_operators_are_never_sampled():
    rng = random.Random(0)
    for _ in range(300):
        assert sample_op(rng, PALETTE).name not in UNPROPOSABLE


def test_sampled_operators_are_all_catalogued():
    rng = random.Random(1)
    for _ in range(300):
        assert sample_op(rng, PALETTE).name in OP_DEFS


def test_proposable_set_excludes_the_unproposable():
    assert set(PROPOSABLE) == set(OP_DEFS) - set(UNPROPOSABLE)


def test_sampled_programs_parse_back():
    rng = random.Random(2)
    for _ in range(200):
        program = sample_program(rng, PALETTE)
        assert parse_program(program.serialize()) is not None


def test_random_grids_are_valid_and_within_bounds():
    from src.hrps.grid import is_valid_grid

    rng = random.Random(3)
    for _ in range(80):
        grid = random_grid(rng, PALETTE)
        assert is_valid_grid(grid)
        assert 1 <= len(grid) <= 30 and 1 <= len(grid[0]) <= 30


def test_generation_is_deterministic_for_a_seed():
    a = [p.serialize() for _, p in synthesize(40, seed=9)[0]]
    b = [p.serialize() for _, p in synthesize(40, seed=9)[0]]
    assert a == b
    assert a != [p.serialize() for _, p in synthesize(40, seed=10)[0]]


def test_the_report_accounts_for_every_attempt():
    _, report = synthesize(100, seed=11)
    accounted = (
        report.produced
        + report.rejected_execution
        + report.rejected_uninformative
        + report.rejected_unverified
    )
    assert accounted == report.attempted


def test_asking_for_none_returns_none():
    pairs, report = synthesize(0, seed=12)
    assert pairs == [] and report.produced == 0

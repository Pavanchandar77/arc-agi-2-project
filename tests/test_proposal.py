"""Neural proposal, symbolic verification.

The load-bearing claims are that verification reads only the demonstrations,
that an unknown operator is rejected rather than executed, and that exactness is
never relaxed. Each has a test that fails loudly if it stops holding.
"""

from __future__ import annotations

import time

import pytest

from src.hrps.dsl import OP_DEFS, Op, Program
from src.hrps.proposal import (
    MAX_OPS,
    UNPROPOSABLE,
    extract_candidates,
    parse_program,
    propose_and_verify,
    verify_program,
)
from src.hrps.task import parse_task

FENCE = "```"


def mk(task_id, train, test):
    payload = {
        "train": [{"input": i, "output": o} for i, o in train],
        "test": [({"input": i, "output": o} if o is not None else {"input": i}) for i, o in test],
    }
    return parse_task(task_id, payload, "test")


# rot180 explains every demonstration.
ROT = mk(
    "rot",
    [([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]),
     ([[7, 0, 8], [0, 9, 0]], [[0, 9, 0], [8, 0, 7]])],
    [([[1, 1, 2], [3, 4, 4]], [[4, 4, 3], [2, 1, 1]])],
)


# --------------------------------------------------------------------------
# Parsing: fail closed
# --------------------------------------------------------------------------


def test_a_bare_operator_parses():
    assert parse_program("rot90") == Program((Op("rot90", ()),))


def test_a_pipeline_parses():
    prog = parse_program("crop_fg:0 | rot90")
    assert prog is not None and prog.names() == ("crop_fg", "rot90")


@pytest.mark.parametrize("sep", ["|", "->", "=>", "then"])
def test_every_separator_is_accepted(sep):
    prog = parse_program(f"rot90 {sep} flip_h")
    assert prog is not None and prog.names() == ("rot90", "flip_h")


def test_an_unknown_operator_is_rejected_not_executed():
    assert parse_program("definitely_not_an_op") is None
    assert parse_program("rot90 | teleport_grid") is None


def test_python_is_not_a_program():
    for hostile in (
        "__import__('os').system('ls')",
        "eval('1+1')",
        "lambda g: g",
        "open('/etc/passwd').read()",
    ):
        assert parse_program(hostile) is None


def test_abstraction_ops_are_searchable_but_not_proposable():
    # They index a mutable runtime library, so a proposal naming one is not
    # self-contained.
    assert "abs" in UNPROPOSABLE
    assert parse_program("abs:0") is None


def test_wrong_arity_is_rejected():
    assert parse_program("tile:2") is None          # tile takes two
    assert parse_program("rot90:3") is None         # rot90 takes none
    assert parse_program("upscale") is None         # upscale takes one


def test_out_of_range_arguments_are_rejected():
    assert parse_program("recolor:1,99") is None            # colour > 9
    assert parse_program("isolate_largest:5,t,0") is None    # connectivity not 4/8
    assert parse_program("keep_color:12,0") is None


def test_arguments_within_range_are_accepted():
    assert parse_program("recolor:1,2") is not None
    assert parse_program("isolate_largest:8,t,0") is not None
    assert parse_program("tile:2,3") is not None
    assert parse_program("apply_colormap:1-2;3-4") is not None


def test_identity_is_a_program():
    assert parse_program("identity") == Program(())


def test_absurdly_long_pipelines_are_rejected():
    assert parse_program(" | ".join(["rot90"] * (MAX_OPS + 1))) is None


def test_empty_and_junk_input_is_rejected():
    for junk in ("", "   ", None, "|", "| |"):
        assert parse_program(junk) is None  # type: ignore[arg-type]


def test_every_catalogued_operator_round_trips_through_the_parser():
    # Whatever search can emit, the proposer must be able to read back.
    for name, defn in OP_DEFS.items():
        if name in UNPROPOSABLE:
            continue
        assert defn.in_types, name


# --------------------------------------------------------------------------
# Candidate extraction
# --------------------------------------------------------------------------


def test_candidates_come_out_of_a_fenced_block():
    text = f"Here is my answer:\n{FENCE}\nrot180\ncrop_fg:0 | rot90\n{FENCE}"
    assert extract_candidates(text) == ["rot180", "crop_fg:0 | rot90"]


def test_prose_lines_are_dropped_not_half_parsed():
    text = "I think the rule is a rotation.\nrot180\nThat should do it."
    assert extract_candidates(text) == ["rot180"]


def test_numbering_and_bullets_are_stripped():
    assert extract_candidates("1. rot90\n2) flip_h\n- transpose") == [
        "rot90", "flip_h", "transpose",
    ]


def test_duplicate_candidates_collapse():
    assert extract_candidates("rot90\nrot90\nrot90") == ["rot90"]


def test_extraction_respects_its_limit():
    text = "\n".join(f"tile:{i},2" for i in range(1, 30))
    assert len(extract_candidates(text, limit=5)) == 5


# --------------------------------------------------------------------------
# Verification: demonstrations decide, exactly
# --------------------------------------------------------------------------


def test_a_program_explaining_every_demo_verifies():
    assert verify_program(parse_program("rot180"), ROT).is_total


def test_a_program_missing_one_demo_does_not_verify():
    verdict = verify_program(parse_program("rot90"), ROT)
    assert not verdict.is_total
    assert verdict.failed_on == 0


def test_verification_never_reads_the_test_output():
    # Same demonstrations, deliberately wrong test answer recorded. If
    # verification consulted it, the verdict would change. It must not.
    poisoned = mk(
        "poisoned",
        [(list(map(list, p.input)), list(map(list, p.output))) for p in ROT.train],
        [([[1, 1, 2], [3, 4, 4]], [[9, 9, 9], [9, 9, 9]])],
    )
    assert verify_program(parse_program("rot180"), poisoned).is_total


def test_semicolon_is_not_a_separator_because_colormaps_use_it():
    prog = parse_program("apply_colormap:1-2;3-4")
    assert prog is not None and prog.names() == ("apply_colormap",)


def test_a_task_with_no_demonstrations_verifies_nothing():
    # parse_task refuses these, but verify_program is defensive in its own
    # right: a vacuous "all zero demonstrations matched" must never be total.
    from src.hrps.task import ArcTask

    empty = ArcTask(task_id="empty", train=(), test=ROT.test, split="test")
    assert not verify_program(parse_program("rot90"), empty).is_total
    assert not verify_program(Program(()), empty).is_total


def test_near_misses_are_not_accepted():
    # Right on one demo, wrong on the other. Exactness is all or nothing.
    task = mk(
        "half",
        [([[1, 2], [3, 4]], [[4, 3], [2, 1]]),      # rot180 works
         ([[1, 2], [3, 4]], [[1, 2], [3, 4]])],     # rot180 does not
        [([[1, 2], [3, 4]], None)],
    )
    assert not verify_program(parse_program("rot180"), task).is_total


# --------------------------------------------------------------------------
# Consensus among survivors
# --------------------------------------------------------------------------


def test_a_verified_proposal_answers_the_test_input():
    (a1, _), report = propose_and_verify(["rot180"], ROT)
    assert a1[0] == ((4, 4, 3), (2, 1, 1))
    assert report.n_verified == 1
    assert report.winning_program == "rot180"


def test_unverified_proposals_produce_no_answer():
    (a1, a2), report = propose_and_verify(["rot90", "flip_h", "transpose"], ROT)
    assert a1 == [None] and a2 == [None]
    assert report.n_verified == 0


def test_invalid_proposals_are_counted_as_rejected():
    _, report = propose_and_verify(["not_an_op", "rot180", "eval('x')"], ROT)
    assert report.n_rejected == 2
    assert report.n_parsed == 1
    assert report.n_verified == 1


def test_survivors_that_agree_give_full_consensus():
    # rot180 and its two-step equivalent both explain the demos and agree.
    _, report = propose_and_verify(["rot180", "flip_h | flip_v"], ROT)
    assert report.n_verified == 2
    assert report.n_distinct_outputs == 1
    assert report.consensus == pytest.approx(1.0)


def test_duplicate_proposals_are_not_counted_twice():
    _, report = propose_and_verify(["rot180", "rot180", "rot180"], ROT)
    assert report.n_verified == 1


def test_the_simplest_survivor_is_reported_as_the_winner():
    _, report = propose_and_verify(["flip_h | flip_v", "rot180"], ROT)
    assert report.winning_program == "rot180"


def test_disagreeing_survivors_are_ranked_by_consensus_and_simplicity():
    # A task whose demos are explained by several programs that then diverge:
    # a single all-same-colour demo pair.
    task = mk(
        "ambiguous",
        [([[5, 5], [5, 5]], [[5, 5], [5, 5]])],
        [([[1, 2], [3, 4]], None)],
    )
    (a1, a2), report = propose_and_verify(["identity", "rot180", "transpose"], task)
    assert report.n_verified == 3
    assert report.n_distinct_outputs > 1
    assert a1[0] is not None and a2[0] is not None
    assert a1[0] != a2[0]
    assert 0.0 < report.consensus <= 1.0


def test_multiple_test_inputs_stay_aligned():
    task = mk(
        "two",
        [([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]])],
        [([[1, 1, 2], [3, 4, 4]], None), ([[5, 5, 5], [6, 7, 8]], None)],
    )
    (a1, _), _ = propose_and_verify(["rot180"], task)
    assert len(a1) == 2
    assert a1[0] == ((4, 4, 3), (2, 1, 1))
    assert a1[1] == ((8, 7, 6), (5, 5, 5))


def test_an_expired_deadline_stops_the_search():
    _, report = propose_and_verify(["rot180"] * 50, ROT, deadline=time.perf_counter() - 1)
    assert report.n_candidates == 0


def test_the_survivor_cap_is_respected():
    equivalents = ["rot180", "flip_h | flip_v", "flip_v | flip_h", "transpose | anti_transpose"]
    _, report = propose_and_verify(equivalents, ROT, max_survivors=2)
    assert report.n_verified == 2


def test_results_are_deterministic():
    first = propose_and_verify(["rot180", "flip_h | flip_v"], ROT)[0]
    second = propose_and_verify(["rot180", "flip_h | flip_v"], ROT)[0]
    assert first == second

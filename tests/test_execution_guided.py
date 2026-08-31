"""Execution-guided decoding: run each operator, then write the next.

The claims worth testing are that a dead branch dies immediately, that success
is detected by the executor rather than asserted by the model, that the search
is over grid states rather than program spellings, and that the test output is
never consulted on the way.
"""

from __future__ import annotations

import time

import pytest

from src.hrps.dsl import Op
from src.hrps.execution_guided import (
    ExecResult,
    apply_op,
    build_step_prompt,
    initial_state,
    is_solved,
    parse_ops,
    search_execution_guided,
    targets,
)
from src.hrps.task import parse_task


def mk(task_id, train, test):
    payload = {
        "train": [{"input": i, "output": o} for i, o in train],
        "test": [({"input": i, "output": o} if o is not None else {"input": i}) for i, o in test],
    }
    return parse_task(task_id, payload, "test")


# rot180 in one step.
ROT = mk(
    "rot",
    [([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]),
     ([[7, 0, 8], [0, 9, 0]], [[0, 9, 0], [8, 0, 7]])],
    [([[1, 1, 2], [3, 4, 4]], [[4, 4, 3], [2, 1, 1]])],
)

# Needs two operators: transpose then flip_h  (== rot90 on these).
TWO_STEP = mk(
    "two",
    [([[1, 2], [3, 4]], [[3, 1], [4, 2]]),
     ([[5, 6], [7, 8]], [[7, 5], [8, 6]])],
    [([[1, 0], [0, 2]], None)],
)


class ScriptedSolver:
    """Replies from a script, and records the prompts it was shown."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.calls = 0

    def complete(self, messages, *, temperature=0.0, attempt=0, max_new_tokens=None):
        self.prompts.append(messages[-1]["content"])
        reply = self.replies[self.calls] if self.calls < len(self.replies) else ""
        self.calls += 1
        return reply


# --------------------------------------------------------------------------
# State and execution
# --------------------------------------------------------------------------


def test_the_initial_state_is_the_untouched_inputs():
    state = initial_state(ROT)
    assert state.ops == ()
    assert state.demo_grids == tuple(p.input for p in ROT.train)
    assert state.test_grids == tuple(ROT.test_inputs())


def test_applying_an_operator_moves_every_grid():
    state = apply_op(initial_state(ROT), Op("rot180", ()))
    assert state is not None
    assert state.demo_grids[0] == ((6, 5, 4), (3, 2, 1))
    assert state.test_grids[0] == ((4, 4, 3), (2, 1, 1))


def test_a_branch_dies_when_an_operator_fails_on_a_demonstration():
    # top_half needs height >= 2; a one-row grid cannot supply it.
    task = mk("h", [([[1, 2]], [[1, 2]])], [([[3, 4]], None)])
    assert apply_op(initial_state(task), Op("top_half", ())) is None


def test_a_failure_on_the_test_input_alone_keeps_the_branch():
    # The demonstrations still verify; only that test input loses its answer.
    task = mk(
        "t",
        [([[1, 2], [3, 4]], [[1, 2]]), ([[5, 6], [7, 8]], [[5, 6]])],
        [([[9, 9]], None)],
    )
    state = apply_op(initial_state(task), Op("top_half", ()))
    assert state is not None
    assert state.demo_grids[0] == ((1, 2),)
    assert state.test_grids[0] is None


def test_solved_means_every_demonstration_matches_exactly():
    state = apply_op(initial_state(ROT), Op("rot180", ()))
    assert is_solved(state, ROT)
    near = apply_op(initial_state(ROT), Op("flip_h", ()))
    assert not is_solved(near, ROT)


def test_states_are_keyed_by_grids_not_by_spelling():
    # rot180 and flip_h|flip_v are different programs reaching the same state.
    a = apply_op(initial_state(ROT), Op("rot180", ()))
    b = apply_op(apply_op(initial_state(ROT), Op("flip_h", ())), Op("flip_v", ()))
    assert a.key() == b.key()
    assert a.program.serialize() != b.program.serialize()


# --------------------------------------------------------------------------
# The prompt shows the loop
# --------------------------------------------------------------------------


def test_the_prompt_shows_current_state_beside_the_target():
    text = build_step_prompt(ROT, initial_state(ROT))
    assert "current state" in text and "target output" in text
    assert "1 2 3" in text   # the untouched input
    assert "6 5 4" in text   # the target


def test_the_prompt_carries_the_program_so_far():
    state = apply_op(initial_state(ROT), Op("flip_h", ()))
    text = build_step_prompt(ROT, state)
    assert "flip_h" in text
    assert "3 2 1" in text, "must show what flip_h actually produced"


def test_the_prompt_never_contains_the_test_answer():
    # The one invariant that makes any of these numbers meaningful.
    text = build_step_prompt(ROT, initial_state(ROT))
    answer = "4 4 3"
    assert answer not in text


# --------------------------------------------------------------------------
# Parsing single operators
# --------------------------------------------------------------------------


def test_operators_are_read_one_per_line():
    ops = parse_ops("rot90\nflip_h\ntranspose")
    assert [o.name for o in ops] == ["rot90", "flip_h", "transpose"]


def test_prose_and_unknown_names_are_dropped():
    ops = parse_ops("I think we should\nrot90\nteleport_grid\nmaybe that works")
    assert [o.name for o in ops] == ["rot90"]


def test_a_whole_pipeline_is_not_a_single_operator():
    assert parse_ops("rot90 | flip_h") == []


def test_duplicate_operators_collapse():
    assert len(parse_ops("rot90\nrot90\nrot90")) == 1


def test_operators_with_arguments_parse():
    ops = parse_ops("tile:2,3\ncrop_fg:0")
    assert [o.name for o in ops] == ["tile", "crop_fg"]


def test_out_of_range_arguments_are_dropped_before_execution():
    assert parse_ops("recolor:1,99") == []


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------


def test_a_one_step_solution_is_found_and_verified():
    result = search_execution_guided(ScriptedSolver(["rot180"]), ROT)
    assert result.solved
    assert result.report.program == "rot180"
    assert result.answers()[0] == ((4, 4, 3), (2, 1, 1))


def test_the_search_composes_across_steps():
    # Neither operator alone solves it; the pair does.
    solver = ScriptedSolver(["transpose", "flip_h"])
    result = search_execution_guided(solver, TWO_STEP, beam_width=1)
    assert result.solved
    assert result.report.program == "transpose | flip_h"


def test_the_second_prompt_shows_what_the_first_operator_did():
    # This is the whole idea: the model writes op two having seen op one's work.
    solver = ScriptedSolver(["transpose", "flip_h"])
    search_execution_guided(solver, TWO_STEP, beam_width=1)
    assert len(solver.prompts) >= 2
    assert "transpose" in solver.prompts[1]
    assert "1 3" in solver.prompts[1], "the transposed grid must be shown"


def test_an_invalid_alternative_is_rejected_while_a_good_one_survives():
    # The prompt asks for alternatives on separate lines, so one reply carries
    # several. top_half cannot run on a one-row grid; flip_h can.
    task = mk("h", [([[1, 2]], [[2, 1]])], [([[3, 4]], None)])
    solver = ScriptedSolver(["top_half\nflip_h"])
    result = search_execution_guided(solver, task, beam_width=2)
    assert result.report.n_ops_rejected == 1
    assert result.solved and result.report.program == "flip_h"


def test_a_depth_whose_every_proposal_fails_ends_the_search():
    # Nothing survived, so there is nothing to build the next operator on.
    task = mk("h", [([[1, 2]], [[2, 1]])], [([[3, 4]], None)])
    solver = ScriptedSolver(["top_half", "flip_h"])
    result = search_execution_guided(solver, task, beam_width=2)
    assert not result.solved
    assert solver.calls == 1, "must not keep asking once the beam is empty"


def test_a_model_offering_nothing_usable_fails_cleanly():
    result = search_execution_guided(ScriptedSolver(["no idea", "still nothing"]), ROT)
    assert not result.solved
    assert result.answers() == []


def test_an_expired_deadline_stops_before_any_model_call():
    solver = ScriptedSolver(["rot180"])
    result = search_execution_guided(ROT and solver, ROT, deadline=time.perf_counter() - 1)
    assert solver.calls == 0
    assert not result.solved


def test_the_search_respects_its_depth_limit():
    solver = ScriptedSolver(["flip_h"] * 10)
    result = search_execution_guided(solver, TWO_STEP, max_depth=2, beam_width=1)
    assert result.report.max_depth_reached <= 2


def test_an_identity_task_is_solved_without_a_model_call():
    task = mk("id", [([[1, 2]], [[1, 2]]), ([[3, 4]], [[3, 4]])], [([[5, 6]], None)])
    solver = ScriptedSolver([])
    result = search_execution_guided(solver, task)
    assert result.solved and result.report.program == "identity"
    assert solver.calls == 0


def test_a_task_with_no_demonstrations_solves_nothing():
    from src.hrps.task import ArcTask

    empty = ArcTask(task_id="e", train=(), test=ROT.test, split="test")
    result = search_execution_guided(ScriptedSolver(["rot180"]), empty)
    assert not result.solved


def test_success_is_confirmed_by_the_standalone_verifier():
    # The branch claiming to match is not taken at its word.
    result = search_execution_guided(ScriptedSolver(["rot180"]), ROT)
    from src.hrps.proposal import parse_program, verify_program

    assert verify_program(parse_program(result.report.program), ROT).is_total


def test_the_report_is_serializable():
    import json

    result = search_execution_guided(ScriptedSolver(["rot180"]), ROT)
    assert "program" in json.dumps(result.report.as_dict())

"""Sampling programs and letting the demonstrations decide.

A fake completer stands in for the model so the propose-verify loop is testable
without weights. What is under test is the loop, not the model: that greedy
comes first, that junk proposals cost nothing, and that expert iteration only
ever writes labels it has re-verified itself.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.hrps.program_prompt import (
    SYSTEM_PROMPT,
    build_inference_messages,
    build_prompt,
    operator_reference,
)
from src.hrps.program_solver import append_verified, solve_by_proposal
from src.hrps.proposal import parse_program
from src.hrps.task import parse_task

FENCE = "```"


def mk(task_id, train, test):
    payload = {
        "train": [{"input": i, "output": o} for i, o in train],
        "test": [({"input": i, "output": o} if o is not None else {"input": i}) for i, o in test],
    }
    return parse_task(task_id, payload, "test")


ROT = mk(
    "rot",
    [([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]),
     ([[7, 0, 8], [0, 9, 0]], [[0, 9, 0], [8, 0, 7]])],
    [([[1, 1, 2], [3, 4, 4]], [[4, 4, 3], [2, 1, 1]])],
)


class FakeSolver:
    """Returns canned completions and records how it was called."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, messages, *, temperature=0.0, attempt=0, max_new_tokens=None):
        self.calls.append(
            {"temperature": temperature, "attempt": attempt, "messages": messages}
        )
        if attempt < len(self.replies):
            return self.replies[attempt]
        return self.replies[-1] if self.replies else None


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------


def test_the_prompt_asks_for_programs_not_grids():
    assert "program" in SYSTEM_PROMPT.lower()
    assert "never by" in SYSTEM_PROMPT.lower()


def test_the_catalog_is_generated_from_the_dsl():
    # Advertising an operator the executor lacks would train hallucination.
    from src.hrps.dsl import OP_DEFS
    from src.hrps.proposal import UNPROPOSABLE

    reference = operator_reference()
    for name in OP_DEFS:
        if name in UNPROPOSABLE:
            continue
        assert name in reference, f"{name} missing from the prompt catalog"


def test_unproposable_operators_are_not_advertised():
    assert "abs:" not in operator_reference()


def test_the_prompt_carries_the_demonstrations_and_the_test_input():
    text = build_prompt(ROT)
    assert "1 2 3" in text and "6 5 4" in text   # demo in and out
    assert "1 1 2" in text                        # test input
    assert "Test input" in text


def test_inference_messages_are_system_plus_user():
    msgs = build_inference_messages(ROT)
    assert [m["role"] for m in msgs] == ["system", "user"]


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def test_a_verified_proposal_solves_the_task():
    result = solve_by_proposal(FakeSolver(["rot180"]), ROT, n_samples=1)
    assert result.solved
    assert result.attempts[0][0] == ((4, 4, 3), (2, 1, 1))


def test_the_first_sample_is_greedy_and_the_rest_are_not():
    fake = FakeSolver(["rot180"])
    solve_by_proposal(fake, ROT, n_samples=4, temperature=0.9)
    assert fake.calls[0]["temperature"] == 0.0
    assert all(c["temperature"] == 0.9 for c in fake.calls[1:])


def test_junk_proposals_cost_nothing_and_solve_nothing():
    fake = FakeSolver(["I think it rotates.", "banana", "rot90 | teleport"])
    result = solve_by_proposal(fake, ROT, n_samples=3)
    assert not result.solved
    assert result.attempts[0][0] is None


def test_one_good_proposal_among_junk_still_wins():
    fake = FakeSolver(["nonsense here", "flip_h", "rot180"])
    result = solve_by_proposal(fake, ROT, n_samples=3)
    assert result.solved
    assert result.attempts[0][0] == ((4, 4, 3), (2, 1, 1))


def test_candidates_are_collected_across_samples():
    fake = FakeSolver(["rot90", "flip_h", "rot180"])
    result = solve_by_proposal(fake, ROT, n_samples=3)
    assert result.n_raw_candidates == 3
    assert result.report.n_verified == 1


def test_several_programs_in_one_reply_are_all_considered():
    fake = FakeSolver([f"{FENCE}\nrot90\nflip_h\nrot180\n{FENCE}"])
    result = solve_by_proposal(fake, ROT, n_samples=1)
    assert result.n_raw_candidates == 3
    assert result.solved


def test_a_model_that_returns_nothing_is_survivable():
    result = solve_by_proposal(FakeSolver([None]), ROT, n_samples=2)
    assert not result.solved
    assert result.report.n_candidates == 0


def test_an_expired_deadline_stops_sampling():
    fake = FakeSolver(["rot180"])
    result = solve_by_proposal(fake, ROT, n_samples=8, deadline=time.perf_counter() - 1)
    assert fake.calls == []
    assert result.n_samples == 0


def test_the_report_is_serializable():
    result = solve_by_proposal(FakeSolver(["rot180"]), ROT, n_samples=1)
    payload = json.dumps(result.as_dict())
    assert "winning_program" in payload


# --------------------------------------------------------------------------
# Expert iteration
# --------------------------------------------------------------------------


def test_verified_programs_are_appended_as_training_examples(tmp_path: Path):
    corpus = tmp_path / "programs.jsonl"
    assert append_verified(ROT, ["rot180"], corpus) == 1
    rows = [json.loads(l) for l in corpus.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["program"] == "rot180"
    assert rows[0]["messages"][-1]["content"] == "rot180"
    assert rows[0]["messages"][-1]["role"] == "assistant"


def test_unverified_programs_are_never_appended(tmp_path: Path):
    corpus = tmp_path / "programs.jsonl"
    # rot90 does not explain the demos; nonsense does not even parse.
    assert append_verified(ROT, ["rot90", "not_an_op", "eval('x')"], corpus) == 0
    assert not corpus.exists() or corpus.read_text() == ""


def test_appending_re_verifies_rather_than_trusting_the_caller(tmp_path: Path):
    # Caller insists these are good. Two of them are not, and the corpus is
    # what later training will believe, so the claim is checked not taken.
    corpus = tmp_path / "programs.jsonl"
    assert append_verified(ROT, ["rot180", "flip_h", "transpose"], corpus) == 1


def test_duplicates_are_not_appended_twice(tmp_path: Path):
    corpus = tmp_path / "programs.jsonl"
    known: set[str] = set()
    assert append_verified(ROT, ["rot180"], corpus, known=known) == 1
    assert append_verified(ROT, ["rot180"], corpus, known=known) == 0
    assert len(corpus.read_text().strip().splitlines()) == 1


def test_appending_creates_the_corpus_directory(tmp_path: Path):
    corpus = tmp_path / "nested" / "deeper" / "programs.jsonl"
    assert append_verified(ROT, ["rot180"], corpus) == 1
    assert corpus.is_file()


# --------------------------------------------------------------------------
# Wiring into the two-phase runner
# --------------------------------------------------------------------------


def test_the_runner_writes_a_verified_program_answer():
    from src.kaggle_llm_run import LlmPhaseConfig, _program_pass

    raw = {
        "train": [{"input": [[1, 2, 3], [4, 5, 6]], "output": [[6, 5, 4], [3, 2, 1]]},
                  {"input": [[7, 0, 8], [0, 9, 0]], "output": [[0, 9, 0], [8, 0, 7]]}],
        "test": [{"input": [[1, 1, 2], [3, 4, 4]]}],
    }
    entry = [{"attempt_1": [[0]], "attempt_2": [[0]]}]
    cfg = LlmPhaseConfig(
        model_path="x", adapter_path=None, seconds=60, per_task_seconds=60,
        ttt_steps=0, max_new_tokens=64, temperature=0.8, top_p=0.9, seed=0,
    )
    wrote = _program_pass(
        FakeSolver(["rot180"]), raw, "rot", entry, cfg, time.perf_counter() + 30
    )
    assert wrote == {0}
    assert entry[0]["attempt_1"] == [[4, 4, 3], [2, 1, 1]]


def test_the_runner_leaves_the_entry_alone_when_nothing_verifies():
    from src.kaggle_llm_run import LlmPhaseConfig, _program_pass

    raw = {
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3, 4]]}],
    }
    original = [{"attempt_1": [[0]], "attempt_2": [[0]]}]
    entry = [dict(original[0])]
    cfg = LlmPhaseConfig(
        model_path="x", adapter_path=None, seconds=60, per_task_seconds=60,
        ttt_steps=0, max_new_tokens=64, temperature=0.8, top_p=0.9, seed=0,
    )
    assert _program_pass(
        FakeSolver(["not_a_program"]), raw, "t", entry, cfg, time.perf_counter() + 30
    ) == set()
    assert entry == original


def test_a_malformed_task_does_not_crash_the_runner():
    from src.kaggle_llm_run import LlmPhaseConfig, _program_pass

    cfg = LlmPhaseConfig(
        model_path="x", adapter_path=None, seconds=60, per_task_seconds=60,
        ttt_steps=0, max_new_tokens=64, temperature=0.8, top_p=0.9, seed=0,
    )
    assert _program_pass(
        FakeSolver(["rot180"]), {"garbage": True}, "t", [{}], cfg,
        time.perf_counter() + 30,
    ) == set()


def test_program_proposal_is_on_by_default():
    from src.kaggle_llm_run import LlmPhaseConfig

    cfg = LlmPhaseConfig(
        model_path="x", adapter_path=None, seconds=1, per_task_seconds=1,
        ttt_steps=0, max_new_tokens=1, temperature=0.0, top_p=1.0, seed=0,
    )
    assert cfg.propose_programs is True


def test_sampling_is_abandoned_when_the_model_emits_no_programs():
    # An untrained model must cost one generation, not the whole budget.
    fake = FakeSolver(["I have no idea what this puzzle wants."])
    result = solve_by_proposal(fake, ROT, n_samples=8)
    assert result.n_samples == 2, "should stop once it is clear no program is coming"
    assert not result.solved


def test_sampling_continues_while_programs_are_being_produced():
    fake = FakeSolver(["rot90", "flip_h", "transpose", "rot180"])
    result = solve_by_proposal(fake, ROT, n_samples=4)
    assert result.n_samples == 4
    assert result.solved


def test_a_single_sample_request_is_honoured():
    fake = FakeSolver(["nothing parseable here"])
    result = solve_by_proposal(fake, ROT, n_samples=1)
    assert result.n_samples == 1

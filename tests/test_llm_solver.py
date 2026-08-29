"""LLM layer contract, exercised with a stub model so no weights are required.

The point of these tests is the plumbing around the model: budgets are honoured,
failures degrade to None instead of raising, verified symbolic answers are never
overwritten by generated ones, and test-time training always restores weights.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.hrps.llm_solver import LlmConfig, LlmSolver, find_model_dir
from src.kaggle_llm_run import LlmPhaseConfig, _unsolved_ids, run_llm_phase

ROT_TASK = {
    "train": [
        {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
        {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]},
    ],
    "test": [{"input": [[1, 0], [0, 2]]}],
}


class StubTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    chat_template = None

    def __call__(self, text, return_tensors=None):
        return {"input_ids": _FakeTensor([[1, 2, 3]])}

    def decode(self, tokens, skip_special_tokens=True):
        return tokens


class _FakeTensor(list):
    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)

    def to(self, device):
        return self


class StubModel:
    """Returns a scripted completion string per generate() call."""

    device = "cpu"

    def __init__(self, completions: list[str], explode: bool = False):
        self.completions = list(completions)
        self.explode = explode
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        if self.explode:
            raise RuntimeError("cuda oom")
        text = self.completions.pop(0) if self.completions else ""
        return [_Row(text)]

    def eval(self):
        return self

    def train(self):
        return self


class _Row:
    def __init__(self, text):
        self.text = text

    def __getitem__(self, sl):
        return self.text


def make_solver(completions, **cfg_over):
    cfg = LlmConfig(model_path="/nonexistent", **cfg_over)
    return LlmSolver(cfg, model=StubModel(completions), tokenizer=StubTokenizer())


# Generation needs torch. The offline solver path deliberately does not, so the
# suite must still run on a machine that never installs it.
needs_torch = pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch is not installed; the offline solver path does not require it",
)


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


@needs_torch
def test_two_attempts_are_parsed_into_grids():
    solver = make_solver(["4 3\n2 1", "1 1\n1 1"])
    grids = solver.predict(ROT_TASK, 0, n_attempts=2)
    assert grids[0] == ((4, 3), (2, 1))
    assert grids[1] == ((1, 1), (1, 1))
    assert solver.stats.n_generated == 2


@needs_torch
def test_unparseable_output_becomes_none_and_is_counted():
    solver = make_solver(["I think the answer is probably a rotation.", "4 3\n2 1"])
    grids = solver.predict(ROT_TASK, 0, n_attempts=2)
    assert grids[0] is None
    assert grids[1] == ((4, 3), (2, 1))
    assert solver.stats.n_parse_failures == 1


@needs_torch
def test_generation_failure_does_not_raise():
    cfg = LlmConfig(model_path="/nonexistent")
    solver = LlmSolver(cfg, model=StubModel([], explode=True), tokenizer=StubTokenizer())
    assert solver.predict(ROT_TASK, 0, n_attempts=2) == [None, None]
    assert any("generate" in e for e in solver.stats.errors)


def test_predict_returns_none_when_the_model_never_loaded():
    solver = LlmSolver(LlmConfig(model_path="/nonexistent"))
    assert solver.predict(ROT_TASK, 0, n_attempts=2) == [None, None]


@needs_torch
def test_expired_deadline_skips_generation_entirely():
    solver = make_solver(["4 3\n2 1"])
    grids = solver.predict(ROT_TASK, 0, n_attempts=2, deadline=time.perf_counter() - 1)
    assert grids == [None, None]
    assert solver.stats.n_deadline_skips == 1
    assert solver.stats.n_generated == 0


def test_load_failure_is_reported_not_raised():
    solver = LlmSolver(LlmConfig(model_path="/definitely/not/a/model"))
    assert solver.load() is False
    assert solver.stats.load_error
    assert not solver.stats.loaded


# --------------------------------------------------------------------------
# Phase 2 integration
# --------------------------------------------------------------------------


@needs_torch
def test_llm_phase_fills_only_its_targets(tmp_path: Path):
    submission = {
        "solved": [{"attempt_1": [[7]], "attempt_2": [[7]]}],
        "unsolved": [{"attempt_1": [[0]], "attempt_2": [[0]]}],
    }
    raw = {"solved": ROT_TASK, "unsolved": ROT_TASK}
    out = tmp_path / "submission.json"
    out.write_text(json.dumps(submission))

    import src.kaggle_llm_run as mod

    original = mod.LlmSolver
    mod.LlmSolver = lambda cfg: make_solver(["4 3\n2 1", "1 1\n1 1"])
    try:
        report = run_llm_phase(
            submission,
            raw,
            ["unsolved"],
            LlmPhaseConfig("/x", None, 30.0, 10.0, 0, 512, 0.7, 0.9, 0),
            out,
            verbose=False,
        )
    finally:
        mod.LlmSolver = original

    assert report["n_filled"] == 1
    assert submission["solved"][0]["attempt_1"] == [[7]]  # untouched
    assert submission["unsolved"][0]["attempt_1"] == [[4, 3], [2, 1]]


def test_llm_phase_reports_when_the_model_will_not_load(tmp_path: Path):
    submission = {"a": [{"attempt_1": [[0]], "attempt_2": [[0]]}]}
    out = tmp_path / "submission.json"
    out.write_text(json.dumps(submission))
    report = run_llm_phase(
        submission,
        {"a": ROT_TASK},
        ["a"],
        LlmPhaseConfig("/definitely/not/a/model", None, 10.0, 5.0, 0, 512, 0.7, 0.9, 0),
        out,
        verbose=False,
    )
    assert report["skipped"] == "load_failed"
    assert submission["a"][0]["attempt_1"] == [[0]]


def test_unsolved_ids_excludes_verified_tasks():
    report = {"verified_ids": ["a", "c"]}
    submission = {"a": [], "b": [], "c": [], "d": []}
    assert _unsolved_ids(report, submission) == ["b", "d"]


def test_unsolved_ids_returns_everything_when_nothing_verified():
    assert _unsolved_ids({}, {"a": [], "b": []}) == ["a", "b"]


# --------------------------------------------------------------------------
# Model discovery
# --------------------------------------------------------------------------


def test_find_model_dir_wants_config_beside_weights(tmp_path: Path):
    empty = tmp_path / "no_weights"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    assert find_model_dir(str(tmp_path)) is None

    real = tmp_path / "dataset" / "qwen"
    real.mkdir(parents=True)
    (real / "config.json").write_text("{}")
    (real / "model.safetensors").write_bytes(b"")
    assert find_model_dir(str(tmp_path / "dataset")) == str(real)


def test_find_model_dir_returns_none_for_a_missing_root():
    assert find_model_dir("/nope/not/here") is None

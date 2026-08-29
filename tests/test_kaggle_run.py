"""The runner's contract: a submittable file exists no matter what goes wrong."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.kaggle_run import (
    RunConfig,
    find_challenges,
    load_tasks,
    run,
    score_submission,
    validate_submission,
)

ROT = {
    "train": [
        {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
        {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]},
    ],
    "test": [{"input": [[1, 0], [0, 2]]}],
}
ROT_SOLUTION = [[[2, 0], [0, 1]]]
UNSOLVABLE = {
    "train": [{"input": [[1]], "output": [[2]]}, {"input": [[1]], "output": [[3]]}],
    "test": [{"input": [[1]]}, {"input": [[4]]}],
}


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    folder = tmp_path / "arc-prize-2025"
    folder.mkdir()
    (folder / "arc-agi_test_challenges.json").write_text(
        json.dumps({"rot": ROT, "bad": UNSOLVABLE, "broken": {"train": [], "test": []}})
    )
    (folder / "arc-agi_test_solutions.json").write_text(
        json.dumps({"rot": ROT_SOLUTION, "bad": [[[9]], [[9]]], "broken": [[[9]]]})
    )
    return folder


def cfg_for(dataset: Path, tmp_path: Path, **over) -> RunConfig:
    base = dict(
        challenges=dataset / "arc-agi_test_challenges.json",
        solutions=dataset / "arc-agi_test_solutions.json",
        output=tmp_path / "submission.json",
        total_seconds=60.0,
        per_task_seconds=3.0,
        min_task_seconds=0.5,
        workers=1,
        use_search=False,
        stage="L",
        limit=None,
        verbose=False,
    )
    base.update(over)
    return RunConfig(**base)  # type: ignore[arg-type]


def test_run_emits_every_task_id_and_valid_schema(dataset: Path, tmp_path: Path):
    report = run(cfg_for(dataset, tmp_path))
    assert report["schema_problems"] == []
    submission = json.loads((tmp_path / "submission.json").read_text())
    assert set(submission) == {"rot", "bad", "broken"}
    assert report["score"]["tasks_solved"] == 1  # the rotation task


def test_entry_length_matches_test_input_count(dataset: Path, tmp_path: Path):
    run(cfg_for(dataset, tmp_path))
    submission = json.loads((tmp_path / "submission.json").read_text())
    assert len(submission["rot"]) == 1
    assert len(submission["bad"]) == 2  # two test inputs, neither solvable


def test_a_task_that_cannot_be_parsed_still_gets_an_entry(dataset: Path, tmp_path: Path):
    run(cfg_for(dataset, tmp_path))
    submission = json.loads((tmp_path / "submission.json").read_text())
    assert validate_submission({"broken": submission["broken"]}, ["broken"]) == []


def test_zero_budget_still_writes_a_complete_file(dataset: Path, tmp_path: Path):
    report = run(cfg_for(dataset, tmp_path, total_seconds=0.0))
    submission = json.loads((tmp_path / "submission.json").read_text())
    assert set(submission) == {"rot", "bad", "broken"}
    assert report["schema_problems"] == []


def test_parallel_run_matches_serial_schema(dataset: Path, tmp_path: Path):
    report = run(cfg_for(dataset, tmp_path, workers=2))
    assert report["schema_problems"] == []
    submission = json.loads((tmp_path / "submission.json").read_text())
    assert set(submission) == {"rot", "bad", "broken"}


def test_limit_scores_only_attempted_tasks(dataset: Path, tmp_path: Path):
    report = run(cfg_for(dataset, tmp_path, limit=1))
    assert report["score"]["tasks"] == 1


# --------------------------------------------------------------------------
# Schema validator
# --------------------------------------------------------------------------


def test_validator_flags_a_missing_task():
    assert validate_submission({}, ["a"])


def test_validator_flags_ragged_and_oversized_grids():
    assert validate_submission({"a": [{"attempt_1": [[1, 2], [3]], "attempt_2": [[1]]}]}, ["a"])
    big = [[0] * 31]
    assert validate_submission({"a": [{"attempt_1": big, "attempt_2": [[1]]}]}, ["a"])


def test_validator_flags_out_of_range_colours():
    assert validate_submission({"a": [{"attempt_1": [[10]], "attempt_2": [[1]]}]}, ["a"])
    assert validate_submission({"a": [{"attempt_1": [[-1]], "attempt_2": [[1]]}]}, ["a"])


def test_validator_flags_a_missing_second_attempt():
    assert validate_submission({"a": [{"attempt_1": [[1]]}]}, ["a"])


def test_validator_accepts_a_well_formed_submission():
    assert validate_submission({"a": [{"attempt_1": [[1]], "attempt_2": [[2]]}]}, ["a"]) == []


# --------------------------------------------------------------------------
# Discovery and scoring
# --------------------------------------------------------------------------


def test_find_challenges_prefers_the_test_split(dataset: Path):
    assert find_challenges(str(dataset / "arc-agi_test_challenges.json")).name.endswith(
        "test_challenges.json"
    )


def test_find_challenges_raises_when_absent(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.kaggle_run.SEARCH_ROOTS", ("nowhere",))
    with pytest.raises(FileNotFoundError):
        find_challenges()


def test_load_tasks_keeps_ids_for_unparseable_tasks(dataset: Path):
    tasks = load_tasks(dataset / "arc-agi_test_challenges.json")
    assert [t.task_id for t in tasks] == ["bad", "broken", "rot"]


def test_score_counts_pass_at_2_on_either_attempt():
    solutions = {"a": [[[1]]]}
    sub_second = {"a": [{"attempt_1": [[9]], "attempt_2": [[1]]}]}
    import json as _json
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        _json.dump(solutions, fh)
        path = Path(fh.name)
    assert score_submission(sub_second, path)["pass_at_2"] == 1.0
    assert score_submission({"a": [{"attempt_1": [[9]], "attempt_2": [[8]]}]}, path)["pass_at_2"] == 0.0

"""Building the Kaggle-format pair from a directory of ARC task files.

The one thing that must never break: test outputs belong in the solutions file
and nowhere else. A challenges file carrying answers would let the solver read
what it is scored against, and every number after that would be worthless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.make_challenges import build, find_split_dir, main


def write_task(directory: Path, task_id: str, *, with_answers: bool = True) -> None:
    payload = {
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3, 4]], "output": [[4, 3]]}] if with_answers
        else [{"input": [[3, 4]]}],
    }
    (directory / f"{task_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_test_outputs_are_withheld_from_the_challenges_file(tmp_path: Path):
    write_task(tmp_path, "a")
    challenges, solutions = build(tmp_path)
    assert list(challenges["a"]["test"][0]) == ["input"]
    assert "output" not in challenges["a"]["test"][0]
    assert solutions["a"] == [[[4, 3]]]


def test_demonstrations_keep_their_outputs(tmp_path: Path):
    # The demonstrations are the whole point; only the test answer is withheld.
    write_task(tmp_path, "a")
    challenges, _ = build(tmp_path)
    assert challenges["a"]["train"][0]["output"] == [[2, 1]]


def test_every_task_in_the_directory_appears(tmp_path: Path):
    for name in ("a", "b", "c"):
        write_task(tmp_path, name)
    challenges, solutions = build(tmp_path)
    assert set(challenges) == {"a", "b", "c"}
    assert set(solutions) == {"a", "b", "c"}


def test_a_task_without_answers_yields_no_solution_entry(tmp_path: Path):
    write_task(tmp_path, "unanswered", with_answers=False)
    challenges, solutions = build(tmp_path)
    assert "unanswered" in challenges
    assert "unanswered" not in solutions


def test_multiple_test_inputs_are_all_carried(tmp_path: Path):
    (tmp_path / "multi.json").write_text(json.dumps({
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}, {"input": [[5]], "output": [[6]]}],
    }), encoding="utf-8")
    challenges, solutions = build(tmp_path)
    assert len(challenges["multi"]["test"]) == 2
    assert solutions["multi"] == [[[4]], [[6]]]


def test_the_output_is_scoreable_by_the_runner(tmp_path: Path):
    from src.kaggle_run import score_submission

    write_task(tmp_path, "a")
    _, solutions = build(tmp_path)
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    sol_path = out / "sol.json"
    sol_path.write_text(json.dumps(solutions), encoding="utf-8")

    perfect = {"a": [{"attempt_1": [[4, 3]], "attempt_2": [[0]]}]}
    assert score_submission(perfect, sol_path)["tasks_solved"] == 1
    wrong = {"a": [{"attempt_1": [[0]], "attempt_2": [[0]]}]}
    assert score_submission(wrong, sol_path)["tasks_solved"] == 0


def test_an_empty_directory_is_reported_not_silently_accepted(tmp_path: Path):
    challenges, solutions = build(tmp_path)
    assert challenges == {} and solutions == {}


def test_a_missing_split_fails_with_an_actionable_message(tmp_path: Path):
    with pytest.raises(SystemExit) as excinfo:
        find_split_dir("nope", str(tmp_path))
    assert "no such directory" in str(excinfo.value)


def test_main_writes_both_files(tmp_path: Path):
    split = tmp_path / "data" / "evaluation"
    split.mkdir(parents=True)
    write_task(split, "a")
    out = tmp_path / "out"
    assert main(["--split", "evaluation", "--data-root", str(tmp_path / "data"),
                 "--out", str(out)]) == 0
    assert (out / "arc-agi_test_challenges.json").is_file()
    assert (out / "arc-agi_test_solutions.json").is_file()

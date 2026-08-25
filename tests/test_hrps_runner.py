"""Active Bond runner: budgets, JSON actions, four systems, adapter save/load."""

from __future__ import annotations

import json
from pathlib import Path

from src.hrps.backend import load_scripted_adapter, save_scripted_adapter
from src.hrps.bond import SYSTEMS, bond_deltas, run_bond_eval
from src.hrps.kinds import Kind, kind_of
from src.hrps.model import ScriptedModel
from src.hrps.runner import RunnerBudget, run_direct, run_hrps, run_system
from src.hrps.task import parse_task


def _task():
    return parse_task(
        "synth_rot180",
        {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
                {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]},
            ],
            "test": [{"input": [[1, 0], [0, 2]], "output": [[2, 0], [0, 1]]}],
        },
        "training",
    )


def _j(action, **arguments):
    return json.dumps({"action": action, "arguments": arguments})


def test_runner_components_labeled():
    assert kind_of("bond_runner") is Kind.EXACT
    assert kind_of("bond_model_backend") is Kind.EXACT


def test_base_direct_works_without_hrps():
    task = _task()
    model = ScriptedModel(responses=["1 0\n0 2", "1 0\n0 2"])
    res = run_direct(task, model, RunnerBudget(max_model_calls=2, max_seconds=5), system="base_direct")
    assert res.system == "base_direct"
    assert res.episode.solved is False
    assert res.termination in {"direct_done", "max_calls"}
    assert all(i.parsed_action and i.parsed_action.get("action") == "answer" for i in res.interactions)


def test_base_hrps_works_without_adapter():
    task = _task()
    model = ScriptedModel(
        responses=[
            _j("inspect_objects"),
            _j("execute_program", program="rot180"),
            _j("commit_candidates", program="rot180"),
        ]
    )
    res = run_hrps(task, model, RunnerBudget(max_model_calls=8, max_seconds=5), system="base_hrps")
    assert res.system == "base_hrps"
    assert res.episode.solved is True
    assert res.episode.joint_demo_exact
    assert res.termination == "committed"
    assert res.episode.n_representation_requests >= 1
    assert res.episode.n_verifier_calls >= 1


def test_runner_terminates_on_call_budget():
    task = _task()
    model = ScriptedModel(responses=[_j("inspect_objects")] * 5)
    res = run_hrps(task, model, RunnerBudget(max_model_calls=3, max_seconds=5), system="base_hrps")
    assert res.termination == "max_calls"
    assert res.episode.n_model_calls == 3


def test_runner_terminates_on_token_budget():
    task = _task()
    model = ScriptedModel(responses=[_j("inspect_objects")] * 10)
    res = run_hrps(
        task,
        model,
        RunnerBudget(max_model_calls=20, max_total_tokens=5, max_seconds=5, max_tokens_per_call=8),
        system="base_hrps",
    )
    assert res.termination == "max_tokens"


def test_runner_terminates_on_time_budget():
    task = _task()
    model = ScriptedModel(responses=[_j("inspect_objects")] * 10)
    res = run_hrps(task, model, RunnerBudget(max_model_calls=50, max_seconds=0.0), system="base_hrps")
    assert res.termination == "timeout"


def test_invalid_actions_are_rejected_and_logged():
    task = _task()
    model = ScriptedModel(
        responses=[
            json.dumps({"action": "shell", "arguments": {"cmd": "rm"}}),
            "import os; os.system('x')",
            _j("execute_program", program="rot180"),
            _j("commit_candidates", program="rot180"),
        ]
    )
    res = run_hrps(task, model, RunnerBudget(max_model_calls=8, max_seconds=5), system="base_hrps")
    assert res.n_invalid_actions >= 2
    assert res.episode.solved is True


def test_same_seed_reproduces_scripted_trace():
    task = _task()
    responses = [
        _j("inspect_objects"),
        _j("execute_program", program="rot180"),
        _j("commit_candidates", program="rot180"),
    ]
    b = RunnerBudget(seed=7, max_model_calls=8, max_seconds=5)
    a = run_hrps(task, ScriptedModel(responses=list(responses)), b, system="base_hrps")
    c = run_hrps(task, ScriptedModel(responses=list(responses)), b, system="base_hrps")
    assert [i.parsed_action for i in a.interactions] == [i.parsed_action for i in c.interactions]
    assert a.episode.solved == c.episode.solved


def test_adapter_save_reload_preserves_behavior(tmp_path: Path):
    task = _task()
    responses = ["2 0\n0 1", "2 0\n0 1"]
    save_scripted_adapter(responses, tmp_path)
    loaded = load_scripted_adapter(tmp_path, name="bond_scripted")
    res = run_direct(task, loaded, RunnerBudget(max_model_calls=2, max_seconds=5), system="bond_direct")
    assert res.episode.solved is True
    assert loaded.name == "Bond-smoke"
    assert loaded.is_bond is True


def test_four_way_eval_distinguishes_systems(tmp_path: Path):
    task = _task()
    budget = RunnerBudget(max_model_calls=8, max_seconds=5, seed=0)
    base = ScriptedModel(
        responses=["1 0\n0 2", "1 0\n0 2", _j("execute_program", program="rot90"), _j("commit_candidates", program="rot90")]
    )
    bond = ScriptedModel(
        responses=[
            "2 0\n0 1",
            "2 0\n0 1",
            _j("revise_hypothesis", text="rotate 180"),
            _j("inspect_objects"),
            _j("execute_program", program="rot180"),
            _j("commit_candidates", program="rot180"),
        ]
    )
    report = run_bond_eval(
        [task],
        base_model=base,
        bond_model=bond,
        budget=budget,
        out_dir=tmp_path,
        include_aabf_probe=False,
    )
    assert report["summaries"]["base_direct"]["solved"] == 0
    assert report["summaries"]["bond_direct"]["solved"] == 1
    assert report["summaries"]["bond_hrps"]["solved"] == 1
    assert report["deltas"]["delta_train_bond_direct_minus_base_direct"] >= 1
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "runs.csv").is_file()
    assert (tmp_path / "runs.jsonl").is_file()
    assert (tmp_path / "trajectories.jsonl").is_file()
    systems_logged = {json.loads(line)["system"] for line in (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    assert "base_direct" in systems_logged
    assert "bond_hrps" in systems_logged
    assert set(SYSTEMS) == {"base_direct", "base_hrps", "bond_direct", "bond_hrps"}
    assert report["is_final_bond"] is False

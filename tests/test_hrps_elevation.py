"""M0–M3 elevation harness: environment, gold-free feedback, no task patches."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.hrps.abstractions import Abstraction, AbstractionLibrary, load_library_json, set_active_library
from src.hrps.agent import ElevationBudget, run_episode, score_attempts
from src.hrps.dsl import Op, Program, replay
from src.hrps.env import (
    Action,
    HrpsEnv,
    gold_free_constraint_feedback,
    grid_to_compact,
    parse_model_actions,
    serialize_task_raw,
)
from src.hrps.kinds import Kind, kind_of
from src.hrps.model import LOCAL_DEFAULT, PREFERRED_INKLING, ScriptedModel, resolve_model_name
from src.hrps.task import parse_task

TRAIN = Path(__file__).resolve().parent.parent / "ARC-AGI-2" / "data" / "training"
H_JSON = Path(__file__).resolve().parent.parent / "artifacts" / "hrps_separability_h" / "abstractions.json"


def _rot180_task():
    return parse_task(
        "synth_rot180",
        {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
                {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]},
            ],
            "test": [
                {"input": [[1, 0], [0, 2]], "output": [[2, 0], [0, 1]]},
            ],
        },
        "training",
    )


def test_elevation_components_are_labeled():
    assert kind_of("hrps_environment") is Kind.EXACT
    assert kind_of("gold_free_constraint_feedback") is Kind.EXACT
    assert kind_of("open_model_reasoner") is Kind.LEARNED


def test_observe_hides_test_outputs():
    task = _rot180_task()
    text = serialize_task_raw(task)
    assert "TEST 0 INPUT" in text
    assert "TEST 0 OUTPUT" not in text
    assert "2 0" not in text  # test gold
    assert "1 0" in text  # test input


def test_parse_model_actions():
    acts = parse_model_actions("INSPECT colors\nAPPLY rot180\nCOMMIT rot180")
    assert [a.kind for a in acts] == ["inspect", "apply", "commit"]
    assert acts[1].payload == "rot180"
    acts2 = parse_model_actions('{"action": "apply", "program": "flip_h"}')
    assert acts2[0].kind == "apply" and acts2[0].payload == "flip_h"


def test_apply_rot180_is_jointly_exact_and_transfers():
    task = _rot180_task()
    env = HrpsEnv(task)
    fb = env.step(Action("apply", "rot180"))
    assert fb.accepted
    assert fb.data["residual"]["all_exact"] is True
    env.step(Action("commit", "rot180"))
    attempts = env.finalize_attempts()
    exact, pass2 = score_attempts(task, attempts)
    assert pass2 and exact == [True]


def test_unknown_op_is_rejected():
    env = HrpsEnv(_rot180_task())
    fb = env.step(Action("apply", "magic_solve"))
    assert not fb.accepted
    assert env.n_rejected == 1


@pytest.mark.skipif(not TRAIN.exists(), reason="ARC-AGI-2 training data missing")
def test_aabf363d_feedback_is_gold_free_and_exposes_underconstraint():
    from src.hrps.task import load_task_file

    task = load_task_file(TRAIN / "aabf363d.json", "training")
    env = HrpsEnv(task)
    prog = "apply_colormap:0-0;2-4;3-6;4-0;6-0"
    fb = env.step(Action("apply", prog))
    assert fb.accepted
    assert fb.data["residual"]["all_exact"] is True
    text = fb.text
    gold = task.test[0].output
    assert gold is not None
    assert grid_to_compact(gold) not in text
    assert "pred_equals_gt" not in text
    assert "gt_colors" not in text
    flags = fb.data["constraint"]["underconstraint_flags"]
    assert "joint_map_is_union_of_disjoint_per_demo_palettes" in flags
    assert "test_introduces_unseen_input_colors" in flags
    assert "color_plays_marker_role_in_one_pair_and_blob_role_in_another" in flags
    assert 8 in fb.data["constraint"]["test_unseen_input_colors"]
    assert 2 in fb.data["constraint"]["role_collision_colors"]
    assert fb.data["constraint"]["uses_test_labels"] is False
    from src.hrps.abstractions import parse_program_text

    info = gold_free_constraint_feedback(task, Program(parse_program_text(prog)))
    assert info["uses_test_labels"] is False
    assert "test_input_replay" in info
    assert "pred_equals_gt" not in str(info)


def test_m0_direct_baseline_fails_when_model_copies_input():
    task = _rot180_task()
    model = ScriptedModel(responses=["1 0\n0 2"])
    ep = run_episode(task, model, "M0", ElevationBudget(max_calls=2, max_seconds=5, max_tokens=64))
    assert ep.condition == "M0"
    assert ep.solved is False
    assert ep.pass2 is False
    assert ep.n_model_calls >= 1


def test_m1_oneshot_dsl_can_solve_expressible_task():
    task = _rot180_task()
    model = ScriptedModel(responses=["rot180"])
    ep = run_episode(task, model, "M1", ElevationBudget(max_calls=2, max_seconds=5, max_tokens=64))
    assert ep.joint_demo_exact
    assert ep.solved
    assert "rot180" in ep.programs[0]


def test_m2_active_loop_elevates_same_scripted_model():
    """Same scripted policy: copying grids fails M0; acting on residuals solves M2."""
    task = _rot180_task()
    budget = ElevationBudget(max_calls=8, max_seconds=5, max_tokens=64)
    m0 = run_episode(task, ScriptedModel(responses=["1 0\n0 2", "1 0\n0 2"]), "M0", budget)
    m2 = run_episode(
        task,
        ScriptedModel(
            responses=[
                "HYPOTHESIZE rotate the grid",
                "INSPECT shapes",
                "APPLY rot90",
                "HYPOTHESIZE residual remains; try 180",
                "APPLY rot180",
                "COMMIT rot180",
            ]
        ),
        "M2",
        budget,
    )
    assert m0.solved is False
    assert m2.solved is True
    assert m2.n_hypothesis_revisions >= 1
    assert m2.n_representation_requests >= 1
    assert m2.n_verifier_calls >= 2
    assert m2.n_contradiction_resolutions >= 1
    assert any(x > m2.residual_trace[-1] for x in m2.residual_trace[:-1])


def test_m3_abstraction_executes_when_enabled_and_rejected_otherwise():
    task = _rot180_task()
    abs_ = Abstraction(
        name="abs_rot180_via_90",
        body=(Op("rot90", ()), Op("rot90", ())),
        provenance=("synth",),
        cost=4,
    )
    lib = AbstractionLibrary((abs_,))
    set_active_library(lib)
    try:
        env2 = HrpsEnv(task, library=lib, enable_h=False)
        fb = env2.step(Action("apply", "abs:abs_rot180_via_90"))
        assert not fb.accepted
        env3 = HrpsEnv(task, library=lib, enable_h=True)
        fb = env3.step(Action("apply", "abs:abs_rot180_via_90"))
        assert fb.accepted
        assert fb.data["residual"]["all_exact"] is True
    finally:
        set_active_library(AbstractionLibrary())


def test_m3_episode_with_macro_matches_body_replay():
    task = _rot180_task()
    abs_ = Abstraction(
        name="abs_rot180_via_90",
        body=(Op("rot90", ()), Op("rot90", ())),
        provenance=("synth",),
        cost=4,
    )
    lib = AbstractionLibrary((abs_,))
    g = task.train[0].input
    set_active_library(lib)
    try:
        assert replay(Program((Op("abs", ("abs_rot180_via_90",)),)), g) == replay(Program(abs_.body), g)
        model = ScriptedModel(responses=["APPLY abs:abs_rot180_via_90", "COMMIT abs:abs_rot180_via_90"])
        ep = run_episode(
            task,
            model,
            "M3",
            ElevationBudget(max_calls=4, max_seconds=5, max_tokens=64),
            library=lib,
        )
        assert ep.solved
    finally:
        set_active_library(AbstractionLibrary())


def test_two_attempts_always_emitted():
    task = _rot180_task()
    env = HrpsEnv(task)
    env.step(Action("commit", "rot180"))
    attempts = env.finalize_attempts()
    assert len(attempts) == 2
    assert attempts[0] == attempts[1]


def test_source_has_no_task_id_patches():
    root = Path(__file__).resolve().parent.parent / "src" / "hrps"
    for name in ("env.py", "agent.py", "elevation.py", "model.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert '== "aabf363d"' not in text
        assert "== 'aabf363d'" not in text
        assert "if task.task_id" not in text or name != "env.py"


def test_local_default_is_qwen_not_inkling():
    assert LOCAL_DEFAULT.startswith("Qwen/")
    assert "Inkling" in PREFERRED_INKLING
    assert resolve_model_name() == LOCAL_DEFAULT


def test_verdict_primary_metric_is_m2_minus_m0():
    from src.hrps.elevation import elevation_verdict

    v = elevation_verdict(
        {
            "M0": {"n": 40, "solved": 0},
            "M1": {"n": 40, "solved": 2},
            "M2": {"n": 40, "solved": 5},
            "M3": {"n": 40, "solved": 5},
        }
    )
    assert v["elevated_vs_direct"] is True
    assert v["loop_beat_oneshot"] is True
    assert v["primary_delta_m2_minus_m0"] == 5
    v2 = elevation_verdict(
        {
            "M0": {"n": 1, "solved": 0},
            "M1": {"n": 1, "solved": 1},
            "M2": {"n": 1, "solved": 1},
            "M3": {"n": 1, "solved": 1},
        }
    )
    assert v2["elevated_vs_direct"] is True
    assert v2["loop_beat_oneshot"] is False


def test_open_model_loader_reports_no_torch():
    from src.hrps.model import try_load_open_model

    model, status = try_load_open_model()
    assert model is None
    assert status in {"no_torch", "no_transformers"} or status.startswith("load_failed")


@pytest.mark.skipif(not H_JSON.exists(), reason="H library artifact missing")
def test_h_library_json_loads_exact_macros():
    lib = load_library_json(H_JSON)
    assert len(lib) >= 1
    assert all("apply_colormap" not in a.body_serialize() for a in lib.items)
    assert all(len(a.body) >= 2 for a in lib.items)

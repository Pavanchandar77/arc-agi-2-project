"""Bond training data, hold-out safety, and four-way deltas."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.hrps.abstractions import Abstraction, AbstractionLibrary, set_active_library
from src.hrps.agent import ElevationBudget, run_episode
from src.hrps.bond import SYSTEMS, bond_deltas, bond_manifest, run_bond_eval
from src.hrps.dsl import Op
from src.hrps.episodes import (
    ACTION_SCHEMA,
    assert_training_safe,
    generate_from_trace_row,
    teacher_direct_episode,
    teacher_hrps_episode,
)
from src.hrps.env import grid_to_compact
from src.hrps.kinds import Kind, kind_of
from src.hrps.model import LOCAL_DEFAULT, PREFERRED_INKLING, ScriptedModel
from src.hrps.separability import held_out_training_ids
from src.hrps.task import parse_task

TRAIN = Path(__file__).resolve().parent.parent / "ARC-AGI-2" / "data" / "training"


def _rot180_task():
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


def test_bond_components_are_labeled():
    assert kind_of("bond_episode") is Kind.EXACT
    assert kind_of("bond_adapter") is Kind.LEARNED
    assert kind_of("bond_inference_controller") is Kind.EXACT
    assert "commit" in ACTION_SCHEMA["actions"]


def test_teacher_success_trajectory_solves_via_env():
    task = _rot180_task()
    ep = teacher_hrps_episode(task, "rot180", test_transfer=True, kind_hint="success_trajectory")
    assert ep is not None
    assert ep.split == "training"
    assert ep.held_out is False
    assert ep.joint_demo_exact
    assert any(t.assistant.startswith("COMMIT") for t in ep.turns)
    assert any("APPLY rot90" in t.assistant for t in ep.turns)
    assert any("APPLY rot180" in t.assistant for t in ep.turns)
    msgs = ep.to_sft_messages()
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "assistant"
    assert "TEST 0 OUTPUT" not in ep.observe
    assert "2 0" not in ep.observe


def test_direct_answer_uses_training_label_only_as_target():
    task = _rot180_task()
    ep = teacher_direct_episode(task, "rot180", True)
    assert ep is not None
    assert ep.kind == "direct_answer"
    assert ep.turns[0].assistant == "2 0\n0 1"
    assert "2 0" not in ep.observe  # gold not in the user observation
    assert "TEST 0 OUTPUT" not in ep.observe


def test_held_out_rows_are_dropped():
    held = set(held_out_training_ids())
    # 73182012 is a G-positive inside the diagnostic slice.
    assert "73182012" in held
    task = _rot180_task()
    row = {
        "task_id": "73182012",
        "split": "training",
        "solved": True,
        "programs": ["rot180"],
        "test_exact": [True],
        "telemetry": {"joint_verified": True},
    }
    # The task object id is synth; the row id is held-out. Filter is on task.task_id
    # in generate_from_trace_row — use a parsed task with the held-out id.
    held_task = parse_task(
        "73182012",
        {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
            "test": [{"input": [[1, 0], [0, 2]], "output": [[2, 0], [0, 1]]}],
        },
        "training",
    )
    out = generate_from_trace_row(row, held_task, held_out_ids=held)
    assert out == []
    with pytest.raises(ValueError, match="held-out"):
        fake = teacher_hrps_episode(task, "rot180", test_transfer=True, kind_hint="success_trajectory")
        fake.task_id = "73182012"  # type: ignore[misc]
        fake.held_out = False
        assert_training_safe([fake], held)


def test_evaluation_split_rejected():
    task = parse_task(
        "synth_eval",
        {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
            "test": [{"input": [[1, 0], [0, 2]], "output": [[2, 0], [0, 1]]}],
        },
        "evaluation",
    )
    row = {
        "task_id": "synth_eval",
        "split": "evaluation",
        "solved": True,
        "programs": ["rot180"],
        "test_exact": [True],
        "telemetry": {"joint_verified": True},
    }
    assert generate_from_trace_row(row, task, held_out_ids=set()) == []


@pytest.mark.skipif(not TRAIN.exists(), reason="ARC-AGI-2 training data missing")
def test_aabf363d_underconstraint_episode_has_no_test_gold():
    from src.hrps.task import load_task_file

    task = load_task_file(TRAIN / "aabf363d.json", "training")
    held = set(held_out_training_ids())
    assert task.task_id not in held
    row = {
        "task_id": "aabf363d",
        "split": "training",
        "solved": False,
        "programs": ["apply_colormap:0-0;2-4;3-6;4-0;6-0"],
        "test_exact": [False],
        "telemetry": {"joint_verified": True},
        "failure_category": "consistency",
    }
    eps = generate_from_trace_row(row, task, held_out_ids=held)
    assert eps
    ep = next(e for e in eps if e.kind != "direct_answer")
    assert ep.kind == "underconstraint"
    assert ep.joint_demo_exact
    assert ep.test_transfer is False
    gold = grid_to_compact(task.test[0].output)  # type: ignore[arg-type]
    blob = ep.observe + "".join(t.feedback for t in ep.turns)
    assert gold not in blob
    assert "pred_equals_gt" not in blob
    assert any("underconstrained" in t.assistant.lower() for t in ep.turns)
    assert not any(t.assistant.startswith("COMMIT") for t in ep.turns)


def test_no_task_id_patches_in_bond_sources():
    root = Path(__file__).resolve().parent.parent / "src" / "hrps"
    for name in ("episodes.py", "bond.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert '== "aabf363d"' not in text
        assert "== 'aabf363d'" not in text


def test_bond_deltas_distinguish_learned_and_substrate():
    strong = bond_deltas(
        {
            "base_direct": {"solved": 0},
            "base_hrps": {"solved": 1},
            "bond_direct": {"solved": 2},
            "bond_hrps": {"solved": 4},
        }
    )
    assert strong["claim"] == "learned_and_substrate_gains"
    assert strong["delta_train_bond_direct_minus_base_direct"] == 2
    assert strong["delta_substrate_bond_hrps_minus_bond_direct"] == 2
    substrate_only = bond_deltas(
        {
            "base_direct": {"solved": 0},
            "base_hrps": {"solved": 2},
            "bond_direct": {"solved": 0},
            "bond_hrps": {"solved": 3},
        }
    )
    assert substrate_only["claim"] == "substrate_gain_without_learned_direct_gain"


def test_four_way_eval_scripted_bond_vs_base():
    task = _rot180_task()
    budget = ElevationBudget(max_calls=8, max_seconds=5, max_tokens=64)
    base = ScriptedModel(responses=["1 0\n0 2", "1 0\n0 2", "APPLY rot90", "COMMIT rot90"])
    bond = ScriptedModel(
        responses=[
            "2 0\n0 1",
            "2 0\n0 1",
            "HYPOTHESIZE rotate 180",
            "INSPECT shapes",
            "APPLY rot180",
            "COMMIT rot180",
        ]
    )
    report = run_bond_eval([task], base_model=base, bond_model=bond, budget=budget, out_dir=Path("artifacts/bond/eval_smoke"))
    d = report["deltas"]
    assert report["summaries"]["base_direct"]["solved"] == 0
    assert report["summaries"]["bond_direct"]["solved"] == 1
    assert report["summaries"]["bond_hrps"]["solved"] == 1
    assert d["delta_train_bond_direct_minus_base_direct"] >= 1
    assert set(SYSTEMS) == {"base_direct", "base_hrps", "bond_direct", "bond_hrps"}


def test_manifest_records_foundation_and_holdout():
    held = held_out_training_ids()[:3]
    man = bond_manifest(
        model_name=LOCAL_DEFAULT,
        adapter_dir=None,
        train_config={"lora_r": 16},
        episode_summary={"n_episodes": 0},
        held_out_ids=held,
        status="episodes_ready",
        notes=["test"],
    )
    assert man["foundation"]["preferred_foundation"] == PREFERRED_INKLING
    assert man["foundation"]["local_foundation"] == LOCAL_DEFAULT
    assert man["data_provenance"]["public_evaluation_used"] is False
    assert man["data_provenance"]["held_out_excluded"] == held
    assert man["schemas"]["executor"] == "src.hrps.dsl.replay"


def test_abstraction_episode_requires_h_enabled():
    task = _rot180_task()
    abs_ = Abstraction("abs_rot180_via_90", (Op("rot90", ()), Op("rot90", ())), ("synth",), 4)
    lib = AbstractionLibrary((abs_,))
    set_active_library(AbstractionLibrary())
    try:
        ep = teacher_hrps_episode(
            task,
            "abs:abs_rot180_via_90",
            test_transfer=True,
            kind_hint="abstraction",
            library=lib,
            enable_h=True,
            include_competing=False,
        )
        assert ep is not None
        assert ep.joint_demo_exact
        assert any("abs:abs_rot180_via_90" in t.assistant for t in ep.turns)
    finally:
        set_active_library(AbstractionLibrary())

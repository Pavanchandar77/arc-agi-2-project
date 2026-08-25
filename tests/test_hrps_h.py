"""H: training-only exact abstractions. No task-specific patches."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.hrps.abstractions import (
    Abstraction,
    AbstractionLibrary,
    mine_abstractions_from_rows,
    parse_program_text,
    set_active_library,
    transfer_report,
)
from src.hrps.dsl import Op, Program, execute_op, replay, stage_config
from src.hrps.kinds import Kind, kind_of
from src.hrps.search import SearchBudget, search_task
from src.hrps.separability import characterize_colormap_constraint, held_out_training_ids
from src.hrps.task import load_task_file

TRAIN = Path(__file__).resolve().parent.parent / "ARC-AGI-2" / "data" / "training"
G_POSITIVES = ("6f8cd79b", "73182012", "7468f01a", "74dd1130")


def test_abstraction_execution_is_exact_replay():
    body = (Op("rot90", ()), Op("rot90", ()))
    abs_ = Abstraction(name="abs_rot180_via_90", body=body, provenance=("synth",), cost=4)
    set_active_library(AbstractionLibrary((abs_,)))
    g = ((1, 2), (3, 4))
    via_abs = execute_op(Op("abs", ("abs_rot180_via_90",)), g)
    via_body = replay(Program(body), g)
    assert via_abs == via_body
    assert via_abs == ((4, 3), (2, 1))
    set_active_library(AbstractionLibrary())


def test_mining_excludes_held_out_and_colormap_and_unsolved():
    held = {"held1", "held2"}
    rows = [
        {
            "task_id": "src1",
            "solved": True,
            "programs": ["crop_fg:0 | left_half | top_half"],
        },
        {
            "task_id": "held1",
            "solved": True,
            "programs": ["crop_fg:0 | flip_h"],
        },
        {
            "task_id": "fail1",
            "solved": False,
            "programs": ["apply_colormap:0-0;2-4"],
        },
        {
            "task_id": "src2",
            "solved": True,
            "programs": ["left_half | top_half | rot270"],
        },
        {
            "task_id": "src3",
            "solved": True,
            "programs": ["apply_colormap:1-2;3-4"],
        },
        {
            "task_id": "src4",
            "solved": True,
            "programs": ["rot180"],
        },
    ]
    lib = mine_abstractions_from_rows(rows, exclude_task_ids=held)
    names_bodies = {a.body_serialize(): a.provenance for a in lib.items}
    assert "crop_fg:0 | flip_h" not in names_bodies
    assert "crop_fg:0 | left_half | top_half" in names_bodies
    assert "left_half | top_half" in names_bodies
    assert names_bodies["left_half | top_half"] == ("src1", "src2")
    assert all("apply_colormap" not in a.body_serialize() for a in lib.items)
    assert all(len(a.body) >= 2 for a in lib.items)


def test_abstraction_kinds():
    assert kind_of("abstraction_library") is Kind.LEARNED
    assert kind_of("abstraction_execution") is Kind.EXACT


@pytest.mark.skipif(not TRAIN.exists(), reason="ARC-AGI-2 training data missing")
def test_aabf363d_remains_documented_colormap_failure():
    task = load_task_file(TRAIN / "aabf363d.json", "training")
    prog = "apply_colormap:0-0;2-4;3-6;4-0;6-0"
    info = characterize_colormap_constraint(task, prog)
    assert info["test_replay"][0]["pred_equals_gt"] is False
    label = (
        "Jointly demonstration-consistent but test-underdetermined "
        "under the current global-colormap representation."
    )
    assert "joint_map_is_union_of_disjoint_per_demo_palettes" in info["constraint_failure"]
    assert label.startswith("Jointly demonstration-consistent")


@pytest.mark.skipif(not TRAIN.exists(), reason="ARC-AGI-2 training data missing")
@pytest.mark.parametrize("task_id", G_POSITIVES)
def test_four_g_positives_still_solve_without_h(task_id):
    task = load_task_file(TRAIN / f"{task_id}.json", "training")
    set_active_library(AbstractionLibrary())
    res = search_task(task, stage="G", budget=SearchBudget(max_depth=3, max_nodes=200, max_seconds=1.5, max_ops_per_node=36))
    assert res.solved, (task_id, res.programs, res.failure_category)


@pytest.mark.skipif(not TRAIN.exists(), reason="ARC-AGI-2 training data missing")
def test_held_out_ids_match_slice():
    ids = held_out_training_ids(400, 40)
    assert "73182012" in ids
    assert "6f8cd79b" in ids
    assert "aabf363d" not in ids

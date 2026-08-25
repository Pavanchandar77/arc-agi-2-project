"""Separability experiment: classification, held-out slice, colormap transfer."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.hrps.conclusions import FROZEN_CONCLUSIONS, NEXT_CHANGE_POLICY, frozen_payload
from src.hrps.separability import (
    FROZEN_DEPTH,
    FROZEN_OPS_PER_NODE,
    analyze_joint_demo_row,
    characterize_colormap_constraint,
    classify_budget_trace,
    compare_depth_reports,
    held_out_training_ids,
    language_ceiling,
    make_budget_ladder,
)
from src.hrps.task import load_task_file

TRAIN = Path(__file__).resolve().parent.parent / "ARC-AGI-2" / "data" / "training"


def _rec(budget_id: str, solved: bool, exhausted: bool = False) -> dict:
    return {
        "budget_id": budget_id,
        "solved": solved,
        "telemetry": {"enumerated_exhausted": exhausted},
    }


def test_frozen_conclusions_are_the_five_locked_statements():
    ids = [c["id"] for c in FROZEN_CONCLUSIONS]
    assert ids == [
        "G_is_useful_search_control",
        "F_has_not_reduced_search",
        "bounded_language_is_shallow",
        "joint_consistency_not_sufficient",
        "colormap_misses_role_relative_semantics",
    ]
    assert "Qwen" in " ".join(NEXT_CHANGE_POLICY["forbidden_until_then"])
    assert any("abstractions" in a for a in NEXT_CHANGE_POLICY["allowed"])
    assert any("A/F/G" in r or "A/F/G" in str(NEXT_CHANGE_POLICY["required_after_change"]) for r in NEXT_CHANGE_POLICY["required_after_change"])
    payload = frozen_payload()
    assert payload["status"] == "frozen"


def test_classify_solved_at_low_budget():
    assert (
        classify_budget_trace([_rec("B0", True)])
        == "solved_at_low_budget"
    )


def test_classify_solved_only_at_higher_budget():
    trace = [_rec("B0", False), _rec("B1", False), _rec("B2", True)]
    assert classify_budget_trace(trace) == "solved_only_at_higher_budget"


def test_classify_not_expressible_when_exhausted():
    trace = [_rec("B0", False), _rec("B1", False, exhausted=True)]
    assert classify_budget_trace(trace) == "not_expressible"


def test_classify_still_truncated():
    trace = [_rec("B0", False), _rec("B1", False), _rec("B3", False, exhausted=False)]
    assert classify_budget_trace(trace) == "unsolved_search_still_truncated"


def test_language_ceiling_both():
    from collections import Counter

    c = Counter(
        {
            "solved_at_low_budget": 2,
            "solved_only_at_higher_budget": 1,
            "not_expressible": 10,
            "unsolved_search_still_truncated": 7,
        }
    )
    ceil = language_ceiling(c)
    assert ceil["solved_any_budget"] == 3
    assert ceil["accuracy_lower_bound"] == round(3 / 20, 6)
    assert ceil["accuracy_upper_bound"] == round(10 / 20, 6)
    assert ceil["verdict"] == "both_truncation_and_expressiveness"


@pytest.mark.skipif(not TRAIN.exists(), reason="ARC-AGI-2 training data missing")
def test_d4_ladder_changes_only_max_depth():
    d3 = make_budget_ladder(3)
    d4 = make_budget_ladder(4)
    assert [b.max_depth for _, b in d3] == [3, 3, 3, 3]
    assert [b.max_depth for _, b in d4] == [4, 4, 4, 4]
    for (_, a), (_, b) in zip(d3, d4):
        assert a.max_nodes == b.max_nodes
        assert a.max_seconds == b.max_seconds
        assert a.max_frontier == b.max_frontier
        assert a.max_ops_per_node == b.max_ops_per_node == FROZEN_OPS_PER_NODE
    assert FROZEN_DEPTH == 3


def test_compare_depth_reports_detects_new_solves():
    def _rep(classes: dict[str, str]) -> dict:
        from collections import Counter

        counts = Counter(classes.values())
        return {
            "protocol": {"frozen": {"depth": 3}, "task_ids": sorted(classes)},
            "stages": {
                "G": {
                    "class_counts": dict(counts),
                    "class_by_task": classes,
                    "language_ceiling": language_ceiling(counts),
                }
            },
        }

    base = _rep({"t1": "solved_at_low_budget", "t2": "not_expressible"})
    raised = _rep({"t1": "solved_at_low_budget", "t2": "solved_only_at_higher_budget"})
    raised["protocol"]["frozen"]["depth"] = 4
    diff = compare_depth_reports(base, raised)
    assert diff["stages"]["G"]["newly_solved"] == ["t2"]
    assert diff["stages"]["G"]["n_lost_solved"] == 0


def test_held_out_slice_skips_phase1_prefix():
    ids = held_out_training_ids(offset=400, n=40)
    assert len(ids) == 40
    prefix = held_out_training_ids(offset=0, n=60)
    assert set(ids).isdisjoint(set(prefix))
    assert ids == sorted(ids)


@pytest.mark.skipif(not TRAIN.exists(), reason="ARC-AGI-2 training data missing")
def test_aabf363d_colormap_is_disjoint_palette_union():
    task = load_task_file(TRAIN / "aabf363d.json", "training")
    prog = "apply_colormap:0-0;2-4;3-6;4-0;6-0"
    info = characterize_colormap_constraint(task, prog)
    assert info["disjoint_per_demo_color_support"] is True
    assert 8 in info["test_unseen_input_colors"]
    assert 2 in info["role_collision_colors"]
    assert info["test_replay"][0]["pred_equals_gt"] is False
    assert "joint_map_is_union_of_disjoint_per_demo_palettes" in info["constraint_failure"]
    assert "test_introduces_unseen_input_colors" in info["constraint_failure"]
    assert "color_plays_marker_role_in_one_pair_and_blob_role_in_another" in info["constraint_failure"]


@pytest.mark.skipif(not TRAIN.exists(), reason="ARC-AGI-2 training data missing")
def test_joint_demo_row_flags_failed_transfer():
    task = load_task_file(TRAIN / "aabf363d.json", "training")
    row = {
        "task_id": "aabf363d",
        "solved": False,
        "failure_category": "consistency",
        "programs": ["apply_colormap:0-0;2-4;3-6;4-0;6-0"],
        "telemetry": {
            "time_to_first_exact_demonstration_solution": 0.02,
            "nodes_expanded": 17,
            "description_length": 8,
        },
    }
    rec = analyze_joint_demo_row(row, task)
    assert rec["solved"] is False
    assert rec["family"] == "apply_colormap"
    assert rec["colormap"]["disjoint_per_demo_color_support"] is True

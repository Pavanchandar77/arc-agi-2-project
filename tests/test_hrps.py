"""HRPS Phase-1 tests: executor, joint residual, signatures, replay, search."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.hrps.dsl import Op, Program, execute_op, infer_colormap, operator_catalog, replay, stage_config
from src.hrps.failure import classify_failure
from src.hrps.grid import (
    as_grid,
    crop_fg,
    fill_holes,
    flip_h,
    grids_equal,
    rot90,
    rot180,
    shape,
    tile,
    upscale,
)
from src.hrps.kinds import Kind, kind_of
from src.hrps.representation import SPEC_4_ZERO, SPEC_8A_ZERO, extract_objects
from src.hrps.residual import joint_residual
from src.hrps.search import SearchBudget, search_task
from src.hrps.signature import continuation_signature
from src.hrps.task import parse_task

TRAIN = Path(__file__).resolve().parent.parent / "ARC-AGI-2" / "data" / "training"


def G(*rows):
    return as_grid(rows)


def test_component_labels_are_complete_for_shipped_mechanisms():
    assert kind_of("executor") is Kind.EXACT
    assert kind_of("continuation_signature") is Kind.SOUND_INCOMPLETE
    assert kind_of("residual_operator_prioritization") is Kind.HEURISTIC
    assert kind_of("admissible_remaining_cost_bound") is Kind.EXACT
    with pytest.raises(KeyError):
        kind_of("mystery_widget")


def test_geometry_roundtrip():
    g = G([1, 2], [3, 4])
    assert rot180(rot180(g)) == g
    assert flip_h(flip_h(g)) == g
    assert shape(rot90(g)) == (2, 2)


def test_crop_tile_upscale_fill():
    g = G([0, 0, 0], [0, 5, 0], [0, 0, 0])
    assert crop_fg(g, 0) == G([5])
    t = tile(G([1, 2]), 2, 2)
    assert t == G([1, 2, 1, 2], [1, 2, 1, 2])
    u = upscale(G([7]), 3)
    assert u == G([7, 7, 7], [7, 7, 7], [7, 7, 7])
    hole = G([1, 1, 1], [1, 0, 1], [1, 1, 1])
    assert fill_holes(hole, 0) == G([1, 1, 1], [1, 1, 1], [1, 1, 1])


def test_executor_and_replay_are_exact():
    g = G([1, 2], [3, 4])
    p = Program((Op("rot90", ()), Op("flip_h", ())))
    out = replay(p, g)
    assert out == flip_h(rot90(g))
    assert execute_op(Op("recolor", (1, 9)), g) == G([9, 2], [3, 4])
    assert p.serialize() == "rot90 | flip_h"
    assert Op.deserialize("recolor:1,9") == Op("recolor", (1, 9))
    assert Op.deserialize("isolate_largest:4,t,0") == Op("isolate_largest", (4, True, 0))


def test_invalid_ops_return_none_not_crash():
    g = G([1, 2, 3])
    assert execute_op(Op("left_half", ()), G([1])) is None
    huge = execute_op(Op("tile", ((3, 3),)), as_grid([[1] * 12] * 12))
    assert huge is None


def test_connected_components_4_vs_8_vs_agnostic():
    g = G(
        [1, 0, 1],
        [0, 1, 0],
        [2, 2, 0],
    )
    c4 = extract_objects(g, 4, False, 0)
    c8 = extract_objects(g, 8, False, 0)
    a8 = extract_objects(g, 8, True, 0)
    assert len(c4) == 4  # three 1s isolated + one 2-bar
    assert len(c8) == 2  # diagonal 1s merge, 2-bar
    assert len(a8) == 1  # all non-bg 8-connected through the center 1


def test_joint_residual_zero_iff_all_demos_match():
    a = G([1, 2])
    b = G([2, 1])
    jr = joint_residual((a, a), (a, a), SPEC_4_ZERO)
    assert jr.all_exact
    assert jr.pixel_total == 0
    jr2 = joint_residual((a, b), (a, a), SPEC_4_ZERO)
    assert not jr2.all_exact
    assert jr2.n_exact == 1
    assert jr2.pixel_total > 0


def test_colormap_generator_is_exact_and_rejects_nonfunctions():
    preds = (G([1, 2]), G([1, 2]))
    gts = (G([3, 4]), G([3, 4]))
    cmap = infer_colormap(preds, gts)
    assert cmap == ((1, 3), (2, 4))
    bad = infer_colormap((G([1, 1]),), (G([2, 3]),))
    assert bad is None


def test_signature_no_false_merge():
    g1 = G([1, 2], [3, 4])
    g2 = rot90(g1)
    family = "flip_h,rot90"
    s_same_a = continuation_signature((g1, g1), "G", family)
    s_same_b = continuation_signature((g1, g1), "G", family)
    s_diff = continuation_signature((g2, g1), "G", family)
    s_fam = continuation_signature((g1, g1), "G", "rot90")
    assert s_same_a == s_same_b
    assert s_same_a != s_diff
    assert s_same_a != s_fam


def test_search_finds_rot180_on_synthetic_task():
    task = parse_task(
        "synth_rot180",
        {
            "train": [
                {"input": [[1, 2], [3, 0]], "output": [[0, 3], [2, 1]]},
                {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]},
            ],
            "test": [{"input": [[9, 1], [2, 3]], "output": [[3, 2], [1, 9]]}],
        },
        "training",
    )
    res = search_task(task, stage="A", budget=SearchBudget(max_depth=2, max_nodes=80, max_seconds=2.0))
    assert res.solved
    tokens = res.programs[0].split(" | ")
    found = Program(tuple(Op.deserialize(tok) for tok in tokens if tok != "identity"))
    assert replay(found, task.test[0].input) == task.test[0].output


def test_search_finds_recolor_via_colormap():
    task = parse_task(
        "synth_recolor",
        {
            "train": [
                {"input": [[1, 1], [2, 0]], "output": [[3, 3], [4, 0]]},
                {"input": [[2, 1], [1, 0]], "output": [[4, 3], [3, 0]]},
            ],
            "test": [{"input": [[1, 2], [0, 1]], "output": [[3, 4], [0, 3]]}],
        },
        "training",
    )
    res = search_task(task, stage="A", budget=SearchBudget(max_depth=2, max_nodes=120, max_seconds=2.0))
    assert res.solved
    assert res.telemetry["nodes_expanded"] >= 1


def test_joint_consistency_rejects_program_that_fits_only_one_demo():
    # Shared palette so a per-demo colormap cannot be jointly consistent.
    # rot90 fits demo 1 only; demo 2 is identity.
    task = parse_task(
        "synth_inconsistent",
        {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            ],
            "test": [{"input": [[4, 3], [2, 1]], "output": [[4, 3], [2, 1]]}],
        },
        "training",
    )
    res = search_task(task, stage="A", budget=SearchBudget(max_depth=2, max_nodes=80, max_seconds=2.0))
    assert not res.solved
    assert res.failure_category in {"consistency", "DSL_expressiveness", "search_explosion", "timeout"}


def test_stage_flags_change_operator_families():
    a = stage_config("A")
    b = stage_config("B")
    c = stage_config("C")
    g = stage_config("G")
    assert a.object_specs == ()
    assert b.object_specs[0].connectivity == 4
    assert c.object_specs[0].color_agnostic is True
    assert g.residual_factorization and g.continuation_dedup
    assert not a.continuation_dedup


def test_operator_catalog_is_typed_and_finite():
    cat = operator_catalog()
    names = {r["name"] for r in cat}
    assert "rot90" in names
    assert "apply_colormap" in names
    assert "isolate_largest" in names
    for row in cat:
        assert row["kind"] == "exact"
        assert row["out_type"] == "Grid"
        assert row["cost"] >= 2


def test_failure_taxonomy_values():
    assert classify_failure(
        solved=True,
        timed_out=False,
        hit_node_limit=False,
        hit_frontier_limit=False,
        enumerated_exhausted=False,
        n_ops_generated=10,
        n_executor_rejects=0,
        n_partial_consistency=0,
        n_exact_demos_best=3,
        n_demos=3,
        representation_instability=0.0,
        used_object_ops=False,
    ) == "solved"
    assert classify_failure(
        solved=False,
        timed_out=True,
        hit_node_limit=False,
        hit_frontier_limit=False,
        enumerated_exhausted=False,
        n_ops_generated=10,
        n_executor_rejects=0,
        n_partial_consistency=0,
        n_exact_demos_best=0,
        n_demos=3,
        representation_instability=0.0,
        used_object_ops=False,
    ) == "timeout"


@pytest.mark.skipif(not TRAIN.exists(), reason="ARC-AGI-2 training data missing")
def test_real_rot180_task_3c9b0459():
    import json

    payload = json.loads((TRAIN / "3c9b0459.json").read_text(encoding="utf-8"))
    task = parse_task("3c9b0459", payload, "training")
    res = search_task(task, stage="A", budget=SearchBudget(max_depth=2, max_nodes=60, max_seconds=2.0))
    assert res.solved
    from src.hrps.dsl import Program as P
    from src.hrps.dsl import Op as O

    prog = None
    for text in res.programs:
        if text == "rot180":
            prog = P((O("rot180", ()),))
            break
        if text == "flip_v | flip_h" or text == "flip_h | flip_v":
            names = text.split(" | ")
            prog = P(tuple(O(n, ()) for n in names))
            break
    assert prog is not None
    for pair in task.train:
        assert replay(prog, pair.input) == pair.output
    assert replay(prog, task.test[0].input) == task.test[0].output


@pytest.mark.skipif(not TRAIN.exists(), reason="ARC-AGI-2 training data missing")
def test_search_does_not_hardcode_task_ids(tmp_path):
    # The solver path has no per-id branches; running two different tasks is enough.
    import json

    from src.hrps.search import search_task as st

    src = Path(__file__).resolve().parent.parent / "src" / "hrps"
    text = ""
    for p in src.glob("*.py"):
        text += p.read_text(encoding="utf-8")
    assert "3c9b0459" not in text
    assert "00576224" not in text
    payload = json.loads((TRAIN / "3c9b0459.json").read_text(encoding="utf-8"))
    task = parse_task("3c9b0459", payload, "training")
    res = st(task, stage="G", budget=SearchBudget(max_depth=2, max_nodes=40, max_seconds=1.5))
    assert "task_id" in res.__dict__
    assert res.telemetry["nodes_expanded"] >= 1

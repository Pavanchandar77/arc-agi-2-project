"""Bond-L1 language, role ops, guided search, engine, curriculum."""

from __future__ import annotations

from pathlib import Path

from src.hrps.bond_engine import run_bond_engine
from src.hrps.bond_reward import verifier_reward
from src.hrps.dsl import Op, Program, generate_ops, replay, stage_config
from src.hrps.guided_search import guided_search
from src.hrps.kinds import Kind, kind_of
from src.hrps.language import LANGUAGE_DEPTH, LANGUAGE_ID, bond_l1_budget, language_manifest
from src.hrps.role_ops import (
    erase_smallest,
    recolor_nonsingleton_to_singleton_color,
)
from src.hrps.search import SearchBudget, search_task
from src.hrps.synthesize import synthesize_batch, synthesize_task
from src.hrps.task import parse_task


def _rot180_task():
    return parse_task(
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


def _marker_task():
    """Singleton marker + blob. Not a named official task."""
    return parse_task(
        "synth_marker_role",
        {
            "train": [
                {
                    "input": [[9, 0, 0], [0, 2, 2], [0, 2, 2]],
                    "output": [[0, 0, 0], [0, 9, 9], [0, 9, 9]],
                },
                {
                    "input": [[0, 0, 4], [3, 3, 0], [3, 3, 0]],
                    "output": [[0, 0, 0], [4, 4, 0], [4, 4, 0]],
                },
            ],
            "test": [
                {
                    "input": [[8, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                    "output": [[0, 0, 0, 0], [0, 8, 8, 0], [0, 8, 8, 0], [0, 0, 0, 0]],
                }
            ],
        },
        "training",
    )


def test_bond_l1_components_labeled():
    assert kind_of("bond_l1_language") is Kind.EXACT
    assert kind_of("bond_role_ops") is Kind.EXACT
    assert kind_of("bond_guided_search") is Kind.HEURISTIC
    assert kind_of("bond_test_time_compute") is Kind.HEURISTIC
    assert kind_of("bond_synthesizer") is Kind.EXACT
    assert kind_of("bond_verifier_reward") is Kind.EXACT
    assert kind_of("bond_engine") is Kind.EXACT
    assert kind_of("bond_curriculum") is Kind.LEARNED


def test_phase1_stage_g_does_not_emit_bond_role_ops():
    task = _rot180_task()
    cfg = stage_config("G")
    assert cfg.bond_language is False
    ops = generate_ops(task, task.train_inputs(), task.train_outputs(), cfg, "pixel")
    names = {op.name for op in ops}
    assert "rot180" in names
    assert "recolor_nonsingleton_to_singleton_color" not in names


def test_stage_l_emits_bond_role_ops():
    task = _marker_task()
    cfg = stage_config("L")
    assert cfg.bond_language is True
    ops = generate_ops(task, task.train_inputs(), task.train_outputs(), cfg, "object")
    names = {op.name for op in ops}
    assert "recolor_nonsingleton_to_singleton_color" in names
    assert "erase_smallest" in names
    assert "rot180" in names


def test_marker_role_ops_are_exact_and_general():
    grid = (
        (9, 0, 0),
        (0, 2, 2),
        (0, 2, 2),
    )
    mid = recolor_nonsingleton_to_singleton_color(grid, 4, False, 0)
    assert mid == ((9, 0, 0), (0, 9, 9), (0, 9, 9))
    out = erase_smallest(mid, 4, False, 0)
    assert out == ((0, 0, 0), (0, 9, 9), (0, 9, 9))
    prog = Program(
        (
            Op("recolor_nonsingleton_to_singleton_color", (4, False, 0)),
            Op("erase_smallest", (4, False, 0)),
        )
    )
    assert replay(prog, grid) == out


def test_ties_refuse_rather_than_guess():
    grid = ((1, 2), (3, 4))
    assert recolor_nonsingleton_to_singleton_color(grid, 4, False, 0) is None
    assert erase_smallest(grid, 4, False, 0) is None


def test_g_search_still_solves_rot180():
    task = _rot180_task()
    res = search_task(task, stage="G", budget=SearchBudget(max_depth=3, max_nodes=80, max_seconds=2, max_ops_per_node=36))
    assert res.solved is True


def test_l_search_solves_marker_task_g_does_not():
    task = _marker_task()
    g = search_task(task, stage="G", budget=SearchBudget(max_depth=3, max_nodes=200, max_seconds=2, max_ops_per_node=36))
    assert g.solved is False
    l = search_task(task, stage="L", budget=bond_l1_budget(nodes=400, seconds=3.0, frontier=4000))
    assert l.solved is True
    assert any("recolor_nonsingleton" in p or "erase_smallest" in p for p in l.programs)


def test_reward_is_gold_free():
    task = _rot180_task()
    rec = verifier_reward(task, Program((Op("rot180", ()),)))
    assert rec["joint_demo_exact"] is True
    assert rec["uses_test_labels"] is False
    assert rec["reward"] > 0
    bad = verifier_reward(task, Program((Op("rot90", ()),)))
    assert bad["joint_demo_exact"] is False
    assert bad["reward"] == 0.0


def test_engine_solves_rot180_and_marker_without_model():
    rot = run_bond_engine(_rot180_task(), None, run_overseer_loop=False, search_budget=bond_l1_budget(nodes=200, seconds=2))
    assert rot.episode.solved is True
    assert rot.episode.pass2 is True
    mark = run_bond_engine(_marker_task(), None, run_overseer_loop=False, search_budget=bond_l1_budget(nodes=400, seconds=3))
    assert mark.episode.solved is True
    assert mark.episode.pass2 is True


def test_synthesizer_is_jointly_exact_by_construction():
    rec = synthesize_task(__import__("random").Random(0), kind="geom")
    assert rec is not None
    task, program = rec
    from src.hrps.residual import joint_residual

    preds = tuple(replay(program, p.input) for p in task.train)
    jr = joint_residual(preds, task.train_outputs(), spec=None)
    assert jr.all_exact is True
    batch = synthesize_batch(8, seed=1)
    assert len(batch) == 8
    assert all(t.task_id.startswith("synth_") for t, _ in batch)


def test_guided_search_language_id():
    g = guided_search(_rot180_task(), budget=bond_l1_budget(nodes=120, seconds=2))
    assert g.language == LANGUAGE_ID
    assert g.joint_demo_exact is True
    man = language_manifest()
    assert man["max_depth"] == LANGUAGE_DEPTH
    assert man["foundation_hf_id"] == "Qwen/Qwen3.5-4B"
    assert man["phase1_untouched"]["depth"] == 3


def test_curriculum_holdout_clean(tmp_path: Path):
    from src.hrps.bond_curriculum import build_curriculum
    from src.hrps.separability import held_out_training_ids

    summary = build_curriculum(6, seed=2, out_dir=tmp_path, search_verify=False)
    assert summary["n_episodes"] >= 1
    held = set(held_out_training_ids())
    assert not (set(summary["task_ids"]) & held)
    assert (tmp_path / "sft_actions.jsonl").is_file()

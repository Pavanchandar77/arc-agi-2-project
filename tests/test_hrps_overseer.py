"""Bond overseer, memory, tools, and 27B no-download gate."""

from __future__ import annotations

import json

from src.hrps.backend import hardware_gate, resolve_foundation
from src.hrps.bond_eval import named_gains
from src.hrps.bond_manifest import MODULE_MAP, PUBLIC_NAME
from src.hrps.bond_memory import BondMemory
from src.hrps.bond_overseer import run_overseer
from src.hrps.bond_tools import dispatch_tool
from src.hrps.env import HrpsEnv
from src.hrps.kinds import Kind, kind_of
from src.hrps.model import ScriptedModel
from src.hrps.runner import RunnerBudget, run_system
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


def test_overseer_components_labeled():
    assert kind_of("bond_overseer") is Kind.EXACT
    assert kind_of("bond_memory") is Kind.EXACT
    assert kind_of("bond_tools") is Kind.EXACT
    assert MODULE_MAP["bond_overseer"] == "src.hrps.bond_overseer"
    assert MODULE_MAP["public_name"] == PUBLIC_NAME


def test_tools_reject_unknown_and_execute_legal():
    env = HrpsEnv(_task())
    env.observe()
    bad = dispatch_tool(env, {"action": "shell", "arguments": {"cmd": "rm"}})
    assert bad["status"] == "rejected"
    assert bad.get("earned") is False
    ok = dispatch_tool(env, {"action": "execute_program", "arguments": {"program": "rot180"}})
    assert ok["status"] == "ok"
    assert ok["residual"]["all_exact"] is True
    assert ok.get("uses_test_labels") is False


def test_memory_records_revision_cycle():
    mem = BondMemory("t")
    mem.record_action(0, {"action": "propose_program", "arguments": {"program": "rot90"}}, {"status": "ok"})
    mem.record_action(1, {"action": "revise_hypothesis", "arguments": {"text": "try 180"}}, {"status": "ok"})
    mem.record_action(2, {"action": "execute_program", "arguments": {"program": "rot180"}}, {"status": "ok", "residual": {"all_exact": True}})
    assert mem.n_revisions() == 1
    snap = mem.snapshot()
    assert snap["n_actions"] == 3
    assert "TEST 0 OUTPUT" not in mem.prompt_block()


def test_overseer_active_loop_solves_with_feedback():
    model = ScriptedModel(
        responses=[
            _j("inspect_objects"),
            _j("execute_program", program="rot90"),
            _j("revise_hypothesis", text="residual remains; rotate 180"),
            _j("execute_program", program="rot180"),
            _j("commit_candidates", program="rot180"),
        ]
    )
    res = run_overseer(_task(), model, RunnerBudget(max_model_calls=8, max_seconds=5), system="bond_hrps")
    assert res.episode.solved is True
    assert res.termination == "committed"
    assert res.episode.n_model_calls >= 3
    assert res.episode.telemetry.get("overseer") is True
    mem = res.episode.telemetry.get("bond_memory") or {}
    assert mem.get("hypotheses")


def test_run_system_uses_overseer():
    model = ScriptedModel(
        responses=[
            _j("execute_program", program="rot180"),
            _j("commit_candidates", program="rot180"),
        ]
    )
    res = run_system(_task(), model, "base_hrps", RunnerBudget(max_model_calls=8, max_seconds=5))
    assert res.episode.solved is True
    guided = (res.episode.telemetry or {}).get("guided") or {}
    assert guided.get("language") == "bond_l1" or res.episode.telemetry.get("overseer") is True


def test_first_real_bond_experiment_is_4b():
    spec = resolve_foundation("Qwen/Qwen3-4B")
    assert spec["id"] == "qwen3.5_4b"
    assert spec.get("first_real_bond_experiment") is True
    twenty7 = resolve_foundation("Qwen/Qwen3-14B")
    assert twenty7.get("first_real_bond_experiment") is not True


def test_qwen38_27b_not_downloaded_locally():
    spec = resolve_foundation("Qwen/Qwen3-14B")
    assert spec["id"] == "qwen38_27b"
    assert spec["hf_id"] == "Qwen/Qwen3-14B"
    assert spec["refuse_local_download"] is True
    blocked = hardware_gate(spec)
    assert blocked is not None
    assert "27B" in blocked or "CUDA" in blocked or "download" in blocked


def test_named_gains_alias_substrate():
    g = named_gains(
        {
            "base_direct": {"solved": 0},
            "base_hrps": {"solved": 1},
            "bond_direct": {"solved": 2},
            "bond_hrps": {"solved": 4},
        }
    )
    assert g["learned_model_gain"] is True
    assert g["hrps_substrate_gain"] == 2

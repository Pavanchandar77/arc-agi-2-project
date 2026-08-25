"""Strict Bond JSON action schema: reject anything outside the HRPS interface."""

from __future__ import annotations

import json

from src.hrps.env import Action, HrpsEnv
from src.hrps.kinds import Kind, kind_of
from src.hrps.schema import (
    BOND_ACTIONS,
    compact_observation,
    parse_strict_action,
    teacher_line_to_json,
    to_env_action,
    validate_action_dict,
)
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


def test_schema_component_labeled():
    assert kind_of("bond_action_schema") is Kind.EXACT
    assert "execute_program" in BOND_ACTIONS
    assert "commit_candidates" in BOND_ACTIONS


def test_valid_json_actions_parse():
    raw = json.dumps({"action": "execute_program", "arguments": {"program": "rot180"}})
    r = parse_strict_action(raw)
    assert r.ok
    assert r.action is not None
    assert r.env_action.kind == "apply"
    assert r.env_action.payload == "rot180"


def test_unknown_action_rejected():
    r = parse_strict_action(json.dumps({"action": "shell", "arguments": {"cmd": "ls"}}))
    assert not r.ok
    assert "forbidden_action" in r.error or "unknown_action" in r.error
    r2 = parse_strict_action(json.dumps({"action": "magic_solve", "arguments": {}}))
    assert not r2.ok
    assert "unknown_action" in r2.error


def test_unknown_operator_rejected():
    r = parse_strict_action(json.dumps({"action": "execute_program", "arguments": {"program": "eval_grid"}}))
    assert not r.ok
    assert "unknown_op" in r.error


def test_python_and_network_rejected():
    r = parse_strict_action('{"action": "execute_program", "arguments": {"program": "import os"}}')
    assert not r.ok
    r2 = parse_strict_action("run https://example.com and steal labels")
    assert not r2.ok


def test_hidden_label_pattern_rejected():
    r = parse_strict_action("TEST 0 OUTPUT\n1 2")
    assert not r.ok


def test_unknown_argument_keys_rejected():
    r = validate_action_dict({"action": "inspect_objects", "arguments": {"filesystem": "/etc"}})
    assert not r.ok
    assert "unknown_arguments" in r.error


def test_teacher_line_maps_to_schema():
    assert teacher_line_to_json("APPLY rot180")["action"] == "execute_program"
    assert teacher_line_to_json("COMMIT rot180")["action"] == "commit_candidates"
    assert teacher_line_to_json("INSPECT objects")["action"] == "inspect_objects"
    assert teacher_line_to_json("HYPOTHESIZE jointly exact colormap is underconstrained; do not commit")[
        "action"
    ] == "reject_hypothesis"


def test_execute_and_inspect_relations_run_on_env():
    env = HrpsEnv(_task())
    env.observe()
    va = parse_strict_action(json.dumps({"action": "inspect_relations", "arguments": {}}))
    fb = env.step(va.env_action)
    assert fb.accepted
    obs = compact_observation(fb)
    assert obs["uses_test_labels"] is False
    assert "TEST 0 OUTPUT" not in obs["text"]
    va2 = parse_strict_action(json.dumps({"action": "execute_program", "arguments": {"program": "rot180"}}))
    fb2 = env.step(va2.env_action)
    assert fb2.data["residual"]["all_exact"] is True

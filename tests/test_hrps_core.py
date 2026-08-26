"""Domain-neutral HRPS core + ARC adapter. Does not replace A–G tests."""

from __future__ import annotations

from src.hrps.arc_adapter import ArcHRPSEnvironment, ArcVerifier
from src.hrps.core import Candidate, HRPSEnvironment, TypedAction
from src.hrps.kinds import Kind, kind_of
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


def test_core_and_adapter_kinds():
    assert kind_of("hrps_core") is Kind.EXACT
    assert kind_of("arc_hrps_adapter") is Kind.EXACT


def test_arc_adapter_is_hrps_environment():
    env = ArcHRPSEnvironment(_task())
    assert isinstance(env, HRPSEnvironment)
    assert env.domain == "arc"


def test_arc_reset_hides_test_gold():
    env = ArcHRPSEnvironment(_task())
    obs = env.reset()
    assert obs.uses_hidden_labels is False
    assert "TEST 0 OUTPUT" not in obs.text
    assert "TEST 0 INPUT" in obs.text


def test_arc_execute_rot180_is_jointly_exact():
    env = ArcHRPSEnvironment(_task())
    env.reset()
    result = env.execute(TypedAction("execute_program", {"program": "rot180"}, domain="arc"))
    assert result.ok
    assert result.observation.payload["residual"]["all_exact"] is True
    assert result.observation.uses_hidden_labels is False


def test_arc_unknown_action_rejected():
    env = ArcHRPSEnvironment(_task())
    env.reset()
    result = env.execute(TypedAction("shell", {"cmd": "ls"}, domain="arc"))
    assert result.ok is False
    assert result.earned is False


def test_arc_verify_is_joint_demo_only():
    env = ArcHRPSEnvironment(_task())
    v = ArcVerifier(env)
    ok = v.verify(Candidate("rot180", domain="arc", kind="program"))
    assert ok.joint_exact is True
    assert ok.transferred is None  # adapter does not score test gold for the agent
    bad = env.verify(Candidate("rot90", domain="arc", kind="program"))
    assert bad.joint_exact is False

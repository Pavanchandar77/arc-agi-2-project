"""Typed HRPS tools Bond is allowed to call.

Domain-neutral dispatch through HRPSEnvironment.execute. The ARC adapter
is the default implementation. Unknown actions never reach any executor.
"""

from __future__ import annotations

from typing import Any

from src.hrps.arc_adapter import ArcHRPSEnvironment, typed_action_from_payload
from src.hrps.core import HRPSEnvironment, TypedAction
from src.hrps.env import HrpsEnv
from src.hrps.schema import BOND_ACTIONS


TOOL_NAMES = BOND_ACTIONS


def _as_environment(env: HRPSEnvironment | HrpsEnv) -> HRPSEnvironment:
    if isinstance(env, HrpsEnv):
        return ArcHRPSEnvironment.wrap(env)
    return env


def dispatch_tool(env: HRPSEnvironment | HrpsEnv, payload: dict[str, Any] | str) -> dict[str, Any]:
    """Validate and execute one Bond tool. Kind: exact reject of illegal calls."""
    environment = _as_environment(env)
    action, err = typed_action_from_payload(payload)
    if action is None:
        return {
            "status": "rejected",
            "error": err or "invalid_tool",
            "uses_test_labels": False,
            "earned": False,
        }
    result = environment.execute(action)
    out = dict(result.observation.payload)
    if "status" not in out:
        out["status"] = "ok" if result.ok else "rejected"
    if result.error and "error" not in out:
        out["error"] = result.error
    out.setdefault("earned", result.earned)
    out.setdefault("uses_test_labels", False)
    return out

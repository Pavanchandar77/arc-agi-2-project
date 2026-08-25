"""Typed HRPS tools Bond is allowed to call.

Every call is schema-validated. Unknown actions never reach the executor.
HRPS returns exact structured observations; it does not explain the rule.
"""

from __future__ import annotations

from typing import Any

from src.hrps.env import HrpsEnv
from src.hrps.schema import BOND_ACTIONS, compact_observation, parse_strict_action, validate_action_dict


TOOL_NAMES = BOND_ACTIONS


def dispatch_tool(env: HrpsEnv, payload: dict[str, Any] | str) -> dict[str, Any]:
    """Validate and execute one Bond tool. Kind: exact reject of illegal calls."""
    if isinstance(payload, str):
        parsed = parse_strict_action(payload)
    else:
        parsed = validate_action_dict(payload)
    if not parsed.ok or parsed.env_action is None:
        return {
            "status": "rejected",
            "error": parsed.error or "invalid_tool",
            "uses_test_labels": False,
            "earned": False,
        }
    fb = env.step(parsed.env_action)
    obs = compact_observation(fb)
    obs["earned"] = True
    obs["tool"] = parsed.action.action if parsed.action else None
    return obs

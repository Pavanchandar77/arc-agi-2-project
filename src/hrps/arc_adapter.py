"""ARC-AGI-2 domain adapter.

Preserves the existing exact executor (dsl.replay) and exact verifier
(joint residual). This is the first HRPSEnvironment implementation, not
the HRPS core.

Kind: exact on grids. Object/relation summaries remain sound_incomplete.
"""

from __future__ import annotations

from typing import Any, Optional

from src.hrps.abstractions import AbstractionLibrary
from src.hrps.core import (
    ActionResult,
    Candidate,
    Observation,
    TypedAction,
    VerificationResult,
)
from src.hrps.dsl import Program, replay
from src.hrps.env import HrpsEnv, parse_program
from src.hrps.residual import joint_residual
from src.hrps.schema import compact_observation, parse_strict_action, validate_action_dict
from src.hrps.task import ArcTask


class ArcHRPSEnvironment:
    """HRPSEnvironment for ARC. Wraps src.hrps.env.HrpsEnv."""

    domain = "arc"

    def __init__(
        self,
        task: ArcTask,
        *,
        inner: Optional[HrpsEnv] = None,
        library: Optional[AbstractionLibrary] = None,
        enable_h: bool = False,
        max_depth: int = 3,
    ) -> None:
        self.task = task
        self.inner = inner or HrpsEnv(task, library=library, enable_h=enable_h, max_depth=max_depth)

    @classmethod
    def wrap(cls, inner: HrpsEnv) -> "ArcHRPSEnvironment":
        obj = cls(inner.task, inner=inner)
        return obj

    def reset(self) -> Observation:
        fb = self.inner.observe()
        return Observation(payload=dict(fb.data or {}), text=fb.text, uses_hidden_labels=False)

    def observe(self, request: dict[str, Any] | None = None) -> Observation:
        if not request:
            return self.reset()
        name = str(request.get("action") or request.get("what") or "inspect_objects")
        args = dict(request.get("arguments") or {})
        if "what" in request and "action" not in request:
            what = str(request["what"])
            name = "inspect_relations" if what.startswith("relation") else "inspect_objects"
        return self.execute(TypedAction(name, args, domain="arc")).observation

    def catalog(self) -> str:
        return self.inner.catalog_text()

    def execute(self, action: TypedAction) -> ActionResult:
        parsed = validate_action_dict({"action": action.name, "arguments": action.arguments})
        if not parsed.ok or parsed.env_action is None:
            obs = Observation(
                payload={"status": "rejected", "error": parsed.error, "uses_test_labels": False, "earned": False},
                text=f"rejected:{parsed.error}",
                uses_hidden_labels=False,
            )
            return ActionResult(ok=False, observation=obs, error=parsed.error, earned=False)
        fb = self.inner.step(parsed.env_action)
        payload = compact_observation(fb)
        payload["earned"] = True
        payload["tool"] = action.name
        payload["domain"] = "arc"
        obs = Observation(payload=payload, text=fb.text, uses_hidden_labels=False)
        return ActionResult(ok=fb.accepted, observation=obs, error=None if fb.accepted else payload.get("error"), earned=True)

    def verify(self, candidate: Candidate) -> VerificationResult:
        """Joint-demonstration exactness only. No test gold in the result shown to the agent."""
        text = candidate.payload if isinstance(candidate.payload, str) else str(candidate.payload)
        program, err = parse_program(text)
        if program is None or err:
            return VerificationResult(joint_exact=False, details={"error": err or "parse_error"})
        preds = tuple(replay(program, p.input) for p in self.task.train)
        gts = self.task.train_outputs()
        jr = joint_residual(preds, gts, spec=None)
        return VerificationResult(
            joint_exact=jr.all_exact,
            details=jr.as_dict(),
            transferred=None,
        )


class ArcVerifier:
    domain = "arc"

    def __init__(self, env: ArcHRPSEnvironment) -> None:
        self.env = env

    def verify(self, candidate: Candidate) -> VerificationResult:
        return self.env.verify(candidate)


def typed_action_from_payload(payload: dict[str, Any] | str) -> tuple[Optional[TypedAction], str]:
    if isinstance(payload, str):
        parsed = parse_strict_action(payload)
    else:
        parsed = validate_action_dict(payload)
    if not parsed.ok or parsed.action is None:
        return None, parsed.error
    return TypedAction(parsed.action.action, dict(parsed.action.arguments), domain="arc"), ""

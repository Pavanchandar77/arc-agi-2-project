"""Bond episode memory: hypotheses, actions, observations, interpretations.

Kind: exact history. Does not invent semantics. Never stores test gold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class HypothesisRecord:
    step: int
    text: str
    status: str  # proposed | revised | rejected | committed
    program: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {"step": self.step, "text": self.text, "status": self.status, "program": self.program}


@dataclass
class BondMemory:
    task_id: str
    hypotheses: list[HypothesisRecord] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    interpretations: list[str] = field(default_factory=list)

    def record_action(self, step: int, action: Optional[dict[str, Any]], observation: dict[str, Any]) -> None:
        self.actions.append({"step": step, "action": action})
        compact = {k: v for k, v in observation.items() if k != "text"}
        self.observations.append({"step": step, **compact})
        if action:
            name = action.get("action")
            args = action.get("arguments") or {}
            if name == "propose_program":
                prog = args.get("program")
                self.hypotheses.append(
                    HypothesisRecord(
                        step,
                        str(args.get("note") or prog or ""),
                        "proposed",
                        str(prog) if prog else None,
                    )
                )
            elif name == "revise_hypothesis":
                prog = args.get("program")
                self.hypotheses.append(
                    HypothesisRecord(step, str(args.get("text") or ""), "revised", str(prog) if prog else None)
                )
            elif name == "reject_hypothesis":
                prog = args.get("program")
                self.hypotheses.append(
                    HypothesisRecord(step, str(args.get("reason") or ""), "rejected", str(prog) if prog else None)
                )
            elif name == "commit_candidates":
                prog = args.get("program")
                self.hypotheses.append(
                    HypothesisRecord(step, "commit", "committed", str(prog) if prog else None)
                )

    def interpret(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.interpretations.append(text)

    def current_hypothesis(self) -> Optional[HypothesisRecord]:
        return self.hypotheses[-1] if self.hypotheses else None

    def n_revisions(self) -> int:
        return sum(1 for h in self.hypotheses if h.status == "revised")

    def n_rejections(self) -> int:
        return sum(1 for h in self.hypotheses if h.status == "rejected")

    def snapshot(self) -> dict[str, Any]:
        """Compact state for Bond's next prompt. No test gold."""
        hyp = self.current_hypothesis()
        last_obs = self.observations[-1] if self.observations else {}
        return {
            "task_id": self.task_id,
            "n_actions": len(self.actions),
            "n_hypotheses": len(self.hypotheses),
            "n_revisions": self.n_revisions(),
            "n_rejections": self.n_rejections(),
            "current_hypothesis": hyp.as_dict() if hyp else None,
            "last_observation": last_obs,
            "last_interpretation": self.interpretations[-1] if self.interpretations else None,
        }

    def prompt_block(self) -> str:
        snap = self.snapshot()
        lines = ["BOND_STATE"]
        if snap["current_hypothesis"]:
            h = snap["current_hypothesis"]
            lines.append(f"hypothesis[{h['status']}]: {h['text']}")
            if h.get("program"):
                lines.append(f"program: {h['program']}")
        obs = snap["last_observation"] or {}
        if obs:
            flags = obs.get("underconstraint_flags") or []
            res = obs.get("residual") or {}
            lines.append(
                f"last_obs: status={obs.get('status')} joint_exact={res.get('all_exact')} "
                f"pixel={res.get('pixel_total')} flags={flags}"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "hypotheses": [h.as_dict() for h in self.hypotheses],
            "actions": self.actions,
            "observations": self.observations,
            "interpretations": self.interpretations,
        }

"""Domain-neutral HRPS contracts.

ARC is the first adapter, not the definition of HRPS. This module has no
grid types and no DSL operators. Exact ARC execution stays in the ARC
adapter wrapping src.hrps.env / dsl / residual.

Kind: exact interfaces. Future domains plug in through these types.
Do not add code/math/proof adapters until this contract is stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class TypedAction:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    domain: str = "generic"

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.name, "arguments": dict(self.arguments), "domain": self.domain}


@dataclass
class Observation:
    payload: dict[str, Any]
    text: str = ""
    uses_hidden_labels: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"payload": self.payload, "text": self.text, "uses_hidden_labels": self.uses_hidden_labels}


@dataclass
class ActionResult:
    ok: bool
    observation: Observation
    error: Optional[str] = None
    earned: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "earned": self.earned,
            "observation": self.observation.as_dict(),
        }


@dataclass
class Candidate:
    payload: Any
    domain: str
    kind: str = "unknown"


@dataclass
class VerificationResult:
    joint_exact: Optional[bool]
    details: dict[str, Any] = field(default_factory=dict)
    transferred: Optional[bool] = None  # training harness only; never shown as an observation

    def as_dict(self) -> dict[str, Any]:
        return {"joint_exact": self.joint_exact, "details": self.details, "transferred": self.transferred}


@dataclass
class HypothesisUpdate:
    text: str
    status: str  # proposed | revised | rejected | committed
    program: Optional[str] = None


@dataclass
class NextDecision:
    kind: str  # continue | commit | stop
    action: Optional[TypedAction] = None
    reason: str = ""


@dataclass
class HRPSBudget:
    max_steps: int = 8
    max_seconds: float = 30.0
    max_tokens: int = 4096


@dataclass
class HRPSState:
    task_id: str
    domain: str
    observation_text: str
    catalog: str
    memory_snapshot: dict[str, Any]
    step: int = 0

    def prompt_block(self) -> str:
        return (
            f"DOMAIN {self.domain} TASK {self.task_id} STEP {self.step}\n"
            + self.observation_text
            + ("\n\nCATALOG:\n" + self.catalog if self.catalog else "")
        )


@runtime_checkable
class HRPSEnvironment(Protocol):
    domain: str

    def reset(self) -> Observation: ...
    def observe(self, request: dict[str, Any] | None = None) -> Observation: ...
    def execute(self, action: TypedAction) -> ActionResult: ...
    def verify(self, candidate: Candidate) -> VerificationResult: ...
    def catalog(self) -> str: ...


@runtime_checkable
class HRPSVerifier(Protocol):
    domain: str

    def verify(self, candidate: Candidate) -> VerificationResult: ...


@runtime_checkable
class HRPSAgent(Protocol):
    def propose_action(self, state: HRPSState) -> TypedAction: ...
    def interpret(self, result: ActionResult) -> HypothesisUpdate: ...
    def revise(self, update: HypothesisUpdate) -> NextDecision: ...

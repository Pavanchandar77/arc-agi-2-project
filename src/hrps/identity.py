"""Bond public identity.

The operational name is Bond. The Qwen foundation remains in the private
manifest and license attribution because Bond is a fine-tune, not a
from-scratch model. Loading Bond without an adapter is an error:

    Bond adapter not found

Never delete the foundation checkpoint: that would discard the intelligence
Bond is trained to improve.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.hrps.model import FrozenOpenModel, ModelTurn, ScriptedModel

PUBLIC_NAME = "Bond"
SMOKE_PUBLIC_NAME = "Bond-smoke"
FOUNDATION_RUNTIME_NAME = "foundation"
ADAPTER_MISSING = "Bond adapter not found"
BACKEND_NAME = "bond_model_backend"

SYSTEM_BOND_DIRECT = (
    "You are Bond. Analyze the demonstrations and emit the output grid "
    "for each test input. Output ONLY the grid as space-separated integers, "
    "one row per line. No explanations."
)

QWEN_LICENSE_NOTE = (
    "Bond is a fine-tuned model-system built on an open Qwen foundation. "
    "The foundation checkpoint must be retained. Bond does not replace that "
    "license; reproduce the foundation LICENSE alongside this adapter."
)


def adapter_dir_hash(path: Optional[Path]) -> Optional[str]:
    if path is None or not Path(path).exists():
        return None
    h = hashlib.sha256()
    files = sorted(p for p in Path(path).rglob("*") if p.is_file())
    if not files:
        return None
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def is_bond_identity(model: object) -> bool:
    return bool(getattr(model, "is_bond", False))


@dataclass
class BondModel:
    """User-facing Bond handle. Inner weights may be foundation+adapter."""

    inner: FrozenOpenModel
    foundation_id: str
    foundation_hf_id: str
    adapter_path: str
    adapter_hash: Optional[str]
    seed: int = 0
    is_smoke: bool = False
    is_merged: bool = False
    is_bond: bool = True
    backend: str = BACKEND_NAME

    @property
    def name(self) -> str:
        return SMOKE_PUBLIC_NAME if self.is_smoke else PUBLIC_NAME

    @property
    def public_name(self) -> str:
        return self.name

    def provenance(self) -> dict[str, Any]:
        return {
            "public_name": self.name,
            "is_bond": True,
            "is_smoke": self.is_smoke,
            "is_merged": self.is_merged,
            "foundation_id": self.foundation_id,
            "foundation_hf_id": self.foundation_hf_id,
            "adapter_path": self.adapter_path,
            "adapter_hash": self.adapter_hash,
            "seed": self.seed,
            "backend": self.backend,
            "license_note": QWEN_LICENSE_NOTE,
        }

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> ModelTurn:
        turn = self.inner.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        turn.model_name = self.name
        turn.backend = self.backend
        return turn


@dataclass
class FoundationModel:
    """Bare foundation. Not Bond. Used for base_direct / base_hrps."""

    inner: FrozenOpenModel
    foundation_id: str
    foundation_hf_id: str
    seed: int = 0
    is_bond: bool = False
    backend: str = "foundation_backend"

    @property
    def name(self) -> str:
        return FOUNDATION_RUNTIME_NAME

    def provenance(self) -> dict[str, Any]:
        return {
            "public_name": FOUNDATION_RUNTIME_NAME,
            "is_bond": False,
            "foundation_id": self.foundation_id,
            "foundation_hf_id": self.foundation_hf_id,
            "seed": self.seed,
            "backend": self.backend,
        }

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> ModelTurn:
        turn = self.inner.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        turn.model_name = self.name
        turn.backend = self.backend
        return turn


def load_bond(
    inner: FrozenOpenModel,
    *,
    adapter_path: str | Path,
    foundation_id: str,
    foundation_hf_id: str,
    seed: int = 0,
    is_smoke: bool = False,
    is_merged: bool = False,
) -> tuple[Optional[BondModel], str]:
    path = Path(adapter_path)
    if not path.exists() or (path.is_dir() and not any(path.iterdir())):
        return None, ADAPTER_MISSING
    return (
        BondModel(
            inner=inner,
            foundation_id=foundation_id,
            foundation_hf_id=foundation_hf_id,
            adapter_path=str(path),
            adapter_hash=adapter_dir_hash(path),
            seed=seed,
            is_smoke=is_smoke,
            is_merged=is_merged,
        ),
        "ok",
    )


def wrap_foundation(
    inner: FrozenOpenModel,
    *,
    foundation_id: str,
    foundation_hf_id: str,
    seed: int = 0,
) -> FoundationModel:
    return FoundationModel(
        inner=inner,
        foundation_id=foundation_id,
        foundation_hf_id=foundation_hf_id,
        seed=seed,
    )


def wrap_scripted_bond(inner: ScriptedModel, adapter_path: str | Path, *, seed: int = 0) -> BondModel:
    model, status = load_bond(
        inner,
        adapter_path=adapter_path,
        foundation_id="scripted",
        foundation_hf_id="scripted",
        seed=seed,
        is_smoke=True,
    )
    if model is None:
        raise FileNotFoundError(status)
    return model

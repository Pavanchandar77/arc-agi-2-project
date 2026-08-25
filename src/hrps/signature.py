"""Conservative continuation-state signatures.

Kind: sound_but_incomplete.

False non-merges are allowed. False merges are not.
Operators are pure functions of the current grids, so identical predicted
grids plus the same legal-operator family imply identical continuations.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from src.hrps.grid import Grid, shape
from src.hrps.kinds import Kind
from src.hrps.dsl import Op, Program


KIND = Kind.SOUND_INCOMPLETE


def _grid_bytes(grid: Grid) -> bytes:
    h, w = shape(grid)
    return bytes([h, w]) + bytes(v for row in grid for v in row)


def continuation_signature(
    preds: tuple[Grid, ...],
    stage: str,
    legal_op_family: str,
) -> bytes:
    """Decidable signature of live continuation state.

    Includes:
      normalized predicted grids (joint residual is a function of these)
      typed live interface (always Grid in this DSL)
      legal-operator family / stage
    Does not include program text, so distinct programs that reach the same
    grids can be merged. Remaining depth is omitted to avoid false splits.
    """
    h = hashlib.sha256()
    h.update(stage.encode("utf-8"))
    h.update(b"|")
    h.update(legal_op_family.encode("utf-8"))
    for g in preds:
        h.update(b"|g|")
        h.update(_grid_bytes(g))
    return h.digest()


def legal_op_family(op_names: Iterable[str]) -> str:
    return ",".join(sorted(set(op_names)))


def program_structure(program: Program) -> tuple[str, ...]:
    return program.names()


def signatures_equal(a: bytes, b: bytes) -> bool:
    return a == b

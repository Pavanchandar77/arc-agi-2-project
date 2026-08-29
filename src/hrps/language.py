"""Bond-L1 language: closed expansion after Phase-1 D3.

Phase-1 conclusions in src.hrps.conclusions are not edited. A–G still
generate the D3 operator family (depth 3, 36 ops/node). Bond-L1 is a
new frozen language used by the Bond-4B engine. After this change the
A/F/G ladder must be rerun on stage L before claiming a new ceiling.

Kind: exact operator set. Search order remains heuristic.
"""

from __future__ import annotations

from typing import Any

from src.hrps.dsl import BOND_ROLE_NAMES, OP_DEFS, stage_config
from src.hrps.search import SearchBudget

LANGUAGE_ID = "bond_l1"
LANGUAGE_DEPTH = 4
LANGUAGE_OPS_PER_NODE = 80
PHASE1_DEPTH = 3
PHASE1_OPS_PER_NODE = 36
FOUNDATION_HF_ID = "Qwen/Qwen3-4B"


def bond_l1_budget(*, nodes: int = 2000, seconds: float = 8.0, frontier: int = 12000) -> SearchBudget:
    return SearchBudget(
        max_depth=LANGUAGE_DEPTH,
        max_nodes=nodes,
        max_seconds=seconds,
        max_frontier=frontier,
        max_ops_per_node=LANGUAGE_OPS_PER_NODE,
    )


def bond_l1_config():
    return stage_config("L")


def language_manifest() -> dict[str, Any]:
    return {
        "id": LANGUAGE_ID,
        "foundation_hf_id": FOUNDATION_HF_ID,
        "max_depth": LANGUAGE_DEPTH,
        "max_ops_per_node": LANGUAGE_OPS_PER_NODE,
        "role_ops": list(BOND_ROLE_NAMES),
        "n_registered_ops": len(OP_DEFS),
        "phase1_untouched": {
            "depth": PHASE1_DEPTH,
            "ops_per_node": PHASE1_OPS_PER_NODE,
            "stages": ["A", "B", "C", "D", "E", "F", "G"],
        },
        "note": (
            "Bond-L1 does not edit frozen Phase-1 conclusions. "
            "Remeasure language-vs-search with stage L before claiming a new ceiling. "
            "Qwen/Qwen3-4B is the only Bond foundation for this system."
        ),
    }

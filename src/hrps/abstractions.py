"""Training-only abstractions (H).

Each abstraction is a named macro over existing exact DSL operators.
Mining uses only training traces whose task IDs are outside the held-out
slice. Execution is exact replay. Selection of which macros exist is
learned from traces; the macros themselves are not neural.

Kind: learned inventory, exact execution.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from src.hrps.dsl import Op, Program, replay
from src.hrps.grid import Grid
from src.hrps.kinds import Kind
from src.hrps.task import ArcTask

# Do not mine colormap macros: those maps are task-palette specific
# (see aabf363d). H names compositional structure only.
_BLOCKED_OP_NAMES = frozenset({"apply_colormap"})


@dataclass(frozen=True)
class Abstraction:
    """Exact macro. Body is a sequence of existing DSL operators."""

    name: str
    body: tuple[Op, ...]
    provenance: tuple[str, ...]
    cost: int
    kind_inventory: Kind = Kind.LEARNED
    kind_execution: Kind = Kind.EXACT

    def body_serialize(self) -> str:
        return " | ".join(op.serialize() for op in self.body)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "body": self.body_serialize(),
            "n_ops": len(self.body),
            "cost": self.cost,
            "provenance": list(self.provenance),
            "kind_inventory": self.kind_inventory.value,
            "kind_execution": self.kind_execution.value,
            "in_types": ["Grid"],
            "out_type": "Grid",
        }

    def execute(self, grid: Grid) -> Optional[Grid]:
        return replay(Program(self.body), grid)


class AbstractionLibrary:
    def __init__(self, items: tuple[Abstraction, ...] = ()) -> None:
        self.items = items
        self._by_name = {a.name: a for a in items}

    def get(self, name: str) -> Abstraction:
        return self._by_name[name]

    def cost(self, name: str) -> int:
        return self._by_name[name].cost

    def execute(self, name: str, grid: Grid) -> Optional[Grid]:
        return self._by_name[name].execute(grid)

    def __len__(self) -> int:
        return len(self.items)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": len(self.items),
            "abstractions": [a.as_dict() for a in self.items],
        }


_ACTIVE = AbstractionLibrary()


def active_library() -> AbstractionLibrary:
    return _ACTIVE


def set_active_library(library: AbstractionLibrary) -> None:
    global _ACTIVE
    _ACTIVE = library


def parse_program_text(text: str) -> tuple[Op, ...]:
    text = text.strip()
    if not text or text == "identity":
        return ()
    return tuple(Op.deserialize(tok.strip()) for tok in text.split("|"))


def _contains_blocked(ops: tuple[Op, ...]) -> bool:
    return any(op.name in _BLOCKED_OP_NAMES for op in ops)


def _slug(ops: tuple[Op, ...]) -> str:
    parts = []
    for op in ops:
        token = op.serialize().replace(":", "_").replace(",", "_").replace("|", "")
        token = re.sub(r"[^a-zA-Z0-9_]+", "", token)
        parts.append(token)
    slug = "_".join(parts)
    return slug[:80] or "seq"


def _abstraction_cost(n_ops: int) -> int:
    """Named-macro cost: cheaper than expanding the body, still positive."""
    return 2 + n_ops


def mine_abstractions_from_rows(
    rows: Iterable[dict[str, Any]],
    exclude_task_ids: Iterable[str],
) -> AbstractionLibrary:
    """Mine contiguous depth>=2 bodies from exact training solutions.

    Source tasks in exclude_task_ids are ignored (held-out slice).
    Only rows with solved=True are used. Colormap bodies are dropped.
    """
    excluded = set(exclude_task_ids)
    bodies: dict[str, list[str]] = {}
    for row in rows:
        tid = row.get("task_id")
        if not tid or tid in excluded:
            continue
        if not row.get("solved"):
            continue
        programs = row.get("programs") or []
        if not programs:
            continue
        ops = parse_program_text(programs[0])
        if len(ops) < 2 or _contains_blocked(ops):
            continue
        for i in range(len(ops)):
            for j in range(i + 2, len(ops) + 1):
                sub = ops[i:j]
                key = " | ".join(op.serialize() for op in sub)
                bodies.setdefault(key, []).append(tid)

    items: list[Abstraction] = []
    used_names: set[str] = set()
    for key, sources in sorted(bodies.items(), key=lambda kv: (-len(set(kv[1])), kv[0])):
        ops = parse_program_text(key)
        slug = _slug(ops)
        name = f"abs_{slug}"
        n = 2
        while name in used_names:
            name = f"abs_{slug}_{n}"
            n += 1
        used_names.add(name)
        items.append(
            Abstraction(
                name=name,
                body=ops,
                provenance=tuple(sorted(set(sources))),
                cost=_abstraction_cost(len(ops)),
            )
        )
    return AbstractionLibrary(tuple(items))


def library_from_dict(payload: dict[str, Any]) -> AbstractionLibrary:
    """Rebuild an exact macro library from a saved JSON payload."""
    items: list[Abstraction] = []
    for row in payload.get("abstractions") or []:
        body = parse_program_text(row["body"])
        items.append(
            Abstraction(
                name=str(row["name"]),
                body=body,
                provenance=tuple(row.get("provenance") or ()),
                cost=int(row.get("cost") or _abstraction_cost(len(body))),
            )
        )
    return AbstractionLibrary(tuple(items))


def load_library_json(path: Path) -> AbstractionLibrary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return library_from_dict(payload)


def mine_from_jsonl(
    jsonl_path: Path,
    exclude_task_ids: Iterable[str],
) -> AbstractionLibrary:
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return mine_abstractions_from_rows(rows, exclude_task_ids)


def transfer_report(
    library: AbstractionLibrary,
    tasks: Iterable[ArcTask],
) -> dict[str, Any]:
    """Apply each abstraction alone to held-out tasks. Logs reuse and negative transfer."""
    rows = []
    n_joint = 0
    n_transfer = 0
    n_negative = 0
    n_reuse_outside_provenance = 0
    for abs_ in library.items:
        for task in tasks:
            from_source = task.task_id in abs_.provenance
            preds = [abs_.execute(p.input) for p in task.train]
            gts = [p.output for p in task.train]
            joint = all(pred is not None and pred == gt for pred, gt in zip(preds, gts))
            test_ok = False
            if joint and task.test and task.test[0].output is not None:
                tpred = abs_.execute(task.test[0].input)
                test_ok = tpred is not None and tpred == task.test[0].output
                if not from_source:
                    n_reuse_outside_provenance += 1
                    n_joint += 1
                    if test_ok:
                        n_transfer += 1
                    else:
                        n_negative += 1
            if joint:
                rows.append(
                    {
                        "abstraction": abs_.name,
                        "body": abs_.body_serialize(),
                        "task_id": task.task_id,
                        "from_provenance": from_source,
                        "joint_demos": True,
                        "test_transfer": test_ok,
                    }
                )
    return {
        "n_abstractions": len(library),
        "n_joint_hits_outside_provenance": n_joint,
        "n_held_out_transfer": n_transfer,
        "n_negative_transfer": n_negative,
        "n_reuse_events": n_reuse_outside_provenance,
        "hits": rows,
    }

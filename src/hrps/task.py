"""ARC-AGI-2 task loader.

Kind: exact.
Does not read public evaluation during search development unless the caller
explicitly names that split. Default split is training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from src.hrps.grid import Grid, as_grid, is_valid_grid, require_valid, shape

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "ARC-AGI-2" / "data"


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Optional[Grid]


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train: tuple[Pair, ...]
    test: tuple[Pair, ...]
    split: str

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def train_inputs(self) -> tuple[Grid, ...]:
        return tuple(p.input for p in self.train)

    def train_outputs(self) -> tuple[Grid, ...]:
        return tuple(p.output for p in self.train if p.output is not None)

    def test_inputs(self) -> tuple[Grid, ...]:
        return tuple(p.input for p in self.test)

    def test_outputs(self) -> tuple[Optional[Grid], ...]:
        return tuple(p.output for p in self.test)

    def max_grid_area(self) -> int:
        areas = []
        for pair in self.train + self.test:
            h, w = shape(pair.input)
            areas.append(h * w)
            if pair.output is not None:
                oh, ow = shape(pair.output)
                areas.append(oh * ow)
        return max(areas) if areas else 0

    def max_dim(self) -> int:
        dims = []
        for pair in self.train + self.test:
            h, w = shape(pair.input)
            dims.extend([h, w])
            if pair.output is not None:
                oh, ow = shape(pair.output)
                dims.extend([oh, ow])
        return max(dims) if dims else 0


def _parse_grid(raw: object, *, required: bool) -> Optional[Grid]:
    if raw is None:
        if required:
            raise ValueError("missing grid")
        return None
    if not is_valid_grid(raw):
        raise ValueError("invalid grid in task file")
    return require_valid(as_grid(raw))  # type: ignore[arg-type]


def parse_task(task_id: str, payload: dict, split: str) -> ArcTask:
    train = []
    for pair in payload["train"]:
        train.append(
            Pair(
                input=_parse_grid(pair["input"], required=True),  # type: ignore[arg-type]
                output=_parse_grid(pair["output"], required=True),
            )
        )
    test = []
    for pair in payload["test"]:
        test.append(
            Pair(
                input=_parse_grid(pair["input"], required=True),  # type: ignore[arg-type]
                output=_parse_grid(pair.get("output"), required=False),
            )
        )
    if not train:
        raise ValueError(f"task {task_id} has no demonstrations")
    if not test:
        raise ValueError(f"task {task_id} has no test inputs")
    return ArcTask(task_id=task_id, train=tuple(train), test=tuple(test), split=split)


def load_task_file(path: Path, split: str) -> ArcTask:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return parse_task(path.stem, payload, split)


def iter_split(
    split: str = "training",
    data_root: Optional[Path] = None,
    max_tasks: Optional[int] = None,
) -> Iterator[ArcTask]:
    """Iterate tasks in sorted id order. Same order for every ablation."""
    if split not in {"training", "evaluation"}:
        raise ValueError(f"unknown split {split!r}")
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    folder = root / split
    if not folder.is_dir():
        raise FileNotFoundError(f"ARC split not found: {folder}")
    paths = sorted(folder.glob("*.json"), key=lambda p: p.stem)
    if max_tasks is not None:
        paths = paths[: max_tasks]
    for path in paths:
        yield load_task_file(path, split)


def list_task_ids(split: str = "training", data_root: Optional[Path] = None) -> list[str]:
    return [t.task_id for t in iter_split(split, data_root=data_root)]

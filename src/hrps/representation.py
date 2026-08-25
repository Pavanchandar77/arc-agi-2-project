"""Object representations. Kind: exact extraction; sound_incomplete as semantics.

Connected components are structural proxies, not semantic objects.
Multiple hypotheses are retained; none is assumed correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.hrps.grid import (
    DELTAS_4,
    DELTAS_8,
    Cell,
    Grid,
    border_majority,
    majority_color,
    shape,
)


@dataclass(frozen=True)
class Object:
    cells: frozenset[Cell]
    color: int
    connectivity: int
    color_agnostic: bool
    bg: int

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return min(rs), min(cs), max(rs), max(cs)

    @property
    def height(self) -> int:
        r0, _, r1, _ = self.bbox
        return r1 - r0 + 1

    @property
    def width(self) -> int:
        _, c0, _, c1 = self.bbox
        return c1 - c0 + 1

    @property
    def centroid(self) -> tuple[float, float]:
        n = max(self.area, 1)
        return (
            sum(r for r, _ in self.cells) / n,
            sum(c for _, c in self.cells) / n,
        )

    def shape_signature(self) -> tuple:
        r0, c0, _, _ = self.bbox
        rel = tuple(sorted((r - r0, c - c0) for r, c in self.cells))
        return (self.height, self.width, self.color if not self.color_agnostic else -1, rel)

    def match_key(self) -> tuple:
        return (self.color, self.area, self.height, self.width, self.shape_signature()[3])


@dataclass(frozen=True)
class RepresentationSpec:
    spec_id: str
    connectivity: int
    color_agnostic: bool
    bg_mode: str  # "zero" | "majority" | "border"


@dataclass(frozen=True)
class Representation:
    spec: RepresentationSpec
    bg: int
    objects: tuple[Object, ...]

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    def relation_count(self) -> int:
        n = self.n_objects
        return n * (n - 1) // 2


# Frozen hypothesis bank. D uses all; B/C use a subset.
SPEC_4_ZERO = RepresentationSpec("4c_zero", 4, False, "zero")
SPEC_4_BORDER = RepresentationSpec("4c_border", 4, False, "border")
SPEC_8_ZERO = RepresentationSpec("8c_zero", 8, False, "zero")
SPEC_8_BORDER = RepresentationSpec("8c_border", 8, False, "border")
SPEC_8A_ZERO = RepresentationSpec("8a_zero", 8, True, "zero")
SPEC_8A_BORDER = RepresentationSpec("8a_border", 8, True, "border")
SPEC_4A_ZERO = RepresentationSpec("4a_zero", 4, True, "zero")

BANK_B = (SPEC_4_ZERO,)
BANK_C = (SPEC_8A_ZERO,)
BANK_D = (
    SPEC_4_ZERO,
    SPEC_4_BORDER,
    SPEC_8_ZERO,
    SPEC_8_BORDER,
    SPEC_8A_ZERO,
    SPEC_8A_BORDER,
    SPEC_4A_ZERO,
)


def resolve_bg(grid: Grid, mode: str) -> int:
    if mode == "zero":
        return 0
    if mode == "majority":
        return majority_color(grid)
    if mode == "border":
        return border_majority(grid)
    raise ValueError(f"unknown bg mode {mode}")


def extract_objects(
    grid: Grid,
    connectivity: int,
    color_agnostic: bool,
    bg: int,
) -> tuple[Object, ...]:
    h, w = shape(grid)
    seen = [[False] * w for _ in range(h)]
    deltas = DELTAS_4 if connectivity == 4 else DELTAS_8
    found: list[Object] = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] == bg:
                continue
            seed = grid[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            cells: list[Cell] = []
            colors: set[int] = set()
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                colors.add(grid[cr][cc])
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr][nc]:
                        continue
                    val = grid[nr][nc]
                    if val == bg:
                        continue
                    if color_agnostic or val == seed:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            color = seed if len(colors) == 1 else -1
            found.append(
                Object(
                    cells=frozenset(cells),
                    color=color,
                    connectivity=connectivity,
                    color_agnostic=color_agnostic,
                    bg=bg,
                )
            )
    found.sort(key=lambda o: (-o.area, o.bbox, o.color))
    return tuple(found)


def build_representation(grid: Grid, spec: RepresentationSpec) -> Representation:
    bg = resolve_bg(grid, spec.bg_mode)
    objects = extract_objects(grid, spec.connectivity, spec.color_agnostic, bg)
    return Representation(spec=spec, bg=bg, objects=objects)


def correspondence_ambiguity(objects: tuple[Object, ...]) -> int:
    """Count of objects that share a match_key with at least one other object."""
    from collections import Counter

    keys = [o.match_key() for o in objects]
    bag = Counter(keys)
    return sum(1 for o in objects if bag[o.match_key()] > 1)


def instability_across(counts: list[int]) -> float:
    if not counts:
        return 0.0
    return float(max(counts) - min(counts))

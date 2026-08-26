"""Exact object-role, count, and rigid-motion primitives for Bond-L1.

These are structural operators: smallest/largest/singleton are defined by
connected-component area, never by task names. Ties and ambiguous colors
return None (refuse), not a guess.

Kind: exact.
"""

from __future__ import annotations

from typing import Optional

from src.hrps.grid import (
    Grid,
    N_COLORS,
    bbox_of,
    counts,
    keep_color,
    shape,
)
from src.hrps.representation import extract_objects

TRANSLATE_DELTA = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _as_bool(flag: object) -> bool:
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag in {"t", "true", "1"}
    return bool(flag)


def paint_cells(grid: Grid, cells: frozenset[tuple[int, int]], color: int) -> Grid:
    h, w = shape(grid)
    keep = cells
    return tuple(
        tuple(color if (r, c) in keep else grid[r][c] for c in range(w))
        for r in range(h)
    )


def _objects(grid: Grid, conn: int, agn: object, bg: int):
    return extract_objects(grid, int(conn), _as_bool(agn), int(bg))


def recolor_smallest_to_largest_color(grid: Grid, conn: int, agn: object, bg: int) -> Optional[Grid]:
    objs = _objects(grid, conn, agn, bg)
    if len(objs) < 2:
        return None
    large, small = objs[0], objs[-1]
    if large.area == small.area or large.color < 0 or small.color < 0:
        return None
    if small.color == large.color:
        return None
    out = paint_cells(grid, small.cells, large.color)
    return None if out == grid else out


def recolor_largest_to_smallest_color(grid: Grid, conn: int, agn: object, bg: int) -> Optional[Grid]:
    objs = _objects(grid, conn, agn, bg)
    if len(objs) < 2:
        return None
    large, small = objs[0], objs[-1]
    if large.area == small.area or large.color < 0 or small.color < 0:
        return None
    if large.color == small.color:
        return None
    out = paint_cells(grid, large.cells, small.color)
    return None if out == grid else out


def erase_smallest(grid: Grid, conn: int, agn: object, bg: int) -> Optional[Grid]:
    objs = _objects(grid, conn, agn, bg)
    if not objs:
        return None
    small = objs[-1]
    if len(objs) > 1 and small.area == objs[0].area:
        return None
    out = paint_cells(grid, small.cells, int(bg))
    return None if out == grid else out


def erase_largest(grid: Grid, conn: int, agn: object, bg: int) -> Optional[Grid]:
    objs = _objects(grid, conn, agn, bg)
    if not objs:
        return None
    large = objs[0]
    if len(objs) > 1 and large.area == objs[-1].area:
        return None
    out = paint_cells(grid, large.cells, int(bg))
    return None if out == grid else out


def recolor_all_fg_to_smallest_color(grid: Grid, conn: int, agn: object, bg: int) -> Optional[Grid]:
    objs = _objects(grid, conn, agn, bg)
    if len(objs) < 2:
        return None
    small = objs[-1]
    if small.area == objs[0].area or small.color < 0:
        return None
    bg_i = int(bg)
    h, w = shape(grid)
    out = tuple(
        tuple(small.color if v != bg_i else v for v in row)
        for row in grid
    )
    return None if out == grid else out


def recolor_nonsingleton_to_singleton_color(grid: Grid, conn: int, agn: object, bg: int) -> Optional[Grid]:
    """If there is exactly one area-1 object and at least one larger object,
    recolor every larger object's cells to the singleton's color.

    Structural role op. Not a task patch.
    """
    objs = _objects(grid, conn, agn, bg)
    singles = [o for o in objs if o.area == 1]
    rest = [o for o in objs if o.area > 1]
    if len(singles) != 1 or not rest:
        return None
    marker = singles[0]
    if marker.color < 0:
        return None
    cells: set[tuple[int, int]] = set()
    for o in rest:
        cells |= set(o.cells)
    out = paint_cells(grid, frozenset(cells), marker.color)
    return None if out == grid else out


def keep_least_frequent_color(grid: Grid, bg: int) -> Optional[Grid]:
    bag = counts(grid)
    bg_i = int(bg)
    cands = [(bag[k], k) for k in range(N_COLORS) if k != bg_i and bag[k] > 0]
    if not cands:
        return None
    min_f = min(f for f, _ in cands)
    cols = [k for f, k in cands if f == min_f]
    if len(cols) != 1:
        return None
    out = keep_color(grid, cols[0], bg_i)
    return None if out == grid else out


def keep_most_frequent_nonbg(grid: Grid, bg: int) -> Optional[Grid]:
    bag = counts(grid)
    bg_i = int(bg)
    cands = [(bag[k], k) for k in range(N_COLORS) if k != bg_i and bag[k] > 0]
    if not cands:
        return None
    max_f = max(f for f, _ in cands)
    cols = [k for f, k in cands if f == max_f]
    if len(cols) != 1:
        return None
    out = keep_color(grid, cols[0], bg_i)
    return None if out == grid else out


def recolor_least_frequent_to_most_frequent(grid: Grid, bg: int) -> Optional[Grid]:
    bag = counts(grid)
    bg_i = int(bg)
    cands = [(bag[k], k) for k in range(N_COLORS) if k != bg_i and bag[k] > 0]
    if len(cands) < 2:
        return None
    min_f = min(f for f, _ in cands)
    max_f = max(f for f, _ in cands)
    rares = [k for f, k in cands if f == min_f]
    commons = [k for f, k in cands if f == max_f]
    if len(rares) != 1 or len(commons) != 1 or rares[0] == commons[0]:
        return None
    src, dst = rares[0], commons[0]
    out = tuple(tuple(dst if v == src else v for v in row) for row in grid)
    return None if out == grid else out


def translate_fg(grid: Grid, direction: int, bg: int) -> Optional[Grid]:
    d = int(direction)
    if d not in (0, 1, 2, 3):
        return None
    return _shift_fg(grid, TRANSLATE_DELTA[d][0], TRANSLATE_DELTA[d][1], int(bg))


def _shift_fg(grid: Grid, dr: int, dc: int, bg: int) -> Optional[Grid]:
    if dr == 0 and dc == 0:
        return None
    h, w = shape(grid)
    cells = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] != bg]
    if not cells:
        return None
    new = [[bg] * w for _ in range(h)]
    for r, c, v in cells:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        new[nr][nc] = v
    out = tuple(tuple(row) for row in new)
    return None if out == grid else out


def center_fg(grid: Grid, bg: int) -> Optional[Grid]:
    bg_i = int(bg)
    box = bbox_of(grid, lambda v: v != bg_i)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    h, w = shape(grid)
    tr = (h - (r1 - r0 + 1)) // 2
    tc = (w - (c1 - c0 + 1)) // 2
    return _shift_fg(grid, tr - r0, tc - c0, bg_i)

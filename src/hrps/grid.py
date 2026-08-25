"""Immutable ARC grids and exact geometric primitives.

Kind: exact.
Grids are 1..30 on each side, cell values in 0..9.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

MAX_DIM = 30
N_COLORS = 10
Color = int
Grid = tuple[tuple[int, ...], ...]
Cell = tuple[int, int]

DELTAS_4: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DELTAS_8: tuple[Cell, ...] = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr != 0 or dc != 0
)


def as_grid(raw: Sequence[Sequence[int]]) -> Grid:
    return tuple(tuple(int(v) for v in row) for row in raw)


def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), (len(grid[0]) if grid else 0)


def is_valid_grid(grid: object) -> bool:
    if not isinstance(grid, (list, tuple)) or not grid:
        return False
    if not isinstance(grid[0], (list, tuple)) or not grid[0]:
        return False
    h = len(grid)
    w = len(grid[0])
    if h < 1 or w < 1 or h > MAX_DIM or w > MAX_DIM:
        return False
    for row in grid:
        if not isinstance(row, (list, tuple)) or len(row) != w:
            return False
        for val in row:
            if not isinstance(val, int) or val < 0 or val > 9:
                return False
    return True


def require_valid(grid: Grid) -> Grid:
    if not is_valid_grid(grid):
        raise ValueError("invalid ARC grid")
    return as_grid(grid)


def grid_hash(grid: Grid) -> int:
    return hash((shape(grid), grid))


def colors_present(grid: Grid) -> frozenset[int]:
    return frozenset(v for row in grid for v in row)


def counts(grid: Grid) -> tuple[int, ...]:
    c = [0] * N_COLORS
    for row in grid:
        for v in row:
            c[v] += 1
    return tuple(c)


def majority_color(grid: Grid) -> int:
    c = counts(grid)
    return max(range(N_COLORS), key=lambda k: (c[k], -k))


def border_majority(grid: Grid) -> int:
    h, w = shape(grid)
    bag = [0] * N_COLORS
    for c in range(w):
        bag[grid[0][c]] += 1
        if h > 1:
            bag[grid[h - 1][c]] += 1
    for r in range(1, h - 1):
        bag[grid[r][0]] += 1
        if w > 1:
            bag[grid[r][w - 1]] += 1
    return max(range(N_COLORS), key=lambda k: (bag[k], -k))


def grids_equal(a: Optional[Grid], b: Optional[Grid]) -> bool:
    return a is not None and b is not None and a == b


def rot90(grid: Grid) -> Grid:
    h, w = shape(grid)
    return tuple(tuple(grid[h - 1 - r][c] for r in range(h)) for c in range(w))


def rot180(grid: Grid) -> Grid:
    h, w = shape(grid)
    return tuple(tuple(grid[h - 1 - r][w - 1 - c] for c in range(w)) for r in range(h))


def rot270(grid: Grid) -> Grid:
    h, w = shape(grid)
    return tuple(tuple(grid[r][w - 1 - c] for r in range(h)) for c in range(w))


def flip_h(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in grid)


def flip_v(grid: Grid) -> Grid:
    return tuple(reversed(grid))


def transpose(grid: Grid) -> Grid:
    h, w = shape(grid)
    return tuple(tuple(grid[r][c] for r in range(h)) for c in range(w))


def anti_transpose(grid: Grid) -> Grid:
    h, w = shape(grid)
    return tuple(tuple(grid[h - 1 - r][w - 1 - c] for r in range(h)) for c in range(w))


D8_OPS = (
    ("identity", lambda g: g),
    ("rot90", rot90),
    ("rot180", rot180),
    ("rot270", rot270),
    ("flip_h", flip_h),
    ("flip_v", flip_v),
    ("transpose", transpose),
    ("anti_transpose", anti_transpose),
)


def bbox_of(grid: Grid, pred) -> Optional[tuple[int, int, int, int]]:
    h, w = shape(grid)
    r0, c0, r1, c1 = h, w, -1, -1
    for r in range(h):
        row = grid[r]
        for c in range(w):
            if pred(row[c]):
                if r < r0:
                    r0 = r
                if c < c0:
                    c0 = c
                if r > r1:
                    r1 = r
                if c > c1:
                    c1 = c
    if r1 < 0:
        return None
    return r0, c0, r1, c1


def crop(grid: Grid, r0: int, c0: int, r1: int, c1: int) -> Optional[Grid]:
    h, w = shape(grid)
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w or r0 > r1 or c0 > c1:
        return None
    out = tuple(grid[r][c0 : c1 + 1] for r in range(r0, r1 + 1))
    return out if is_valid_grid(out) else None


def crop_fg(grid: Grid, bg: int) -> Optional[Grid]:
    box = bbox_of(grid, lambda v: v != bg)
    if box is None:
        return grid
    return crop(grid, *box)


def recolor(grid: Grid, src: int, dst: int) -> Grid:
    if src == dst:
        return grid
    return tuple(tuple(dst if v == src else v for v in row) for row in grid)


def swap_colors(grid: Grid, a: int, b: int) -> Grid:
    if a == b:
        return grid
    return tuple(tuple(b if v == a else a if v == b else v for v in row) for row in grid)


def apply_colormap(grid: Grid, mapping: dict[int, int]) -> Grid:
    return tuple(tuple(mapping.get(v, v) for v in row) for row in grid)


def keep_color(grid: Grid, color: int, bg: int) -> Grid:
    return tuple(tuple(v if v == color else bg for v in row) for row in grid)


def tile(grid: Grid, nr: int, nc: int) -> Optional[Grid]:
    h, w = shape(grid)
    th, tw = h * nr, w * nc
    if th < 1 or tw < 1 or th > MAX_DIM or tw > MAX_DIM:
        return None
    return tuple(grid[r % h] * nc for r in range(th))


def upscale(grid: Grid, k: int) -> Optional[Grid]:
    if k < 2:
        return None
    h, w = shape(grid)
    th, tw = h * k, w * k
    if th > MAX_DIM or tw > MAX_DIM:
        return None
    rows = []
    for row in grid:
        expanded = tuple(v for v in row for _ in range(k))
        for _ in range(k):
            rows.append(expanded)
    return tuple(rows)


def downscale(grid: Grid, k: int) -> Optional[Grid]:
    if k < 2:
        return None
    h, w = shape(grid)
    if h % k or w % k:
        return None
    rows = []
    for r in range(0, h, k):
        out_row = []
        for c in range(0, w, k):
            val = grid[r][c]
            for dr in range(k):
                for dc in range(k):
                    if grid[r + dr][c + dc] != val:
                        return None
            out_row.append(val)
        rows.append(tuple(out_row))
    out = tuple(rows)
    return out if is_valid_grid(out) else None


def left_half(grid: Grid) -> Optional[Grid]:
    h, w = shape(grid)
    if w < 2:
        return None
    nw = w // 2
    out = tuple(row[:nw] for row in grid)
    return out if is_valid_grid(out) else None


def right_half(grid: Grid) -> Optional[Grid]:
    h, w = shape(grid)
    if w < 2:
        return None
    nw = w - w // 2
    out = tuple(row[-nw:] for row in grid)
    return out if is_valid_grid(out) else None


def top_half(grid: Grid) -> Optional[Grid]:
    h, w = shape(grid)
    if h < 2:
        return None
    nh = h // 2
    out = grid[:nh]
    return out if is_valid_grid(out) else None


def bottom_half(grid: Grid) -> Optional[Grid]:
    h, w = shape(grid)
    if h < 2:
        return None
    nh = h - h // 2
    out = grid[-nh:]
    return out if is_valid_grid(out) else None


def gravity(grid: Grid, direction: int, bg: int) -> Grid:
    """Pack non-bg cells toward 0=N, 1=E, 2=S, 3=W. Kind: exact."""
    h, w = shape(grid)
    cells = [[bg] * w for _ in range(h)]
    if direction == 2:  # south
        for c in range(w):
            col = [grid[r][c] for r in range(h) if grid[r][c] != bg]
            for i, v in enumerate(col):
                cells[h - len(col) + i][c] = v
    elif direction == 0:  # north
        for c in range(w):
            col = [grid[r][c] for r in range(h) if grid[r][c] != bg]
            for i, v in enumerate(col):
                cells[i][c] = v
    elif direction == 3:  # west
        for r in range(h):
            row = [v for v in grid[r] if v != bg]
            for i, v in enumerate(row):
                cells[r][i] = v
    elif direction == 1:  # east
        for r in range(h):
            row = [v for v in grid[r] if v != bg]
            for i, v in enumerate(row):
                cells[r][w - len(row) + i] = v
    else:
        return grid
    return tuple(tuple(row) for row in cells)


def fill_holes(grid: Grid, bg: int) -> Grid:
    """Fill bg components not reachable from the border. Kind: exact."""
    h, w = shape(grid)
    reachable = [[False] * w for _ in range(h)]
    stack: list[Cell] = []
    for c in range(w):
        if grid[0][c] == bg:
            stack.append((0, c))
        if h > 1 and grid[h - 1][c] == bg:
            stack.append((h - 1, c))
    for r in range(1, h - 1):
        if grid[r][0] == bg:
            stack.append((r, 0))
        if w > 1 and grid[r][w - 1] == bg:
            stack.append((r, w - 1))
    while stack:
        r, c = stack.pop()
        if reachable[r][c]:
            continue
        reachable[r][c] = True
        for dr, dc in DELTAS_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr][nc] and grid[nr][nc] == bg:
                stack.append((nr, nc))
    fill = majority_non_bg(grid, bg)
    rows = []
    for r in range(h):
        row = []
        for c in range(w):
            v = grid[r][c]
            if v == bg and not reachable[r][c]:
                row.append(fill)
            else:
                row.append(v)
        rows.append(tuple(row))
    return tuple(rows)


def majority_non_bg(grid: Grid, bg: int) -> int:
    bag = [0] * N_COLORS
    for row in grid:
        for v in row:
            if v != bg:
                bag[v] += 1
    if max(bag) == 0:
        return bg
    return max(range(N_COLORS), key=lambda k: (bag[k], -k))


def outline(grid: Grid, bg: int) -> Grid:
    h, w = shape(grid)
    rows = []
    for r in range(h):
        row = []
        for c in range(w):
            v = grid[r][c]
            if v == bg:
                row.append(bg)
                continue
            edge = False
            for dr, dc in DELTAS_4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or grid[nr][nc] == bg:
                    edge = True
                    break
            row.append(v if edge else bg)
        rows.append(tuple(row))
    return tuple(rows)


def paint_mask(grid: Grid, cells: Iterable[Cell], keep_bg: int) -> Grid:
    keep = frozenset(cells)
    h, w = shape(grid)
    return tuple(
        tuple(grid[r][c] if (r, c) in keep else keep_bg for c in range(w))
        for r in range(h)
    )


def pixel_mismatch(pred: Optional[Grid], gt: Grid) -> tuple[int, bool, int]:
    """Return (mismatched_cells, shape_ok, compared_cells). Kind: exact."""
    if pred is None or not is_valid_grid(pred):
        h, w = shape(gt)
        return h * w, False, h * w
    ph, pw = shape(pred)
    gh, gw = shape(gt)
    if ph != gh or pw != gw:
        return max(ph * pw, gh * gw), False, max(ph * pw, gh * gw)
    miss = 0
    for r in range(gh):
        pr = pred[r]
        gr = gt[r]
        for c in range(gw):
            if pr[c] != gr[c]:
                miss += 1
    return miss, True, gh * gw


def to_lists(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]

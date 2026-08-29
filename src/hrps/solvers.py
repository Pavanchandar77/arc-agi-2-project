"""Exact, train-verified solver bank.

Every rule here is a total function from an input grid to an optional output
grid. A rule is only allowed to predict a test output after it has reproduced
*every* demonstration pair exactly. That gate is the whole safety story: a rule
that guesses is a rule that never fires.

Kind: exact. The families are heuristics about which hypotheses are worth
enumerating; the verification of each hypothesis is exact.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from src.hrps.grid import (
    MAX_DIM,
    Grid,
    border_majority,
    colors_present,
    crop,
    crop_fg,
    majority_color,
    shape,
    rot90,
    rot180,
    rot270,
    flip_h,
    flip_v,
    transpose,
    anti_transpose,
)
from src.hrps.representation import extract_objects
from src.hrps.task import ArcTask

Predictor = Callable[[Grid], Optional[Grid]]

D8: tuple[tuple[str, Callable[[Grid], Grid]], ...] = (
    ("id", lambda g: g),
    ("rot90", rot90),
    ("rot180", rot180),
    ("rot270", rot270),
    ("flip_h", flip_h),
    ("flip_v", flip_v),
    ("transpose", transpose),
    ("anti_transpose", anti_transpose),
)


@dataclass(frozen=True)
class Rule:
    """A verified hypothesis. Lower rank sorts first among equally verified rules."""

    name: str
    rank: int
    predict: Predictor


def _bgs(task: ArcTask) -> tuple[int, ...]:
    out = {0}
    for p in task.train:
        out.add(majority_color(p.input))
        out.add(border_majority(p.input))
    return tuple(sorted(out))


def _blank(h: int, w: int, color: int) -> Grid:
    row = (color,) * w
    return tuple(row for _ in range(h))


def _fits(h: int, w: int) -> bool:
    return 1 <= h <= MAX_DIM and 1 <= w <= MAX_DIM


# --------------------------------------------------------------------------
# Panel splitting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """A grid decomposed into an nrows x ncols array of equally shaped panels."""

    nrows: int
    ncols: int
    panels: tuple[Grid, ...]  # row-major
    sep_color: Optional[int]


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in indices:
        if out and out[-1][1] == i - 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def _split_by_separator(grid: Grid, color: int) -> Optional[Split]:
    h, w = shape(grid)
    sep_rows = [r for r in range(h) if all(v == color for v in grid[r])]
    sep_cols = [c for c in range(w) if all(grid[r][c] == color for r in range(h))]
    row_bands = _bands(h, _runs(sep_rows))
    col_bands = _bands(w, _runs(sep_cols))
    if row_bands is None or col_bands is None:
        return None
    if len(row_bands) * len(col_bands) < 2:
        return None
    ph = row_bands[0][1] - row_bands[0][0] + 1
    pw = col_bands[0][1] - col_bands[0][0] + 1
    if any(b[1] - b[0] + 1 != ph for b in row_bands):
        return None
    if any(b[1] - b[0] + 1 != pw for b in col_bands):
        return None
    panels = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            sub = crop(grid, r0, c0, r1, c1)
            if sub is None:
                return None
            panels.append(sub)
    return Split(len(row_bands), len(col_bands), tuple(panels), color)


def _bands(size: int, sep_runs: list[tuple[int, int]]) -> Optional[list[tuple[int, int]]]:
    """Content bands between separator runs. None if separators touch an edge oddly."""
    bands: list[tuple[int, int]] = []
    cursor = 0
    for a, b in sep_runs:
        if a > cursor:
            bands.append((cursor, a - 1))
        elif a < cursor:
            return None
        cursor = b + 1
    if cursor < size:
        bands.append((cursor, size - 1))
    if not bands:
        return None
    return bands


def _split_equal(grid: Grid, nrows: int, ncols: int) -> Optional[Split]:
    h, w = shape(grid)
    if nrows < 1 or ncols < 1 or nrows * ncols < 2:
        return None
    if h % nrows or w % ncols:
        return None
    ph, pw = h // nrows, w // ncols
    panels = []
    for i in range(nrows):
        for j in range(ncols):
            sub = crop(grid, i * ph, j * pw, i * ph + ph - 1, j * pw + pw - 1)
            if sub is None:
                return None
            panels.append(sub)
    return Split(nrows, ncols, tuple(panels), None)


def split_methods() -> tuple[str, ...]:
    names = [f"sep:{c}" for c in range(10)]
    for k in (2, 3, 4):
        names.append(f"eq:{k}x1")
        names.append(f"eq:1x{k}")
    names.append("eq:2x2")
    return tuple(names)


def apply_split(grid: Grid, method: str) -> Optional[Split]:
    kind, arg = method.split(":", 1)
    if kind == "sep":
        return _split_by_separator(grid, int(arg))
    a, b = arg.split("x")
    return _split_equal(grid, int(a), int(b))


# --------------------------------------------------------------------------
# Family: constant output
# --------------------------------------------------------------------------


def _family_constant(task: ArcTask) -> Iterator[Rule]:
    outs = task.train_outputs()
    if len(outs) >= 2 and len(set(outs)) == 1:
        const = outs[0]
        yield Rule("constant", 40, lambda g, o=const: o)


# --------------------------------------------------------------------------
# Family: whole-grid D8 (with optional colormap)
# --------------------------------------------------------------------------


def _learn_colormap(pairs: list[tuple[Grid, Grid]]) -> Optional[dict[int, int]]:
    mapping: dict[int, int] = {}
    for pred, gt in pairs:
        if shape(pred) != shape(gt):
            return None
        for pr, gr in zip(pred, gt):
            for a, b in zip(pr, gr):
                if mapping.setdefault(a, b) != b:
                    return None
    return mapping


def _family_d8_colormap(task: ArcTask) -> Iterator[Rule]:
    ins, outs = task.train_inputs(), task.train_outputs()
    for name, fn in D8:
        try:
            preds = [fn(g) for g in ins]
        except Exception:
            continue
        cmap = _learn_colormap(list(zip(preds, outs)))
        if cmap is None:
            continue

        def predict(g: Grid, fn=fn, cmap=cmap) -> Optional[Grid]:
            t = fn(g)
            return tuple(tuple(cmap.get(v, v) for v in row) for row in t)

        # A bare transform is a simpler hypothesis than one needing a recolour.
        rank = (10 if name == "id" else 8) + (0 if all(k == v for k, v in cmap.items()) else 2)
        yield Rule(f"d8_colormap:{name}", rank, predict)


# --------------------------------------------------------------------------
# Family: tiling / mosaic
# --------------------------------------------------------------------------


def _tile_factors(task: ArcTask) -> Optional[tuple[int, int]]:
    fr = fc = None
    for p in task.train:
        ih, iw = shape(p.input)
        oh, ow = shape(p.output)  # type: ignore[arg-type]
        if oh % ih or ow % iw:
            return None
        a, b = oh // ih, ow // iw
        if fr is None:
            fr, fc = a, b
        elif (fr, fc) != (a, b):
            return None
    if fr is None or fc is None or fr * fc < 2 or fr > 6 or fc > 6:
        return None
    return fr, fc


def _family_tiling(task: ArcTask) -> Iterator[Rule]:
    factors = _tile_factors(task)
    if factors is None:
        return
    fr, fc = factors
    # For each output block position, find a single D8 transform that explains
    # it across every demonstration.
    chosen: list[list[Optional[Callable[[Grid], Grid]]]] = []
    for i in range(fr):
        row: list[Optional[Callable[[Grid], Grid]]] = []
        for j in range(fc):
            pick = None
            for name, fn in D8:
                ok = True
                for p in task.train:
                    ih, iw = shape(p.input)
                    try:
                        t = fn(p.input)
                    except Exception:
                        ok = False
                        break
                    if shape(t) != (ih, iw):
                        ok = False
                        break
                    block = crop(p.output, i * ih, j * iw, i * ih + ih - 1, j * iw + iw - 1)  # type: ignore[arg-type]
                    if block != t:
                        ok = False
                        break
                if ok:
                    pick = fn
                    break
            row.append(pick)
        chosen.append(row)
    if any(cell is None for row in chosen for cell in row):
        return

    def predict(g: Grid, fr=fr, fc=fc, chosen=chosen) -> Optional[Grid]:
        h, w = shape(g)
        if not _fits(h * fr, w * fc):
            return None
        blocks = [[fn(g) for fn in row] for row in chosen]  # type: ignore[misc]
        rows: list[tuple[int, ...]] = []
        for i in range(fr):
            for r in range(h):
                acc: tuple[int, ...] = ()
                for j in range(fc):
                    b = blocks[i][j]
                    if shape(b) != (h, w):
                        return None
                    acc = acc + b[r]
                rows.append(acc)
        return tuple(rows)

    yield Rule(f"tile_d8:{fr}x{fc}", 6, predict)


# --------------------------------------------------------------------------
# Family: fractal self-tiling (output h*h x w*w, block present per predicate)
# --------------------------------------------------------------------------


def _family_fractal(task: ArcTask) -> Iterator[Rule]:
    for p in task.train:
        ih, iw = shape(p.input)
        oh, ow = shape(p.output)  # type: ignore[arg-type]
        if (oh, ow) != (ih * ih, iw * iw):
            return
    palette = set()
    for p in task.train:
        palette |= set(colors_present(p.input))
    for bg in _bgs(task):
        for invert in (False, True):
            for fill_bg in sorted(palette | {0}):

                def predict(
                    g: Grid, bg=bg, invert=invert, fill_bg=fill_bg
                ) -> Optional[Grid]:
                    h, w = shape(g)
                    if not _fits(h * h, w * w):
                        return None
                    rows: list[tuple[int, ...]] = []
                    for i in range(h):
                        for r in range(h):
                            acc: tuple[int, ...] = ()
                            for j in range(w):
                                on = g[i][j] != bg
                                if invert:
                                    on = not on
                                acc = acc + (g[r] if on else (fill_bg,) * w)
                            rows.append(acc)
                    return tuple(rows)

                yield Rule(f"fractal:bg{bg}:inv{int(invert)}:f{fill_bg}", 6, predict)


# --------------------------------------------------------------------------
# Family: uniform scaling, including content-derived factors
# --------------------------------------------------------------------------


def _upscale(g: Grid, kr: int, kc: int) -> Optional[Grid]:
    h, w = shape(g)
    if kr < 1 or kc < 1 or not _fits(h * kr, w * kc):
        return None
    rows = []
    for row in g:
        wide = tuple(v for v in row for _ in range(kc))
        for _ in range(kr):
            rows.append(wide)
    return tuple(rows)


def _count_nonbg(g: Grid, bg: int) -> int:
    return sum(1 for row in g for v in row if v != bg)


def _n_colors_nonbg(g: Grid, bg: int) -> int:
    return len({v for row in g for v in row if v != bg})


_FACTOR_FEATURES: tuple[tuple[str, Callable[[Grid, int], int]], ...] = (
    ("nonbg_count", _count_nonbg),
    ("n_colors", _n_colors_nonbg),
    ("max_dim", lambda g, bg: max(shape(g))),
    ("min_dim", lambda g, bg: min(shape(g))),
)


def _family_scale(task: ArcTask) -> Iterator[Rule]:
    ratios = []
    for p in task.train:
        ih, iw = shape(p.input)
        oh, ow = shape(p.output)  # type: ignore[arg-type]
        if oh % ih or ow % iw:
            return
        ratios.append((oh // ih, ow // iw))
    if not ratios or all(r == (1, 1) for r in ratios):
        return
    if len(set(ratios)) == 1:
        kr, kc = ratios[0]
        yield Rule(f"scale:{kr}x{kc}", 5, lambda g, kr=kr, kc=kc: _upscale(g, kr, kc))
        return
    # Ratio varies per task instance: try to read it off the input.
    for bg in _bgs(task):
        for fname, feat in _FACTOR_FEATURES:
            ok_sq = True
            for p, (kr, kc) in zip(task.train, ratios):
                try:
                    v = feat(p.input, bg)
                except Exception:
                    ok_sq = False
                    break
                if (kr, kc) != (v, v):
                    ok_sq = False
                    break
            if ok_sq:

                def predict(g: Grid, bg=bg, feat=feat) -> Optional[Grid]:
                    k = feat(g, bg)
                    return _upscale(g, k, k)

                yield Rule(f"scale_feat:{fname}:bg{bg}", 7, predict)


# --------------------------------------------------------------------------
# Family: cellwise function over panels
# --------------------------------------------------------------------------


def _family_panel_cellwise(task: ArcTask) -> Iterator[Rule]:
    for method in split_methods():
        splits = []
        ok = True
        for p in task.train:
            s = apply_split(p.input, method)
            if s is None:
                ok = False
                break
            splits.append(s)
        if not ok or not splits:
            continue
        n = len(splits[0].panels)
        if n < 2 or n > 6:
            continue
        if any(len(s.panels) != n for s in splits):
            continue
        pshape = shape(splits[0].panels[0])
        if any(shape(pan) != pshape for s in splits for pan in s.panels):
            continue
        if any(shape(p.output) != shape(s.panels[0]) for p, s in zip(task.train, splits)):  # type: ignore[arg-type]
            continue
        table: dict[tuple[int, ...], int] = {}
        consistent = True
        for p, s in zip(task.train, splits):
            ph, pw = shape(s.panels[0])
            for r in range(ph):
                for c in range(pw):
                    key = tuple(pan[r][c] for pan in s.panels)
                    val = p.output[r][c]  # type: ignore[index]
                    if table.setdefault(key, val) != val:
                        consistent = False
                        break
                if not consistent:
                    break
            if not consistent:
                break
        if not consistent:
            continue

        def predict(g: Grid, method=method, table=table, n=n) -> Optional[Grid]:
            s = apply_split(g, method)
            if s is None or len(s.panels) != n:
                return None
            ph, pw = shape(s.panels[0])
            if any(shape(pan) != (ph, pw) for pan in s.panels):
                return None
            rows = []
            for r in range(ph):
                row = []
                for c in range(pw):
                    key = tuple(pan[r][c] for pan in s.panels)
                    if key not in table:
                        return None
                    row.append(table[key])
                rows.append(tuple(row))
            return tuple(rows)

        yield Rule(f"panel_cellwise:{method}", 4, predict)


# --------------------------------------------------------------------------
# Family: select one panel
# --------------------------------------------------------------------------


def _panel_scores(panels: tuple[Grid, ...], bg: int) -> dict[str, list[float]]:
    return {
        "n_colors": [float(len(colors_present(p))) for p in panels],
        "n_nonbg": [float(_count_nonbg(p, bg)) for p in panels],
        "n_distinct_nonbg": [float(_n_colors_nonbg(p, bg)) for p in panels],
        "symmetry": [
            float((p == flip_h(p)) + (p == flip_v(p)) + (p == rot180(p))) for p in panels
        ],
    }


def _family_panel_select(task: ArcTask) -> Iterator[Rule]:
    for method in split_methods():
        splits = []
        ok = True
        for p in task.train:
            s = apply_split(p.input, method)
            if s is None:
                ok = False
                break
            splits.append(s)
        if not ok or not splits:
            continue
        n = len(splits[0].panels)
        if n < 2 or any(len(s.panels) != n for s in splits):
            continue
        # Constant index.
        for idx in range(n):
            if all(s.panels[idx] == p.output for p, s in zip(task.train, splits)):

                def predict(g: Grid, method=method, idx=idx, n=n) -> Optional[Grid]:
                    s = apply_split(g, method)
                    if s is None or len(s.panels) != n:
                        return None
                    return s.panels[idx]

                yield Rule(f"panel_index:{method}:{idx}", 5, predict)
        # Extremal by score, and the odd-one-out.
        for bg in _bgs(task):
            for key in ("n_colors", "n_nonbg", "n_distinct_nonbg", "symmetry"):
                for want_max in (True, False):
                    good = True
                    for p, s in zip(task.train, splits):
                        pick = _pick_extremal(s.panels, bg, key, want_max)
                        if pick is None or pick != p.output:
                            good = False
                            break
                    if good:

                        def predict(
                            g: Grid, method=method, bg=bg, key=key, want_max=want_max, n=n
                        ) -> Optional[Grid]:
                            s = apply_split(g, method)
                            if s is None or len(s.panels) != n:
                                return None
                            return _pick_extremal(s.panels, bg, key, want_max)

                        yield Rule(f"panel_{'max' if want_max else 'min'}:{method}:{key}", 6, predict)
            good = all(_pick_unique(s.panels) == p.output for p, s in zip(task.train, splits))
            if good and _pick_unique(splits[0].panels) is not None:

                def predict(g: Grid, method=method, n=n) -> Optional[Grid]:
                    s = apply_split(g, method)
                    if s is None or len(s.panels) != n:
                        return None
                    return _pick_unique(s.panels)

                yield Rule(f"panel_unique:{method}", 6, predict)


def _pick_extremal(panels: tuple[Grid, ...], bg: int, key: str, want_max: bool) -> Optional[Grid]:
    scores = _panel_scores(panels, bg)[key]
    target = max(scores) if want_max else min(scores)
    hits = [p for p, s in zip(panels, scores) if s == target]
    return hits[0] if len(hits) == 1 else None


def _pick_unique(panels: tuple[Grid, ...]) -> Optional[Grid]:
    bag = Counter(panels)
    singles = [p for p in panels if bag[p] == 1]
    return singles[0] if len(singles) == 1 and len(bag) > 1 else None


# --------------------------------------------------------------------------
# Family: select one object and crop it
# --------------------------------------------------------------------------


_OBJ_KEYS: tuple[str, ...] = ("area", "n_colors", "n_cells_bbox", "density", "holes")


def _obj_score(cells: frozenset[tuple[int, int]], grid: Grid, key: str, bg: int) -> float:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    h = max(rs) - min(rs) + 1
    w = max(cs) - min(cs) + 1
    if key == "area":
        return float(len(cells))
    if key == "n_colors":
        return float(len({grid[r][c] for r, c in cells}))
    if key == "n_cells_bbox":
        return float(h * w)
    if key == "density":
        return len(cells) / float(h * w)
    if key == "holes":
        return float(h * w - len(cells))
    return 0.0


def _crop_cells(grid: Grid, cells: frozenset[tuple[int, int]], bg: Optional[int]) -> Optional[Grid]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    box = crop(grid, min(rs), min(cs), max(rs), max(cs))
    if box is None or bg is None:
        return box
    r0, c0 = min(rs), min(cs)
    return tuple(
        tuple(grid[r0 + i][c0 + j] if (r0 + i, c0 + j) in cells else bg for j in range(len(box[0])))
        for i in range(len(box))
    )


def _family_object_select(task: ArcTask) -> Iterator[Rule]:
    specs = [(conn, agn, bg) for conn in (4, 8) for agn in (False, True) for bg in _bgs(task)]
    for conn, agn, bg in specs:
        for mask in (False, True):
            for key in _OBJ_KEYS:
                for want_max in (True, False):

                    def predict(
                        g: Grid, conn=conn, agn=agn, bg=bg, key=key, want_max=want_max, mask=mask
                    ) -> Optional[Grid]:
                        objs = extract_objects(g, conn, agn, bg)
                        if len(objs) < 2:
                            return None
                        scores = [_obj_score(o.cells, g, key, bg) for o in objs]
                        target = max(scores) if want_max else min(scores)
                        hits = [o for o, s in zip(objs, scores) if s == target]
                        if len(hits) != 1:
                            return None
                        return _crop_cells(g, hits[0].cells, bg if mask else None)

                    tag = "max" if want_max else "min"
                    yield Rule(
                        f"obj_{tag}:{key}:c{conn}:a{int(agn)}:bg{bg}:m{int(mask)}", 9, predict
                    )
            # The object whose shape signature is unique among its peers.
            def predict_odd(
                g: Grid, conn=conn, agn=agn, bg=bg, mask=mask
            ) -> Optional[Grid]:
                objs = extract_objects(g, conn, agn, bg)
                if len(objs) < 3:
                    return None
                sigs = [o.shape_signature()[3] for o in objs]
                bag = Counter(sigs)
                odd = [o for o, s in zip(objs, sigs) if bag[s] == 1]
                if len(odd) != 1 or len(bag) < 2:
                    return None
                return _crop_cells(g, odd[0].cells, bg if mask else None)

            yield Rule(f"obj_odd_shape:c{conn}:a{int(agn)}:bg{bg}:m{int(mask)}", 9, predict_odd)


# --------------------------------------------------------------------------
# Family: crop
# --------------------------------------------------------------------------


def _family_crop(task: ArcTask) -> Iterator[Rule]:
    for bg in _bgs(task):
        yield Rule(f"crop_fg:bg{bg}", 7, lambda g, bg=bg: crop_fg(g, bg))
        # Interior of a single-colour rectangular frame.
        for color in range(10):

            def predict(g: Grid, color=color) -> Optional[Grid]:
                cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == color]
                if len(cells) < 4:
                    return None
                r0 = min(r for r, _ in cells)
                r1 = max(r for r, _ in cells)
                c0 = min(c for _, c in cells)
                c1 = max(c for _, c in cells)
                if r1 - r0 < 2 or c1 - c0 < 2:
                    return None
                return crop(g, r0 + 1, c0 + 1, r1 - 1, c1 - 1)

            yield Rule(f"crop_inside:{color}", 8, predict)

            def predict_box(g: Grid, color=color) -> Optional[Grid]:
                cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == color]
                if not cells:
                    return None
                r0 = min(r for r, _ in cells)
                r1 = max(r for r, _ in cells)
                c0 = min(c for _, c in cells)
                c1 = max(c for _, c in cells)
                return crop(g, r0, c0, r1, c1)

            yield Rule(f"crop_color_box:{color}", 8, predict_box)


# --------------------------------------------------------------------------
# Family: symmetry repair / occlusion fill
# --------------------------------------------------------------------------


def _admitted_symmetries(
    grid: Grid, noise: int
) -> list[Callable[[int, int], tuple[int, int]]]:
    h, w = shape(grid)
    cands: list[tuple[str, Callable[[int, int], tuple[int, int]]]] = [
        ("mh", lambda r, c: (r, w - 1 - c)),
        ("mv", lambda r, c: (h - 1 - r, c)),
        ("r180", lambda r, c: (h - 1 - r, w - 1 - c)),
    ]
    if h == w:
        cands.append(("tr", lambda r, c: (c, r)))
        cands.append(("atr", lambda r, c: (w - 1 - c, h - 1 - r)))
        cands.append(("r90", lambda r, c: (c, h - 1 - r)))
        cands.append(("r270", lambda r, c: (w - 1 - c, r)))
    for p in range(1, h):
        if h % p == 0 and p != h:
            cands.append((f"pv{p}", lambda r, c, p=p: ((r + p) % h, c)))
    for q in range(1, w):
        if w % q == 0 and q != w:
            cands.append((f"ph{q}", lambda r, c, q=q: (r, (c + q) % w)))
    good = []
    for _, fn in cands:
        agree = 0
        bad = False
        for r in range(h):
            for c in range(w):
                a = grid[r][c]
                if a == noise:
                    continue
                nr, nc = fn(r, c)
                if not (0 <= nr < h and 0 <= nc < w):
                    bad = True
                    break
                b = grid[nr][nc]
                if b == noise:
                    continue
                if a != b:
                    bad = True
                    break
                agree += 1
            if bad:
                break
        if not bad and agree >= max(4, (h * w) // 8):
            good.append(fn)
    return good


def _repair(grid: Grid, noise: int) -> Optional[Grid]:
    h, w = shape(grid)
    syms = _admitted_symmetries(grid, noise)
    if not syms:
        return None
    cells = [list(row) for row in grid]
    holes = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == noise]
    if not holes:
        return None
    for _ in range(4):
        progress = False
        for r, c in holes:
            if cells[r][c] != noise:
                continue
            for fn in syms:
                nr, nc = fn(r, c)
                if 0 <= nr < h and 0 <= nc < w and cells[nr][nc] != noise:
                    cells[r][c] = cells[nr][nc]
                    progress = True
                    break
        if not progress:
            break
    if any(cells[r][c] == noise for r, c in holes):
        return None
    return tuple(tuple(row) for row in cells)


def _family_symmetry_repair(task: ArcTask) -> Iterator[Rule]:
    palette: set[int] = set()
    for p in task.train:
        palette |= set(colors_present(p.input))
    same_shape = all(shape(p.input) == shape(p.output) for p in task.train)  # type: ignore[arg-type]
    for noise in sorted(palette):
        if same_shape:
            yield Rule(f"sym_repair_full:{noise}", 5, lambda g, n=noise: _repair(g, n))

        def predict_patch(g: Grid, n=noise) -> Optional[Grid]:
            full = _repair(g, n)
            if full is None:
                return None
            cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == n]
            if not cells:
                return None
            r0 = min(r for r, _ in cells)
            r1 = max(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            c1 = max(c for _, c in cells)
            return crop(full, r0, c0, r1, c1)

        yield Rule(f"sym_repair_patch:{noise}", 5, predict_patch)


# --------------------------------------------------------------------------
# Family: colour remap by frequency rank
# --------------------------------------------------------------------------


def _rank_map(g: Grid, bg: int) -> list[int]:
    bag = Counter(v for row in g for v in row if v != bg)
    return [c for c, _ in sorted(bag.items(), key=lambda kv: (-kv[1], kv[0]))]


def _family_rank_recolor(task: ArcTask) -> Iterator[Rule]:
    if not all(shape(p.input) == shape(p.output) for p in task.train):  # type: ignore[arg-type]
        return
    for bg in _bgs(task):
        perm: dict[int, int] = {}
        ok = True
        for p in task.train:
            ranks = _rank_map(p.input, bg)
            pos = {c: i for i, c in enumerate(ranks)}
            for ri, row in enumerate(p.input):
                for ci, v in enumerate(row):
                    o = p.output[ri][ci]  # type: ignore[index]
                    if v == bg:
                        if o != bg:
                            ok = False
                            break
                        continue
                    src = pos.get(v)
                    dst = pos.get(o)
                    if src is None or dst is None:
                        ok = False
                        break
                    if perm.setdefault(src, dst) != dst:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok and perm and any(k != v for k, v in perm.items()):

            def predict(g: Grid, bg=bg, perm=dict(perm)) -> Optional[Grid]:
                ranks = _rank_map(g, bg)
                pos = {c: i for i, c in enumerate(ranks)}
                out = []
                for row in g:
                    orow = []
                    for v in row:
                        if v == bg:
                            orow.append(v)
                            continue
                        i = pos.get(v)
                        j = perm.get(i) if i is not None else None
                        if j is None or j >= len(ranks):
                            return None
                        orow.append(ranks[j])
                    out.append(tuple(orow))
                return tuple(out)

            yield Rule(f"rank_recolor:bg{bg}", 8, predict)


# --------------------------------------------------------------------------
# Family: compress duplicate rows/columns
# --------------------------------------------------------------------------


def _dedup(g: Grid, rows: bool, cols: bool) -> Optional[Grid]:
    cur = g
    if rows:
        keep = [cur[0]]
        for row in cur[1:]:
            if row != keep[-1]:
                keep.append(row)
        cur = tuple(keep)
    if cols:
        t = transpose(cur)
        keep = [t[0]]
        for row in t[1:]:
            if row != keep[-1]:
                keep.append(row)
        cur = transpose(tuple(keep))
    return cur if cur else None


def _family_dedup(task: ArcTask) -> Iterator[Rule]:
    for rows, cols in ((True, True), (True, False), (False, True)):
        yield Rule(
            f"dedup:r{int(rows)}c{int(cols)}", 7, lambda g, r=rows, c=cols: _dedup(g, r, c)
        )


# --------------------------------------------------------------------------
# Family: single-cell / uniform answers
# --------------------------------------------------------------------------


_CELL_SELECTORS: tuple[tuple[str, Callable[[Grid, int], Optional[int]]], ...] = (
    ("most_common_nonbg", lambda g, bg: _freq_pick(g, bg, True)),
    ("least_common_nonbg", lambda g, bg: _freq_pick(g, bg, False)),
    ("majority", lambda g, bg: majority_color(g)),
    ("border", lambda g, bg: border_majority(g)),
)


def _freq_pick(g: Grid, bg: int, most: bool) -> Optional[int]:
    bag = Counter(v for row in g for v in row if v != bg)
    if not bag:
        return None
    target = max(bag.values()) if most else min(bag.values())
    hits = [c for c, n in bag.items() if n == target]
    return hits[0] if len(hits) == 1 else None


def _family_uniform(task: ArcTask) -> Iterator[Rule]:
    outs = task.train_outputs()
    oshapes = {shape(o) for o in outs}
    if len(oshapes) != 1:
        return
    oh, ow = next(iter(oshapes))
    if not all(len(set(v for row in o for v in row)) == 1 for o in outs):
        return
    for bg in _bgs(task):
        for name, sel in _CELL_SELECTORS:

            def predict(g: Grid, bg=bg, sel=sel, oh=oh, ow=ow) -> Optional[Grid]:
                c = sel(g, bg)
                if c is None:
                    return None
                return _blank(oh, ow, c)

            yield Rule(f"uniform:{name}:bg{bg}", 9, predict)


# --------------------------------------------------------------------------
# Family: per-object recolour driven by a learned feature -> colour map
# --------------------------------------------------------------------------


def _object_features(
    obj_cells: frozenset[tuple[int, int]], grid: Grid, all_objs: list, bg: int
) -> dict[str, object]:
    rs = [r for r, _ in obj_cells]
    cs = [c for _, c in obj_cells]
    h = max(rs) - min(rs) + 1
    w = max(cs) - min(cs) + 1
    colors = {grid[r][c] for r, c in obj_cells}
    areas = sorted({len(o) for o in all_objs}, reverse=True)
    sigs = Counter(_norm_sig(o) for o in all_objs)
    gh, gw = shape(grid)
    return {
        "area": len(obj_cells),
        "area_rank": areas.index(len(obj_cells)),
        "area_rank_rev": len(areas) - 1 - areas.index(len(obj_cells)),
        "shape": _norm_sig(obj_cells),
        "color": next(iter(colors)) if len(colors) == 1 else -1,
        "dims": (h, w),
        "n_same_shape": sigs[_norm_sig(obj_cells)],
        "touches_border": any(r in (0, gh - 1) or c in (0, gw - 1) for r, c in obj_cells),
        "is_rect": len(obj_cells) == h * w,
        "n_holes": h * w - len(obj_cells),
    }


def _norm_sig(cells: frozenset[tuple[int, int]]) -> tuple:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


_RECOLOR_FEATURES = ("area", "area_rank", "area_rank_rev", "shape", "n_same_shape", "dims", "n_holes")


def _object_specs(task: ArcTask) -> list[tuple[int, bool, int]]:
    return [(conn, agn, bg) for conn in (4, 8) for agn in (False, True) for bg in _bgs(task)]


def _family_object_recolor(task: ArcTask) -> Iterator[Rule]:
    if not all(shape(p.input) == shape(p.output) for p in task.train):  # type: ignore[arg-type]
        return
    for conn, agn, bg in _object_specs(task):
        # Every object must become monochromatic; background must be untouched.
        table_by_feat: dict[str, dict[object, int]] = {f: {} for f in _RECOLOR_FEATURES}
        alive = set(_RECOLOR_FEATURES)
        usable = True
        for p in task.train:
            objs = extract_objects(p.input, conn, agn, bg)
            if not objs or len(objs) > 60:
                usable = False
                break
            cellsets = [o.cells for o in objs]
            covered: set[tuple[int, int]] = set()
            for cells in cellsets:
                covered |= cells
            gh, gw = shape(p.input)
            for r in range(gh):
                for c in range(gw):
                    if (r, c) not in covered and p.output[r][c] != p.input[r][c]:  # type: ignore[index]
                        usable = False
                        break
                if not usable:
                    break
            if not usable:
                break
            for cells in cellsets:
                outc = {p.output[r][c] for r, c in cells}  # type: ignore[index]
                if len(outc) != 1:
                    usable = False
                    break
                dst = next(iter(outc))
                feats = _object_features(cells, p.input, cellsets, bg)
                for f in list(alive):
                    key = feats[f]
                    if table_by_feat[f].setdefault(key, dst) != dst:
                        alive.discard(f)
                if not alive:
                    usable = False
                    break
            if not usable:
                break
        if not usable:
            continue
        for f in sorted(alive):
            table = table_by_feat[f]
            if len(set(table.values())) < 2 and len(table) < 2:
                continue

            def predict(g: Grid, conn=conn, agn=agn, bg=bg, f=f, table=dict(table)) -> Optional[Grid]:
                objs = extract_objects(g, conn, agn, bg)
                if not objs:
                    return None
                cellsets = [o.cells for o in objs]
                cells_out = [list(row) for row in g]
                for cells in cellsets:
                    feats = _object_features(cells, g, cellsets, bg)
                    dst = table.get(feats[f])
                    if dst is None:
                        return None
                    for r, c in cells:
                        cells_out[r][c] = dst
                return tuple(tuple(row) for row in cells_out)

            yield Rule(f"obj_recolor:{f}:c{conn}:a{int(agn)}:bg{bg}", 7, predict)


# --------------------------------------------------------------------------
# Family: keep / erase objects by a learned single-feature predicate
# --------------------------------------------------------------------------


_KEEP_PREDICATES: tuple[str, ...] = (
    "max_area",
    "min_area",
    "unique_shape",
    "common_shape",
    "touches_border",
    "is_rect",
    "has_holes",
)


def _keep_flags(cellsets: list, grid: Grid, bg: int, pred: str, negate: bool) -> Optional[list[bool]]:
    if not cellsets:
        return None
    feats = [_object_features(cs, grid, cellsets, bg) for cs in cellsets]
    areas = [int(f["area"]) for f in feats]
    if pred == "max_area":
        flags = [a == max(areas) for a in areas]
    elif pred == "min_area":
        flags = [a == min(areas) for a in areas]
    elif pred == "unique_shape":
        flags = [int(f["n_same_shape"]) == 1 for f in feats]
    elif pred == "common_shape":
        top = max(int(f["n_same_shape"]) for f in feats)
        flags = [int(f["n_same_shape"]) == top for f in feats]
    elif pred == "touches_border":
        flags = [bool(f["touches_border"]) for f in feats]
    elif pred == "is_rect":
        flags = [bool(f["is_rect"]) for f in feats]
    elif pred == "has_holes":
        flags = [int(f["n_holes"]) > 0 for f in feats]
    else:
        return None
    return [(not f) if negate else f for f in flags]


def _family_object_filter(task: ArcTask) -> Iterator[Rule]:
    if not all(shape(p.input) == shape(p.output) for p in task.train):  # type: ignore[arg-type]
        return
    for conn, agn, bg in _object_specs(task):
        for pred in _KEEP_PREDICATES:
            for negate in (False, True):

                def predict(
                    g: Grid, conn=conn, agn=agn, bg=bg, pred=pred, negate=negate
                ) -> Optional[Grid]:
                    objs = extract_objects(g, conn, agn, bg)
                    if len(objs) < 2 or len(objs) > 80:
                        return None
                    cellsets = [o.cells for o in objs]
                    flags = _keep_flags(cellsets, g, bg, pred, negate)
                    if flags is None:
                        return None
                    out = [list(row) for row in g]
                    for cells, keep in zip(cellsets, flags):
                        if keep:
                            continue
                        for r, c in cells:
                            out[r][c] = bg
                    return tuple(tuple(row) for row in out)

                tag = f"not_{pred}" if negate else pred
                yield Rule(f"obj_keep:{tag}:c{conn}:a{int(agn)}:bg{bg}", 8, predict)


# --------------------------------------------------------------------------
# Family: local neighbourhood rule (learned lookup, abstains when unseen)
# --------------------------------------------------------------------------


def _patch(g: Grid, r: int, c: int, radius: int) -> tuple[int, ...]:
    h, w = shape(g)
    out = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            out.append(g[nr][nc] if 0 <= nr < h and 0 <= nc < w else 10)
    return tuple(out)


def _family_neighborhood(task: ArcTask) -> Iterator[Rule]:
    if not all(shape(p.input) == shape(p.output) for p in task.train):  # type: ignore[arg-type]
        return
    if sum(shape(p.input)[0] * shape(p.input)[1] for p in task.train) > 6000:
        return
    for radius in (1, 2):
        table: dict[tuple[int, ...], int] = {}
        ok = True
        for p in task.train:
            gh, gw = shape(p.input)
            for r in range(gh):
                for c in range(gw):
                    key = _patch(p.input, r, c, radius)
                    val = p.output[r][c]  # type: ignore[index]
                    if table.setdefault(key, val) != val:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue

        def predict(g: Grid, radius=radius, table=table) -> Optional[Grid]:
            gh, gw = shape(g)
            rows = []
            for r in range(gh):
                row = []
                for c in range(gw):
                    v = table.get(_patch(g, r, c, radius))
                    if v is None:
                        return None
                    row.append(v)
                rows.append(tuple(row))
            return tuple(rows)

        # Ranked last: it memorises, so it only wins when nothing structural does.
        yield Rule(f"neighborhood:r{radius}", 20, predict)
        return


# --------------------------------------------------------------------------
# Family: denoise and border edits
# --------------------------------------------------------------------------


def _family_denoise(task: ArcTask) -> Iterator[Rule]:
    for conn, agn, bg in _object_specs(task):
        for max_area in (1, 2):

            def predict(g: Grid, conn=conn, agn=agn, bg=bg, max_area=max_area) -> Optional[Grid]:
                objs = extract_objects(g, conn, agn, bg)
                if not objs:
                    return None
                out = [list(row) for row in g]
                hit = False
                for o in objs:
                    if len(o.cells) <= max_area:
                        hit = True
                        for r, c in o.cells:
                            out[r][c] = bg
                if not hit:
                    return None
                return tuple(tuple(row) for row in out)

            yield Rule(f"denoise:{max_area}:c{conn}:a{int(agn)}:bg{bg}", 9, predict)


def _family_border(task: ArcTask) -> Iterator[Rule]:
    palette: set[int] = set()
    for p in task.train:
        palette |= set(colors_present(p.output))  # type: ignore[arg-type]
    for color in sorted(palette):
        for k in (1, 2):

            def add(g: Grid, color=color, k=k) -> Optional[Grid]:
                h, w = shape(g)
                if not _fits(h + 2 * k, w + 2 * k):
                    return None
                nw = w + 2 * k
                rows = [(color,) * nw] * k
                for row in g:
                    rows.append((color,) * k + row + (color,) * k)
                rows.extend([(color,) * nw] * k)
                return tuple(rows)

            yield Rule(f"add_border:{color}:{k}", 9, add)
    for k in (1, 2):

        def strip(g: Grid, k=k) -> Optional[Grid]:
            h, w = shape(g)
            if h <= 2 * k or w <= 2 * k:
                return None
            return crop(g, k, k, h - 1 - k, w - 1 - k)

        yield Rule(f"strip_border:{k}", 9, strip)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


FAMILIES: tuple[Callable[[ArcTask], Iterator[Rule]], ...] = (
    _family_d8_colormap,
    _family_panel_cellwise,
    _family_panel_select,
    _family_symmetry_repair,
    _family_tiling,
    _family_fractal,
    _family_scale,
    _family_crop,
    _family_dedup,
    _family_border,
    _family_object_recolor,
    _family_rank_recolor,
    _family_object_filter,
    _family_object_select,
    _family_denoise,
    _family_uniform,
    _family_constant,
    _family_neighborhood,
)


def verify(rule: Rule, task: ArcTask) -> bool:
    """Exact gate: the rule must reproduce every demonstration output."""
    for pair in task.train:
        try:
            pred = rule.predict(pair.input)
        except Exception:
            return False
        if pred is None or pred != pair.output:
            return False
    return True


def fit_rules(task: ArcTask, *, deadline: Optional[float] = None) -> list[Rule]:
    """Every rule that survives exact verification, best first.

    `deadline` is an absolute time.perf_counter() value; families are checked
    between yields so a pathological task cannot overrun the caller's budget.
    """
    import time

    found: list[Rule] = []
    seen: set[str] = set()
    for family in FAMILIES:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        try:
            for rule in family(task):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                if rule.name in seen:
                    continue
                if verify(rule, task):
                    seen.add(rule.name)
                    found.append(rule)
        except Exception:
            continue
    found.sort(key=lambda r: (r.rank, r.name))
    return found


def solve(task: ArcTask, *, deadline: Optional[float] = None) -> tuple[list[list[Grid]], list[str]]:
    """Up to two distinct predictions per test input, plus the rule names used.

    Returns ([attempt_1_grids, attempt_2_grids], names). Each attempt list has
    one entry per test input; a missing prediction is reported as None-free by
    falling back to the input grid, which keeps the caller's schema total.
    """
    rules = fit_rules(task, deadline=deadline)
    test_inputs = task.test_inputs()
    per_test: list[list[Grid]] = []
    used: list[str] = []
    for inp in test_inputs:
        preds: list[Grid] = []
        for rule in rules:
            try:
                p = rule.predict(inp)
            except Exception:
                continue
            if p is None:
                continue
            from src.hrps.grid import is_valid_grid

            if not is_valid_grid(p):
                continue
            if p not in preds:
                preds.append(p)
                if rule.name not in used:
                    used.append(rule.name)
            if len(preds) >= 2:
                break
        per_test.append(preds)
    a1 = [(p[0] if p else inp) for p, inp in zip(per_test, test_inputs)]
    a2 = [(p[1] if len(p) > 1 else (p[0] if p else inp)) for p, inp in zip(per_test, test_inputs)]
    return [a1, a2], used

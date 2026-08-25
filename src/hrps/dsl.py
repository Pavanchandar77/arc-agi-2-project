"""Finite typed DSL: operators, costs, preconditions, coordinate-frame effects.

Kind: exact.
This is a closed operator set. Do not grow it because a task failed; measure
the failure mode first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from src.hrps.grid import (
    Grid,
    apply_colormap,
    bottom_half,
    crop_fg,
    downscale,
    fill_holes,
    flip_h,
    flip_v,
    gravity,
    is_valid_grid,
    keep_color,
    left_half,
    outline,
    paint_mask,
    recolor,
    right_half,
    rot90,
    rot180,
    rot270,
    shape,
    swap_colors,
    tile,
    top_half,
    transpose,
    anti_transpose,
    upscale,
    colors_present,
)
from src.hrps.kinds import Kind
from src.hrps.representation import extract_objects
from src.hrps.task import ArcTask


class FrameEffect(str, Enum):
    IDENTITY = "identity"
    ROTATE = "rotate"
    REFLECT = "reflect"
    CROP = "crop"
    RESIZE = "resize"
    RECOLOR = "recolor"
    OBJECT = "object"
    FILTER = "filter"


class DslType(str, Enum):
    GRID = "Grid"
    COLOR = "Color"
    INT = "Int"
    BOOL = "Bool"
    COLORMAP = "ColorMap"
    BG = "Bg"
    CONNECTIVITY = "Connectivity"


@dataclass(frozen=True)
class Op:
    """A fully bound operator application. Canonical serialization is exact."""

    name: str
    args: tuple

    def serialize(self) -> str:
        if not self.args:
            return self.name
        parts = []
        for a in self.args:
            if isinstance(a, tuple):
                if a and isinstance(a[0], tuple):
                    parts.append(";".join(f"{x}-{y}" for x, y in a))
                else:
                    parts.append("x".join(str(x) for x in a))
            elif isinstance(a, bool):
                parts.append("t" if a else "f")
            else:
                parts.append(str(a))
        return self.name + ":" + ",".join(parts)

    @staticmethod
    def deserialize(text: str) -> "Op":
        if ":" not in text:
            return Op(text, ())
        name, rest = text.split(":", 1)
        if name == "apply_colormap":
            pairs = tuple(tuple(int(x) for x in p.split("-")) for p in rest.split(";") if p)
            return Op(name, (pairs,))
        raw_args: list = []
        for tok in rest.split(","):
            if tok in {"t", "f"}:
                raw_args.append(tok == "t")
            elif "x" in tok and name == "tile":
                a, b = tok.split("x")
                raw_args.append((int(a), int(b)))
            else:
                try:
                    raw_args.append(int(tok))
                except ValueError:
                    raw_args.append(tok)
        return Op(name, tuple(raw_args))


@dataclass(frozen=True)
class OpDef:
    name: str
    in_types: tuple[DslType, ...]
    out_type: DslType
    frame_effect: FrameEffect
    cost: int
    kind: Kind
    preconditions: str
    execute: Callable[..., Optional[Grid]]


def _ok(grid: Optional[Grid]) -> Optional[Grid]:
    if grid is None or not is_valid_grid(grid):
        return None
    return grid


def _exec_isolate(grid: Grid, connectivity: int, agnostic: bool, bg: int, smallest: bool) -> Optional[Grid]:
    objs = extract_objects(grid, connectivity, agnostic, bg)
    if not objs:
        return None
    obj = objs[-1] if smallest else objs[0]
    return paint_mask(grid, obj.cells, bg)


OP_DEFS: dict[str, OpDef] = {}


def _reg(defn: OpDef) -> None:
    OP_DEFS[defn.name] = defn


_reg(OpDef("rot90", (DslType.GRID,), DslType.GRID, FrameEffect.ROTATE, 2, Kind.EXACT, "any grid", lambda g: _ok(rot90(g))))
_reg(OpDef("rot180", (DslType.GRID,), DslType.GRID, FrameEffect.ROTATE, 2, Kind.EXACT, "any grid", lambda g: _ok(rot180(g))))
_reg(OpDef("rot270", (DslType.GRID,), DslType.GRID, FrameEffect.ROTATE, 2, Kind.EXACT, "any grid", lambda g: _ok(rot270(g))))
_reg(OpDef("flip_h", (DslType.GRID,), DslType.GRID, FrameEffect.REFLECT, 2, Kind.EXACT, "any grid", lambda g: _ok(flip_h(g))))
_reg(OpDef("flip_v", (DslType.GRID,), DslType.GRID, FrameEffect.REFLECT, 2, Kind.EXACT, "any grid", lambda g: _ok(flip_v(g))))
_reg(OpDef("transpose", (DslType.GRID,), DslType.GRID, FrameEffect.REFLECT, 2, Kind.EXACT, "any grid", lambda g: _ok(transpose(g))))
_reg(OpDef("anti_transpose", (DslType.GRID,), DslType.GRID, FrameEffect.REFLECT, 2, Kind.EXACT, "any grid", lambda g: _ok(anti_transpose(g))))
_reg(OpDef("crop_fg", (DslType.GRID, DslType.BG), DslType.GRID, FrameEffect.CROP, 3, Kind.EXACT, "some non-bg cell", lambda g, bg: _ok(crop_fg(g, bg))))
_reg(OpDef("tile", (DslType.GRID, DslType.INT, DslType.INT), DslType.GRID, FrameEffect.RESIZE, 3, Kind.EXACT, "nr*h<=30 and nc*w<=30", lambda g, nr, nc: _ok(tile(g, nr, nc))))
_reg(OpDef("upscale", (DslType.GRID, DslType.INT), DslType.GRID, FrameEffect.RESIZE, 3, Kind.EXACT, "k in {2,3}, scaled dim<=30", lambda g, k: _ok(upscale(g, k))))
_reg(OpDef("downscale", (DslType.GRID, DslType.INT), DslType.GRID, FrameEffect.RESIZE, 3, Kind.EXACT, "dims divisible by k, uniform blocks", lambda g, k: _ok(downscale(g, k))))
_reg(OpDef("recolor", (DslType.GRID, DslType.COLOR, DslType.COLOR), DslType.GRID, FrameEffect.RECOLOR, 6, Kind.EXACT, "src != dst", lambda g, s, d: _ok(recolor(g, s, d))))
_reg(OpDef("swap_colors", (DslType.GRID, DslType.COLOR, DslType.COLOR), DslType.GRID, FrameEffect.RECOLOR, 6, Kind.EXACT, "a < b", lambda g, a, b: _ok(swap_colors(g, a, b))))
_reg(OpDef("keep_color", (DslType.GRID, DslType.COLOR, DslType.BG), DslType.GRID, FrameEffect.FILTER, 5, Kind.EXACT, "color present", lambda g, c, bg: _ok(keep_color(g, c, bg))))
_reg(OpDef("left_half", (DslType.GRID,), DslType.GRID, FrameEffect.CROP, 3, Kind.EXACT, "width>=2", lambda g: _ok(left_half(g))))
_reg(OpDef("right_half", (DslType.GRID,), DslType.GRID, FrameEffect.CROP, 3, Kind.EXACT, "width>=2", lambda g: _ok(right_half(g))))
_reg(OpDef("top_half", (DslType.GRID,), DslType.GRID, FrameEffect.CROP, 3, Kind.EXACT, "height>=2", lambda g: _ok(top_half(g))))
_reg(OpDef("bottom_half", (DslType.GRID,), DslType.GRID, FrameEffect.CROP, 3, Kind.EXACT, "height>=2", lambda g: _ok(bottom_half(g))))
_reg(OpDef("fill_holes", (DslType.GRID, DslType.BG), DslType.GRID, FrameEffect.OBJECT, 4, Kind.EXACT, "any grid", lambda g, bg: _ok(fill_holes(g, bg))))
_reg(OpDef("outline", (DslType.GRID, DslType.BG), DslType.GRID, FrameEffect.OBJECT, 4, Kind.EXACT, "any grid", lambda g, bg: _ok(outline(g, bg))))
_reg(OpDef("gravity", (DslType.GRID, DslType.INT, DslType.BG), DslType.GRID, FrameEffect.OBJECT, 4, Kind.EXACT, "dir in {0,1,2,3}", lambda g, d, bg: _ok(gravity(g, d, bg))))
_reg(OpDef("isolate_largest", (DslType.GRID, DslType.CONNECTIVITY, DslType.BOOL, DslType.BG), DslType.GRID, FrameEffect.OBJECT, 5, Kind.EXACT, "at least one object", lambda g, conn, agn, bg: _ok(_exec_isolate(g, conn, agn, bg, False))))
_reg(OpDef("isolate_smallest", (DslType.GRID, DslType.CONNECTIVITY, DslType.BOOL, DslType.BG), DslType.GRID, FrameEffect.OBJECT, 5, Kind.EXACT, "at least one object", lambda g, conn, agn, bg: _ok(_exec_isolate(g, conn, agn, bg, True))))


def _exec_colormap(grid: Grid, mapping: tuple[tuple[int, int], ...]) -> Optional[Grid]:
    return _ok(apply_colormap(grid, dict(mapping)))


_reg(OpDef(
    "apply_colormap",
    (DslType.GRID, DslType.COLORMAP),
    DslType.GRID,
    FrameEffect.RECOLOR,
    8,
    Kind.EXACT,
    "mapping is a total function on observed colors",
    _exec_colormap,
))

GEOM_UNARY = ("rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "anti_transpose")
HALF_UNARY = ("left_half", "right_half", "top_half", "bottom_half")
MIN_OP_COST = 2

# Residual-domain families used by stage F (heuristic prioritization).
DOMAIN_OPS = {
    "shape": frozenset({"crop_fg", "tile", "upscale", "downscale", *HALF_UNARY}),
    "object": frozenset({"isolate_largest", "isolate_smallest", "fill_holes", "outline", "gravity", "keep_color"}),
    "pixel": frozenset({"recolor", "swap_colors", "apply_colormap", *GEOM_UNARY}),
    "relation": frozenset({"gravity", "isolate_largest", "isolate_smallest", *GEOM_UNARY}),
}


@dataclass(frozen=True)
class Program:
    ops: tuple[Op, ...]

    def serialize(self) -> str:
        if not self.ops:
            return "identity"
        return " | ".join(op.serialize() for op in self.ops)

    def cost(self) -> int:
        total = 0
        for op in self.ops:
            if op.name == "abs":
                from src.hrps.abstractions import active_library

                total += active_library().cost(op.args[0])
                continue
            total += OP_DEFS[op.name].cost
            if op.name == "apply_colormap" and op.args:
                total += 2 * len(op.args[0])
        return total

    def depth(self) -> int:
        return len(self.ops)

    def names(self) -> tuple[str, ...]:
        return tuple(op.name for op in self.ops)

    def extend(self, op: Op) -> "Program":
        return Program(self.ops + (op,))


def execute_op(op: Op, grid: Grid) -> Optional[Grid]:
    if op.name == "abs":
        from src.hrps.abstractions import active_library

        try:
            return active_library().execute(op.args[0], grid)
        except Exception:
            return None
    defn = OP_DEFS[op.name]
    try:
        if not op.args:
            return defn.execute(grid)
        if op.name == "tile":
            nr, nc = op.args[0] if isinstance(op.args[0], tuple) else op.args
            return defn.execute(grid, nr, nc)
        return defn.execute(grid, *op.args)
    except Exception:
        return None


def replay(program: Program, grid: Grid) -> Optional[Grid]:
    """Exact sequential execution. Kind: exact."""
    cur: Optional[Grid] = grid
    for op in program.ops:
        if cur is None:
            return None
        cur = execute_op(op, cur)
    return cur if cur is None or is_valid_grid(cur) else None


def infer_colormap(preds: tuple[Grid, ...], gts: tuple[Grid, ...]) -> Optional[tuple[tuple[int, int], ...]]:
    """Exact generator: pixelwise color function on same-shape pairs, or none."""
    mapping: dict[int, int] = {}
    any_change = False
    for pred, gt in zip(preds, gts):
        if shape(pred) != shape(gt):
            return None
        h, w = shape(pred)
        for r in range(h):
            pr, gr = pred[r], gt[r]
            for c in range(w):
                a, b = pr[c], gr[c]
                prev = mapping.get(a)
                if prev is None:
                    mapping[a] = b
                    if a != b:
                        any_change = True
                elif prev != b:
                    return None
    if not any_change or not mapping:
        return None
    return tuple(sorted(mapping.items()))


@dataclass(frozen=True)
class SearchStageConfig:
    """Independently switchable A–G mechanisms. Shared DSL/executor/budget elsewhere."""

    stage: str
    use_geom: bool = True
    use_color: bool = True
    use_size: bool = True
    object_specs: tuple = ()
    residual_factorization: bool = False
    continuation_dedup: bool = False
    score_joint: bool = True
    require_joint_solution: bool = True
    abstractions: tuple = ()


def stage_config(stage: str, abstractions: tuple = ()) -> SearchStageConfig:
    from src.hrps.representation import BANK_B, BANK_C, BANK_D

    stage = stage.upper()
    if stage == "A":
        return SearchStageConfig("A", abstractions=abstractions)
    if stage == "B":
        return SearchStageConfig("B", object_specs=BANK_B, abstractions=abstractions)
    if stage == "C":
        return SearchStageConfig("C", object_specs=BANK_C, abstractions=abstractions)
    if stage == "D":
        return SearchStageConfig("D", object_specs=BANK_D, abstractions=abstractions)
    if stage == "E":
        return SearchStageConfig(
            "E",
            object_specs=BANK_D,
            score_joint=True,
            require_joint_solution=True,
            abstractions=abstractions,
        )
    if stage == "F":
        return SearchStageConfig(
            "F",
            object_specs=BANK_D,
            residual_factorization=True,
            score_joint=True,
            abstractions=abstractions,
        )
    if stage == "G":
        return SearchStageConfig(
            "G",
            object_specs=BANK_D,
            residual_factorization=True,
            continuation_dedup=True,
            score_joint=True,
            abstractions=abstractions,
        )
    raise ValueError(f"unknown stage {stage!r}; expected A–G")


def _bg_candidates(task: ArcTask) -> tuple[int, ...]:
    bgs = {0}
    for p in task.train:
        bgs.add(0)
        from src.hrps.grid import border_majority, majority_color

        bgs.add(majority_color(p.input))
        bgs.add(border_majority(p.input))
    return tuple(sorted(bgs))


def generate_ops(
    task: ArcTask,
    preds: tuple[Grid, ...],
    gts: tuple[Grid, ...],
    cfg: SearchStageConfig,
    dominant_domain: str,
) -> list[Op]:
    """Generate bound operators. Kind: exact generators; F ordering is heuristic."""
    # Family order is canonical and finite: generators, geometry, size, color, objects.
    # Do not sort alphabetically — color ops would starve geometry under a per-node cap.
    ops: list[Op] = []
    seen: set[str] = set()

    def _add(op: Op) -> None:
        key = op.serialize()
        if key not in seen:
            seen.add(key)
            ops.append(op)

    cmap = infer_colormap(preds, gts) if cfg.use_color else None
    if cmap is not None:
        _add(Op("apply_colormap", (cmap,)))
    if cfg.use_geom:
        for name in GEOM_UNARY:
            _add(Op(name, ()))
    for abs_ in cfg.abstractions:
        _add(Op("abs", (abs_.name,)))
    bgs = _bg_candidates(task)
    if cfg.use_size:
        for name in HALF_UNARY:
            _add(Op(name, ()))
        for bg in bgs:
            _add(Op("crop_fg", (bg,)))
        for k in (2, 3):
            _add(Op("tile", ((k, k),)))
            _add(Op("tile", ((k, 1),)))
            _add(Op("tile", ((1, k),)))
            _add(Op("upscale", (k,)))
            _add(Op("downscale", (k,)))
    if cfg.use_color:
        srcs: set[int] = set()
        dsts: set[int] = set()
        for g in preds:
            srcs |= set(colors_present(g))
        for g in gts:
            dsts |= set(colors_present(g))
        palette = srcs | dsts
        for s in sorted(srcs):
            for d in sorted(palette):
                if s != d:
                    _add(Op("recolor", (s, d)))
        for a in sorted(palette):
            for b in sorted(palette):
                if a < b:
                    _add(Op("swap_colors", (a, b)))
        for c in sorted(srcs):
            for bg in bgs:
                if c != bg:
                    _add(Op("keep_color", (c, bg)))
    for spec in cfg.object_specs:
        conn = spec.connectivity
        agn = spec.color_agnostic
        for bg in bgs:
            _add(Op("fill_holes", (bg,)))
            _add(Op("outline", (bg,)))
            for d in (0, 1, 2, 3):
                _add(Op("gravity", (d, bg)))
            _add(Op("isolate_largest", (conn, agn, bg)))
            _add(Op("isolate_smallest", (conn, agn, bg)))
    if cfg.residual_factorization and dominant_domain in DOMAIN_OPS:
        preferred = DOMAIN_OPS[dominant_domain]
        head = [op for op in ops if op.name in preferred]
        tail = [op for op in ops if op.name not in preferred]
        # Keep inferred colormap first even after domain partition.
        cmap_ops = [op for op in head + tail if op.name in {"apply_colormap", "abs"}]
        rest = [op for op in head + tail if op.name not in {"apply_colormap", "abs"}]
        return cmap_ops + rest
    return ops


def operator_catalog() -> list[dict]:
    rows = []
    for name, d in sorted(OP_DEFS.items()):
        rows.append(
            {
                "name": name,
                "in_types": [t.value for t in d.in_types],
                "out_type": d.out_type.value,
                "frame_effect": d.frame_effect.value,
                "cost": d.cost,
                "kind": d.kind.value,
                "preconditions": d.preconditions,
            }
        )
    return rows

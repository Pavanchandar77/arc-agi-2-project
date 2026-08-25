"""Joint residuals across demonstrations.

Pixel residual: exact verifier.
Object/relation residuals: sound but incomplete structural diagnoses.
Shape/control residual: exact size/count mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.hrps.grid import Grid, pixel_mismatch, shape
from src.hrps.kinds import Kind
from src.hrps.representation import (
    Object,
    Representation,
    RepresentationSpec,
    build_representation,
    correspondence_ambiguity,
)


@dataclass(frozen=True)
class PixelResidual:
    kind: Kind = Kind.EXACT
    mismatched_cells: int = 0
    shape_ok: bool = True
    compared_cells: int = 0

    @property
    def solved(self) -> bool:
        return self.shape_ok and self.mismatched_cells == 0

    def as_dict(self) -> dict:
        return {
            "mismatched_cells": self.mismatched_cells,
            "shape_ok": self.shape_ok,
            "compared_cells": self.compared_cells,
            "solved": self.solved,
        }


@dataclass(frozen=True)
class ObjectResidual:
    kind: Kind = Kind.SOUND_INCOMPLETE
    n_pred: int = 0
    n_gt: int = 0
    unmatched: int = 0
    correspondence_ambiguity: int = 0

    def as_dict(self) -> dict:
        return {
            "n_pred": self.n_pred,
            "n_gt": self.n_gt,
            "unmatched": self.unmatched,
            "correspondence_ambiguity": self.correspondence_ambiguity,
        }


@dataclass(frozen=True)
class RelationResidual:
    kind: Kind = Kind.SOUND_INCOMPLETE
    n_pred: int = 0
    n_gt: int = 0
    symmetric_difference: int = 0

    def as_dict(self) -> dict:
        return {
            "n_pred": self.n_pred,
            "n_gt": self.n_gt,
            "symmetric_difference": self.symmetric_difference,
        }


@dataclass(frozen=True)
class ShapeControlResidual:
    kind: Kind = Kind.EXACT
    pred_hw: tuple[int, int] = (0, 0)
    gt_hw: tuple[int, int] = (0, 0)
    size_mismatch: bool = False
    object_count_delta: int = 0

    def as_dict(self) -> dict:
        return {
            "pred_hw": list(self.pred_hw),
            "gt_hw": list(self.gt_hw),
            "size_mismatch": self.size_mismatch,
            "object_count_delta": self.object_count_delta,
        }


@dataclass(frozen=True)
class PairResidual:
    pixel: PixelResidual
    objects: ObjectResidual
    relations: RelationResidual
    shape: ShapeControlResidual
    demos_exact: bool

    def as_dict(self) -> dict:
        return {
            "pixel": self.pixel.as_dict(),
            "objects": self.objects.as_dict(),
            "relations": self.relations.as_dict(),
            "shape": self.shape.as_dict(),
            "demos_exact": self.demos_exact,
        }


@dataclass
class JointResidual:
    """Aggregated residual over all demonstrations. Kind: exact on pixels."""

    kind: Kind = Kind.EXACT
    pairs: tuple[PairResidual, ...] = ()
    pixel_total: int = 0
    n_exact: int = 0
    n_demos: int = 0
    any_shape_mismatch: bool = False
    object_unmatched_total: int = 0
    relation_diff_total: int = 0
    object_count_delta_abs: int = 0
    dominant_domain: str = "pixel"

    @property
    def all_exact(self) -> bool:
        return self.n_demos > 0 and self.n_exact == self.n_demos

    def as_dict(self) -> dict:
        return {
            "pixel_total": self.pixel_total,
            "n_exact": self.n_exact,
            "n_demos": self.n_demos,
            "all_exact": self.all_exact,
            "any_shape_mismatch": self.any_shape_mismatch,
            "object_unmatched_total": self.object_unmatched_total,
            "relation_diff_total": self.relation_diff_total,
            "object_count_delta_abs": self.object_count_delta_abs,
            "dominant_domain": self.dominant_domain,
        }


def _quantize_rel(a: Object, b: Object) -> tuple:
    ar, ac = a.centroid
    br, bc = b.centroid
    dy = 0 if abs(ar - br) < 0.5 else (1 if ar < br else -1)
    dx = 0 if abs(ac - bc) < 0.5 else (1 if ac < bc else -1)
    overlap = not (
        a.bbox[2] < b.bbox[0]
        or b.bbox[2] < a.bbox[0]
        or a.bbox[3] < b.bbox[1]
        or b.bbox[3] < a.bbox[1]
    )
    same = a.color == b.color
    return (dx, dy, int(same), int(overlap), a.color, b.color)


def _relation_set(objects: tuple[Object, ...], k: int = 6) -> frozenset:
    objs = objects[:k]
    rels = []
    for i, a in enumerate(objs):
        for b in objs[i + 1 :]:
            rels.append(_quantize_rel(a, b))
    return frozenset(rels)


def _greedy_unmatched(pred: tuple[Object, ...], gt: tuple[Object, ...]) -> int:
    used = [False] * len(gt)
    matched = 0
    for p in pred:
        best = -1
        for i, g in enumerate(gt):
            if used[i]:
                continue
            if p.match_key() == g.match_key():
                best = i
                break
        if best >= 0:
            used[best] = True
            matched += 1
    return max(len(pred), len(gt)) - matched


def pair_residual(
    pred: Optional[Grid],
    gt: Grid,
    spec: Optional[RepresentationSpec] = None,
) -> PairResidual:
    miss, shape_ok, compared = pixel_mismatch(pred, gt)
    pix = PixelResidual(mismatched_cells=miss, shape_ok=shape_ok, compared_cells=compared)
    pred_hw = shape(pred) if pred is not None else (0, 0)
    gt_hw = shape(gt)
    if spec is None or pred is None:
        obj_r = ObjectResidual()
        rel_r = RelationResidual()
        n_pred = n_gt = 0
    else:
        pr = build_representation(pred, spec)
        gr = build_representation(gt, spec)
        n_pred, n_gt = pr.n_objects, gr.n_objects
        obj_r = ObjectResidual(
            n_pred=n_pred,
            n_gt=n_gt,
            unmatched=_greedy_unmatched(pr.objects, gr.objects),
            correspondence_ambiguity=correspondence_ambiguity(gr.objects),
        )
        rp, rg = _relation_set(pr.objects), _relation_set(gr.objects)
        rel_r = RelationResidual(
            n_pred=len(rp),
            n_gt=len(rg),
            symmetric_difference=len(rp.symmetric_difference(rg)),
        )
    shp = ShapeControlResidual(
        pred_hw=pred_hw,
        gt_hw=gt_hw,
        size_mismatch=pred_hw != gt_hw,
        object_count_delta=n_pred - n_gt,
    )
    return PairResidual(
        pixel=pix,
        objects=obj_r,
        relations=rel_r,
        shape=shp,
        demos_exact=pix.solved,
    )


def _dominant_domain(joint: JointResidual) -> str:
    if joint.all_exact:
        return "none"
    if joint.any_shape_mismatch:
        return "shape"
    if joint.object_count_delta_abs > 0 or joint.object_unmatched_total > 0:
        return "object"
    if joint.relation_diff_total > 0:
        return "relation"
    return "pixel"


def joint_residual(
    preds: tuple[Optional[Grid], ...],
    gts: tuple[Grid, ...],
    spec: Optional[RepresentationSpec] = None,
) -> JointResidual:
    pairs = tuple(pair_residual(p, g, spec) for p, g in zip(preds, gts))
    jr = JointResidual(
        pairs=pairs,
        pixel_total=sum(p.pixel.mismatched_cells for p in pairs),
        n_exact=sum(1 for p in pairs if p.demos_exact),
        n_demos=len(pairs),
        any_shape_mismatch=any(p.shape.size_mismatch for p in pairs),
        object_unmatched_total=sum(p.objects.unmatched for p in pairs),
        relation_diff_total=sum(p.relations.symmetric_difference for p in pairs),
        object_count_delta_abs=sum(abs(p.shape.object_count_delta) for p in pairs),
    )
    jr.dominant_domain = _dominant_domain(jr)
    return jr

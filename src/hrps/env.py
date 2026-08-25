"""HRPS environment: structured actions, exact execution, gold-free feedback.

The open model is the reasoner. This module is the environment: it never
decides the semantic rule. Test labels are not shown during the loop;
they are used only by the elevation harness after commit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.hrps.abstractions import AbstractionLibrary, parse_program_text, set_active_library
from src.hrps.dsl import OP_DEFS, Program, infer_colormap, operator_catalog, replay
from src.hrps.grid import (
    Grid,
    colors_present,
    counts,
    shape,
    to_lists,
)
from src.hrps.kinds import Kind
from src.hrps.representation import BANK_D, build_representation
from src.hrps.residual import JointResidual, joint_residual
from src.hrps.separability import _corner_nonzero, _singleton_cells, colormap_from_program
from src.hrps.task import ArcTask

LEGAL_OP_NAMES = frozenset(OP_DEFS) | {"abs", "identity"}
_MAX_DEPTH = 3


def grid_to_compact(grid: Grid) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in grid)


def serialize_task_raw(task: ArcTask) -> str:
    """Raw demonstrations and test inputs. No test outputs."""
    parts = [f"TASK {task.task_id}", f"n_train={task.n_train} n_test={task.n_test}"]
    for i, p in enumerate(task.train):
        parts.append(f"DEMO {i} INPUT {shape(p.input)[0]}x{shape(p.input)[1]}")
        parts.append(grid_to_compact(p.input))
        parts.append(f"DEMO {i} OUTPUT {shape(p.output)[0]}x{shape(p.output)[1]}")  # type: ignore[arg-type]
        parts.append(grid_to_compact(p.output))  # type: ignore[arg-type]
    for i, p in enumerate(task.test):
        parts.append(f"TEST {i} INPUT {shape(p.input)[0]}x{shape(p.input)[1]}")
        parts.append(grid_to_compact(p.input))
    return "\n".join(parts)


def parse_program(text: str) -> tuple[Optional[Program], str]:
    """Parse a DSL program. Reject unknown operators. Kind: exact."""
    raw = text.strip()
    raw = raw.strip("`").strip()
    if raw.lower() in {"identity", ""}:
        return Program(()), ""
    # Keep only the first program-looking line cluster.
    if "\n" in raw:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.lower().startswith(("action", "hypothesis", "inspect")):
                continue
            raw = line
            break
    try:
        ops = parse_program_text(raw)
    except Exception as exc:
        return None, f"parse_error:{exc}"
    if len(ops) > _MAX_DEPTH:
        return None, f"depth_exceeded:{len(ops)}>{_MAX_DEPTH}"
    for op in ops:
        if op.name not in LEGAL_OP_NAMES:
            return None, f"unknown_op:{op.name}"
        if op.name != "abs" and op.name not in OP_DEFS and op.name != "identity":
            return None, f"unknown_op:{op.name}"
    return Program(ops), ""


def gold_free_constraint_feedback(task: ArcTask, program: Program) -> dict[str, Any]:
    """Underconstraint evidence that does not use test labels.

    Kind: exact. Test gold is never read. Test *inputs* may be inspected
    (colors, singletons, corners) because those are part of the puzzle.
    """
    program_text = program.serialize()
    mapping = colormap_from_program(program_text)
    demo_in = [set(colors_present(p.input)) for p in task.train]
    test_in = [set(colors_present(p.input)) for p in task.test]
    train_in_union: set[int] = set().union(*demo_in) if demo_in else set()
    test_unseen = sorted(set().union(*test_in) - train_in_union) if test_in else []

    per_demo_maps = []
    disjoint_support = True
    used: set[int] = set()
    for p in task.train:
        local = infer_colormap((p.input,), (p.output,))  # type: ignore[arg-type]
        srcs: set[int] = set()
        if local:
            srcs = {a for a, b in local if a != b}
            if srcs & used:
                disjoint_support = False
            used |= srcs
        per_demo_maps.append(
            {
                "map": [[a, b] for a, b in (local or ())],
                "input_colors": sorted(colors_present(p.input)),
                "output_colors": sorted(colors_present(p.output)),  # type: ignore[arg-type]
                "singletons": _singleton_cells(p.input),
                "corner_nonzero": _corner_nonzero(p.input),
            }
        )

    test_input_views = []
    for p in task.test:
        pred = replay(program, p.input)
        test_input_views.append(
            {
                "input_colors": sorted(colors_present(p.input)),
                "pred_shape": list(shape(pred)) if pred is not None else None,
                "pred_colors": sorted(colors_present(pred)) if pred is not None else None,
                "pred_valid": pred is not None,
                "singletons": _singleton_cells(p.input),
                "corner_nonzero": _corner_nonzero(p.input),
                "unseen_input_colors": sorted(set(colors_present(p.input)) - train_in_union),
            }
        )

    singleton_colors: set[int] = set()
    blob_colors: set[int] = set()
    for p in list(task.train) + list(task.test):
        sc = {v for _, _, v in _singleton_cells(p.input)}
        allc = set(colors_present(p.input)) - {0}
        singleton_colors |= sc
        blob_colors |= allc - sc
    role_collision = sorted(singleton_colors & blob_colors)

    flags: list[str] = []
    if mapping is not None and disjoint_support:
        flags.append("joint_map_is_union_of_disjoint_per_demo_palettes")
    if test_unseen:
        flags.append("test_introduces_unseen_input_colors")
    if role_collision:
        flags.append("color_plays_marker_role_in_one_pair_and_blob_role_in_another")
    if mapping is not None and flags:
        flags.append("jointly_exact_colormap_may_be_underconstrained")

    return {
        "kind": Kind.EXACT.value,
        "program": program_text,
        "inferred_joint_map": [[k, mapping[k]] for k in sorted(mapping)] if mapping else None,
        "disjoint_per_demo_color_support": disjoint_support and mapping is not None,
        "per_demo_maps": per_demo_maps,
        "test_unseen_input_colors": test_unseen,
        "role_collision_colors": role_collision,
        "test_input_replay": test_input_views,
        "underconstraint_flags": flags,
        "uses_test_labels": False,
    }


@dataclass
class Action:
    kind: str  # observe, inspect, apply, residual, commit, hypothesize, catalog, answer
    payload: str = ""
    raw: str = ""


@dataclass
class EnvFeedback:
    accepted: bool
    action: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    program: Program
    residual: JointResidual
    joint_exact: bool
    source: str  # apply | commit | answer


class HrpsEnv:
    """Stateful environment for one task. Test labels stay hidden."""

    def __init__(
        self,
        task: ArcTask,
        *,
        library: Optional[AbstractionLibrary] = None,
        enable_h: bool = False,
        max_depth: int = _MAX_DEPTH,
    ) -> None:
        self.task = task
        self.library = library or AbstractionLibrary()
        self.enable_h = enable_h
        self.max_depth = max_depth
        if enable_h:
            set_active_library(self.library)
        self.history: list[EnvFeedback] = []
        self.hypotheses: list[str] = []
        self.candidates: list[Candidate] = []
        self.committed: list[Program] = []
        self.answer_attempts: list[list[list[list[int]]]] = []
        self.n_inspect = 0
        self.n_apply = 0
        self.n_commit = 0
        self.n_hypothesize = 0
        self.n_accepted = 0
        self.n_rejected = 0
        self.n_contradiction_resolutions = 0
        self.residual_trace: list[int] = []
        self.valid_actions = 0
        self.total_actions = 0
        self._last_residual: Optional[JointResidual] = None
        self._last_program: Optional[Program] = None

    def observe(self) -> EnvFeedback:
        fb = EnvFeedback(
            accepted=True,
            action="observe",
            text=serialize_task_raw(self.task),
            data={"task_id": self.task.task_id, "n_train": self.task.n_train, "n_test": self.task.n_test},
        )
        self.history.append(fb)
        return fb

    def catalog_text(self) -> str:
        lines = [
            "HRPS DSL (compose with ' | ', depth <= 3). Execution is exact.",
            "Unary geom: rot90 rot180 rot270 flip_h flip_v transpose anti_transpose",
            "Crop/size: crop_fg:<bg> left_half right_half top_half bottom_half tile:<nr>x<nc> upscale:<k> downscale:<k>",
            "Color: recolor:<src>,<dst> swap_colors:<a>,<b> keep_color:<c>,<bg> apply_colormap:<a-b;c-d;...>",
            "Objects: fill_holes:<bg> outline:<bg> gravity:<dir>,<bg> isolate_largest:<conn>,<t|f>,<bg> isolate_smallest:<conn>,<t|f>,<bg>",
            "conn is 4 or 8; t/f is color-agnostic; dir 0=N 1=E 2=S 3=W; colors 0-9.",
        ]
        if self.enable_h and self.library.items:
            lines.append("Named exact abstractions (training-only macros):")
            for abs_ in self.library.items:
                lines.append(f"  abs:{abs_.name}  body={abs_.body_serialize()}")
        lines.append("Actions: INSPECT <colors|shapes|objects|catalog|underconstraint>")
        lines.append("         APPLY <program>")
        lines.append("         HYPOTHESIZE <text>")
        lines.append("         COMMIT <program>")
        return "\n".join(lines)

    def step(self, action: Action) -> EnvFeedback:
        self.total_actions += 1
        kind = action.kind.lower().strip()
        if kind == "observe":
            fb = self.observe()
            self.valid_actions += 1
            self.n_accepted += 1
            return fb
        if kind == "catalog":
            fb = EnvFeedback(True, "catalog", self.catalog_text(), {"n_ops": len(operator_catalog())})
            self._accept(fb)
            return fb
        if kind == "hypothesize":
            return self._hypothesize(action.payload)
        if kind == "inspect":
            return self._inspect(action.payload)
        if kind == "apply":
            return self._apply(action.payload)
        if kind == "residual":
            return self._residual()
        if kind == "commit":
            return self._commit(action.payload)
        if kind == "answer":
            return self._answer(action.payload)
        fb = EnvFeedback(False, kind, f"rejected:unknown_action:{kind}", {"error": "unknown_action"})
        self.n_rejected += 1
        self.history.append(fb)
        return fb

    def _accept(self, fb: EnvFeedback) -> None:
        self.valid_actions += 1
        self.n_accepted += 1
        self.history.append(fb)

    def _reject(self, action: str, reason: str, extra: Optional[dict] = None) -> EnvFeedback:
        data = {"error": reason}
        if extra:
            data.update(extra)
        fb = EnvFeedback(False, action, f"rejected:{reason}", data)
        self.n_rejected += 1
        self.history.append(fb)
        return fb

    def _hypothesize(self, text: str) -> EnvFeedback:
        text = text.strip()
        if not text:
            return self._reject("hypothesize", "empty_hypothesis")
        prev = self.hypotheses[-1] if self.hypotheses else None
        self.hypotheses.append(text)
        self.n_hypothesize += 1
        fb = EnvFeedback(
            True,
            "hypothesize",
            f"recorded hypothesis #{len(self.hypotheses)}: {text}",
            {"n": len(self.hypotheses), "revised": prev is not None and prev != text},
        )
        self._accept(fb)
        return fb

    def _inspect(self, what: str) -> EnvFeedback:
        token = (what or "colors").strip().split()[0].lower()
        self.n_inspect += 1
        if token in {"catalog", "ops", "operators"}:
            fb = EnvFeedback(True, "inspect", self.catalog_text(), {"what": "catalog"})
            self._accept(fb)
            return fb
        if token in {"underconstraint", "constraint"}:
            prog = self._last_program or Program(())
            data = gold_free_constraint_feedback(self.task, prog)
            fb = EnvFeedback(True, "inspect", _format_constraint(data), data)
            self._accept(fb)
            return fb
        if token in {"colors", "palette"}:
            data = self._inspect_colors()
            fb = EnvFeedback(True, "inspect", _format_colors(data), data)
            self._accept(fb)
            return fb
        if token in {"shapes", "shape", "size"}:
            data = self._inspect_shapes()
            fb = EnvFeedback(True, "inspect", _format_shapes(data), data)
            self._accept(fb)
            return fb
        if token in {"objects", "object"}:
            spec_id = "4c_zero"
            rest = what.strip().split()
            if len(rest) >= 2:
                spec_id = rest[1].split("=", 1)[-1]
            data = self._inspect_objects(spec_id)
            fb = EnvFeedback(True, "inspect", _format_objects(data), data)
            self._accept(fb)
            return fb
        return self._reject("inspect", f"unknown_inspect:{token}")

    def _inspect_colors(self) -> dict[str, Any]:
        demos = []
        for i, p in enumerate(self.task.train):
            demos.append(
                {
                    "i": i,
                    "in": sorted(colors_present(p.input)),
                    "out": sorted(colors_present(p.output)),  # type: ignore[arg-type]
                    "in_counts": list(counts(p.input)),
                    "out_counts": list(counts(p.output)),  # type: ignore[arg-type]
                    "singletons_in": _singleton_cells(p.input),
                    "corners_in": _corner_nonzero(p.input),
                }
            )
        tests = []
        for i, p in enumerate(self.task.test):
            tests.append(
                {
                    "i": i,
                    "in": sorted(colors_present(p.input)),
                    "in_counts": list(counts(p.input)),
                    "singletons_in": _singleton_cells(p.input),
                    "corners_in": _corner_nonzero(p.input),
                }
            )
        train_in = set().union(*(set(d["in"]) for d in demos)) if demos else set()
        unseen = sorted(set().union(*(set(t["in"]) for t in tests)) - train_in) if tests else []
        return {"demos": demos, "tests": tests, "test_unseen_input_colors": unseen}

    def _inspect_shapes(self) -> dict[str, Any]:
        demos = []
        for i, p in enumerate(self.task.train):
            demos.append(
                {
                    "i": i,
                    "in_hw": list(shape(p.input)),
                    "out_hw": list(shape(p.output)),  # type: ignore[arg-type]
                    "same_shape": shape(p.input) == shape(p.output),  # type: ignore[arg-type]
                }
            )
        tests = [{"i": i, "in_hw": list(shape(p.input))} for i, p in enumerate(self.task.test)]
        return {"demos": demos, "tests": tests}

    def _inspect_objects(self, spec_id: str) -> dict[str, Any]:
        spec = next((s for s in BANK_D if s.spec_id == spec_id), BANK_D[0])
        demos = []
        for i, p in enumerate(self.task.train):
            rin = build_representation(p.input, spec)
            rout = build_representation(p.output, spec)  # type: ignore[arg-type]
            demos.append(
                {
                    "i": i,
                    "spec": spec.spec_id,
                    "bg_in": rin.bg,
                    "n_in": rin.n_objects,
                    "n_out": rout.n_objects,
                    "in": [_obj_brief(o) for o in rin.objects[:8]],
                    "out": [_obj_brief(o) for o in rout.objects[:8]],
                }
            )
        tests = []
        for i, p in enumerate(self.task.test):
            rin = build_representation(p.input, spec)
            tests.append(
                {
                    "i": i,
                    "spec": spec.spec_id,
                    "bg_in": rin.bg,
                    "n_in": rin.n_objects,
                    "in": [_obj_brief(o) for o in rin.objects[:8]],
                }
            )
        return {"spec": spec.spec_id, "kind": Kind.SOUND_INCOMPLETE.value, "demos": demos, "tests": tests}

    def _execute_program(self, program: Program) -> tuple[tuple[Optional[Grid], ...], JointResidual]:
        preds: list[Optional[Grid]] = []
        for p in self.task.train:
            preds.append(replay(program, p.input))
        gts = self.task.train_outputs()
        residual = joint_residual(tuple(preds), gts, spec=None)
        return tuple(preds), residual

    def _apply(self, text: str) -> EnvFeedback:
        program, err = parse_program(text)
        if program is None:
            return self._reject("apply", err or "parse_error", {"program_text": text})
        if program.depth() > self.max_depth:
            return self._reject("apply", "depth_exceeded")
        if "abs" in program.names() and not self.enable_h:
            return self._reject("apply", "abstractions_disabled")
        self.n_apply += 1
        preds, residual = self._execute_program(program)
        constraint = gold_free_constraint_feedback(self.task, program)
        self._note_residual(residual)
        self._last_program = program
        cand = Candidate(program, residual, residual.all_exact, "apply")
        self.candidates.append(cand)
        data = {
            "program": program.serialize(),
            "residual": residual.as_dict(),
            "constraint": constraint,
            "executor_ok": all(p is not None for p in preds),
        }
        fb = EnvFeedback(True, "apply", _format_apply(data), data)
        self._accept(fb)
        return fb

    def _residual(self) -> EnvFeedback:
        if self._last_residual is None:
            return self._reject("residual", "no_program_applied")
        data = {
            "program": self._last_program.serialize() if self._last_program else "identity",
            "residual": self._last_residual.as_dict(),
        }
        fb = EnvFeedback(True, "residual", _format_residual(data), data)
        self._accept(fb)
        return fb

    def _commit(self, text: str) -> EnvFeedback:
        program, err = parse_program(text if text.strip() else (self._last_program.serialize() if self._last_program else "identity"))
        if program is None:
            return self._reject("commit", err or "parse_error")
        if "abs" in program.names() and not self.enable_h:
            return self._reject("commit", "abstractions_disabled")
        preds, residual = self._execute_program(program)
        self.n_apply += 1
        self._note_residual(residual)
        self._last_program = program
        self.candidates.append(Candidate(program, residual, residual.all_exact, "commit"))
        if residual.all_exact:
            self.committed.append(program)
            self.n_commit += 1
            msg = f"committed jointly exact program: {program.serialize()}"
            accepted = True
        else:
            # Still store as a candidate; model may commit a non-exact program.
            self.committed.append(program)
            self.n_commit += 1
            msg = (
                f"committed program that is NOT jointly exact: {program.serialize()} "
                f"pixel_total={residual.pixel_total} n_exact={residual.n_exact}/{residual.n_demos}"
            )
            accepted = True
        constraint = gold_free_constraint_feedback(self.task, program)
        data = {
            "program": program.serialize(),
            "residual": residual.as_dict(),
            "joint_exact": residual.all_exact,
            "constraint": constraint,
            "n_committed": len(self.committed),
        }
        fb = EnvFeedback(accepted, "commit", msg + "\n" + _format_constraint(constraint), data)
        self._accept(fb)
        return fb

    def _answer(self, text: str) -> EnvFeedback:
        """M0: model emits grids rather than programs. No test labels used."""
        grids = parse_answer_grids(text, n_test=self.task.n_test)
        if not grids:
            return self._reject("answer", "no_valid_grid")
        self.answer_attempts.append(grids)
        fb = EnvFeedback(
            True,
            "answer",
            f"recorded {len(grids)} test grid(s) as attempt {len(self.answer_attempts)}",
            {"n_grids": len(grids), "attempt_index": len(self.answer_attempts)},
        )
        self._accept(fb)
        return fb

    def _note_residual(self, residual: JointResidual) -> None:
        prev = self._last_residual
        self._last_residual = residual
        self.residual_trace.append(residual.pixel_total)
        if prev is not None:
            improved = residual.pixel_total < prev.pixel_total or residual.n_exact > prev.n_exact
            partial = 0 < prev.n_exact < prev.n_demos
            if improved and (partial or prev.pixel_total > 0):
                self.n_contradiction_resolutions += 1

    def distinct_hypotheses(self) -> int:
        return len(set(self.hypotheses))

    def hypothesis_revisions(self) -> int:
        n = 0
        for a, b in zip(self.hypotheses, self.hypotheses[1:]):
            if a != b:
                n += 1
        return n

    def valid_action_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.valid_actions / self.total_actions

    def finalize_programs(self) -> list[Program]:
        """Pick up to two programs for competition outputs.

        Prefer jointly exact commits, then jointly exact applies, then
        best residual. Duplicate if only one exists. Identity only if empty.
        """
        seen: set[str] = set()
        ordered: list[Program] = []

        def _add(prog: Program) -> None:
            key = prog.serialize()
            if key in seen:
                return
            seen.add(key)
            ordered.append(prog)

        for prog in self.committed:
            _add(prog)
        exact = [c for c in self.candidates if c.joint_exact]
        exact.sort(key=lambda c: (c.program.cost(), c.program.serialize()))
        for c in exact:
            _add(c.program)
        rest = [c for c in self.candidates if not c.joint_exact]
        rest.sort(key=lambda c: (c.residual.pixel_total, -c.residual.n_exact, c.program.cost()))
        for c in rest:
            _add(c.program)
        if not ordered:
            ordered.append(Program(()))
        if len(ordered) == 1:
            ordered.append(ordered[0])
        return ordered[:2]

    def finalize_attempts(self) -> list[list[list[list[int]]]]:
        """Two attempts, each a list of per-test grids. Kind: exact replay."""
        if self.answer_attempts:
            attempts = list(self.answer_attempts[:2])
            while len(attempts) < 2:
                attempts.append(attempts[0])
            # Pad missing test slots with [[0]].
            n_test = self.task.n_test
            padded = []
            for att in attempts:
                grids = list(att)
                while len(grids) < n_test:
                    grids.append([[0]])
                padded.append(grids[:n_test])
            return padded
        programs = self.finalize_programs()
        attempts: list[list[list[list[int]]]] = []
        for prog in programs:
            per_test = []
            for inp in self.task.test_inputs():
                pred = replay(prog, inp)
                per_test.append(to_lists(pred) if pred is not None else [[0]])
            attempts.append(per_test)
        return attempts


def _obj_brief(obj) -> dict[str, Any]:
    r0, c0, r1, c1 = obj.bbox
    cr, cc = obj.centroid
    return {
        "color": obj.color,
        "area": obj.area,
        "bbox": [r0, c0, r1, c1],
        "centroid": [round(cr, 2), round(cc, 2)],
    }


def _format_colors(data: dict[str, Any]) -> str:
    lines = ["INSPECT colors (test outputs hidden)"]
    for d in data["demos"]:
        lines.append(
            f"  demo{d['i']}: in={d['in']} out={d['out']} singletons={d['singletons_in']} corners={d['corners_in']}"
        )
    for t in data["tests"]:
        lines.append(
            f"  test{t['i']}_input: in={t['in']} singletons={t['singletons_in']} corners={t['corners_in']}"
        )
    if data["test_unseen_input_colors"]:
        lines.append(f"  test_unseen_input_colors={data['test_unseen_input_colors']}")
    return "\n".join(lines)


def _format_shapes(data: dict[str, Any]) -> str:
    lines = ["INSPECT shapes"]
    for d in data["demos"]:
        lines.append(f"  demo{d['i']}: {d['in_hw']} -> {d['out_hw']} same={d['same_shape']}")
    for t in data["tests"]:
        lines.append(f"  test{t['i']}_input: {t['in_hw']}")
    return "\n".join(lines)


def _format_objects(data: dict[str, Any]) -> str:
    lines = [f"INSPECT objects spec={data['spec']} (connected components are proxies, not semantics)"]
    for d in data["demos"]:
        lines.append(f"  demo{d['i']}: n_in={d['n_in']} n_out={d['n_out']} bg={d['bg_in']}")
        for o in d["in"][:4]:
            lines.append(f"    in color={o['color']} area={o['area']} bbox={o['bbox']}")
    for t in data["tests"]:
        lines.append(f"  test{t['i']}_input: n_in={t['n_in']}")
        for o in t["in"][:4]:
            lines.append(f"    in color={o['color']} area={o['area']} bbox={o['bbox']}")
    return "\n".join(lines)


def _format_apply(data: dict[str, Any]) -> str:
    r = data["residual"]
    flags = data["constraint"].get("underconstraint_flags") or []
    lines = [
        f"APPLY {data['program']}",
        f"  joint_exact={r['all_exact']} n_exact={r['n_exact']}/{r['n_demos']} "
        f"pixel_total={r['pixel_total']} shape_mismatch={r['any_shape_mismatch']} domain={r['dominant_domain']}",
        f"  executor_ok={data['executor_ok']}",
    ]
    if flags:
        lines.append("  underconstraint: " + "; ".join(flags))
    unseen = data["constraint"].get("test_unseen_input_colors") or []
    if unseen:
        lines.append(f"  test_unseen_input_colors={unseen}")
    roles = data["constraint"].get("role_collision_colors") or []
    if roles:
        lines.append(f"  role_collision_colors={roles}")
    if r["all_exact"]:
        lines.append(
            "  note: jointly exact on demonstrations is necessary but not sufficient for test transfer."
        )
    return "\n".join(lines)


def _format_residual(data: dict[str, Any]) -> str:
    r = data["residual"]
    return (
        f"RESIDUAL program={data['program']} joint_exact={r['all_exact']} "
        f"n_exact={r['n_exact']}/{r['n_demos']} pixel_total={r['pixel_total']}"
    )


def _format_constraint(data: dict[str, Any]) -> str:
    lines = [
        f"CONSTRAINT program={data['program']} uses_test_labels={data['uses_test_labels']}",
        f"  disjoint_per_demo_color_support={data['disjoint_per_demo_color_support']}",
        f"  test_unseen_input_colors={data['test_unseen_input_colors']}",
        f"  role_collision_colors={data['role_collision_colors']}",
    ]
    flags = data.get("underconstraint_flags") or []
    if flags:
        lines.append("  flags: " + "; ".join(flags))
    return "\n".join(lines)


_ACTION_HEAD = re.compile(
    r"^\s*(?:ACTION\s+)?(?P<kind>inspect|apply|commit|hypothesize|hypothesis|residual|catalog|observe|answer)\b"
    r"(?:\s*(?::|=|\s)\s*(?P<rest>.*))?$",
    re.I,
)


def parse_model_actions(text: str) -> list[Action]:
    """Extract environment actions from a model turn. Sound but incomplete parser."""
    if not text or not isinstance(text, str):
        return []
    actions: list[Action] = []
    # JSON object?
    stripped = text.strip()
    if stripped.startswith("{") and "action" in stripped.lower():
        try:
            import json

            blob = json.loads(stripped)
            kind = str(blob.get("action") or blob.get("kind") or "").lower()
            payload = str(
                blob.get("program")
                or blob.get("what")
                or blob.get("text")
                or blob.get("payload")
                or blob.get("grid")
                or ""
            )
            if kind:
                actions.append(Action(kind=kind, payload=payload, raw=stripped))
                return actions
        except Exception:
            pass
    for line in text.splitlines():
        m = _ACTION_HEAD.match(line.strip())
        if not m:
            continue
        kind = m.group("kind").lower()
        if kind == "hypothesis":
            kind = "hypothesize"
        rest = (m.group("rest") or "").strip()
        if rest.lower().startswith("program="):
            rest = rest.split("=", 1)[1].strip()
        if rest.lower().startswith("what="):
            rest = rest.split("=", 1)[1].strip()
        actions.append(Action(kind=kind, payload=rest, raw=line))
    if actions:
        # APPLY/COMMIT payload may be the rest of the turn after the header.
        if actions[-1].kind in {"apply", "commit", "hypothesize", "answer"} and not actions[-1].payload:
            # take remaining text after the matching line
            idx = text.lower().find(actions[-1].kind)
            after = text[idx:].split("\n", 1)
            if len(after) > 1:
                actions[-1].payload = after[1].strip()
        return actions
    # Bare program?
    prog, err = parse_program(stripped.splitlines()[0] if stripped else "")
    if prog is not None and not err and prog.ops:
        return [Action(kind="apply", payload=prog.serialize(), raw=stripped)]
    return []


def parse_answer_grids(text: str, n_test: int = 1) -> list[list[list[int]]]:
    from src.data import is_valid_grid, text_to_grid

    grids: list[list[list[int]]] = []
    g = text_to_grid(text)
    if g is not None and is_valid_grid(g):
        grids.append(g)
    # Multiple fenced blocks.
    if len(grids) < n_test:
        blocks = re.findall(r"```(?:json|python|grid)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        for block in blocks:
            g2 = text_to_grid(block)
            if g2 is not None and is_valid_grid(g2) and g2 not in grids:
                grids.append(g2)
    return grids[:n_test] if grids else []

"""Instrumented finite-DSL search microscope.

Stages A–G share this engine, the DSL, the executor, and the output policy.
Search order uses residual then description length (heuristic).
The remaining-cost bound is admissible for description length: 0 if solved,
else MIN_OP_COST. It is used for branch-and-bound only after a feasible
joint solution exists.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.hrps.dsl import (
    GEOM_UNARY,
    MIN_OP_COST,
    OP_DEFS,
    Program,
    SearchStageConfig,
    execute_op,
    generate_ops,
    replay,
    stage_config,
)
from src.hrps.failure import classify_failure
from src.hrps.grid import Grid, as_grid, colors_present, shape, to_lists
from src.hrps.kinds import Kind
from src.hrps.representation import (
    SPEC_4_ZERO,
    SPEC_8_ZERO,
    SPEC_8A_ZERO,
    RepresentationSpec,
    build_representation,
    correspondence_ambiguity,
    instability_across,
)

# Instability bank for telemetry. Smaller than BANK_D so stats cannot dominate
# the per-task wall clock; still includes 4-conn, 8-conn, and color-agnostic.
_STAT_SPECS = (SPEC_4_ZERO, SPEC_8_ZERO, SPEC_8A_ZERO)
from src.hrps.residual import JointResidual, joint_residual
from src.hrps.signature import continuation_signature, legal_op_family
from src.hrps.task import ArcTask


def _rss_bytes() -> Optional[int]:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:
            return None


@dataclass
class SearchBudget:
    max_depth: int = 3
    max_nodes: int = 500
    max_seconds: float = 1.5
    max_frontier: int = 4000
    max_ops_per_node: int = 40


@dataclass
class CandidateRecord:
    program: Program
    preds: tuple[Grid, ...]
    residual: JointResidual
    cost: int
    n_exact: int
    rejection: str
    node_id: int

    def diagnostics(self) -> dict[str, Any]:
        return {
            "program": self.program.serialize(),
            "cost": self.cost,
            "depth": self.program.depth(),
            "demonstrations_passed": self.n_exact,
            "pixel_residual": self.residual.pixel_total,
            "object_residual": self.residual.object_unmatched_total,
            "relation_residual": self.residual.relation_diff_total,
            "shape_control_residual": self.residual.object_count_delta_abs,
            "output_shape_status": "mismatch" if self.residual.any_shape_mismatch else "ok",
            "rejection_reason": self.rejection,
            "dominant_domain": self.residual.dominant_domain,
        }


@dataclass
class SearchTelemetry:
    nodes_generated: int = 0
    nodes_expanded: int = 0
    frontier_peak: int = 0
    unique_signatures: int = 0
    duplicate_states: int = 0
    consistency_prunes: int = 0
    dominance_prunes: int = 0
    bound_prunes: int = 0
    executor_rejects: int = 0
    noop_rejects: int = 0
    raw_states: int = 0
    representation_hypotheses: int = 0
    time_to_first_exact_demo_solution: Optional[float] = None
    time_to_first_verified_test_candidate: Optional[float] = None
    rss_bytes: Optional[int] = None
    timed_out: bool = False
    hit_node_limit: bool = False
    hit_frontier_limit: bool = False
    enumerated_exhausted: bool = False
    best_pixel_residual: Optional[int] = None
    best_n_exact: int = 0
    bound_value: int = MIN_OP_COST
    bound_tightness: Optional[float] = None
    nesting_depth: int = 0
    description_length: Optional[int] = None
    quotient_ratio: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        qratio = None
        if self.raw_states:
            qratio = round(self.unique_signatures / self.raw_states, 6)
        return {
            "nodes_generated": self.nodes_generated,
            "nodes_expanded": self.nodes_expanded,
            "frontier_size_peak": self.frontier_peak,
            "unique_states": self.unique_signatures,
            "duplicate_states": self.duplicate_states,
            "raw_states": self.raw_states,
            "quotient_ratio": qratio if qratio is not None else self.quotient_ratio,
            "consistency_prunes": self.consistency_prunes,
            "dominance_prunes": self.dominance_prunes,
            "bound_prunes": self.bound_prunes,
            "executor_rejects": self.executor_rejects,
            "representation_hypotheses": self.representation_hypotheses,
            "runtime_ready": True,
            "time_to_first_exact_demonstration_solution": self.time_to_first_exact_demo_solution,
            "time_to_first_verified_test_candidate": self.time_to_first_verified_test_candidate,
            "memory_rss_bytes": self.rss_bytes,
            "timed_out": self.timed_out,
            "hit_node_limit": self.hit_node_limit,
            "hit_frontier_limit": self.hit_frontier_limit,
            "enumerated_exhausted": self.enumerated_exhausted,
            "best_pixel_residual": self.best_pixel_residual,
            "best_n_exact": self.best_n_exact,
            "bound_value": self.bound_value,
            "bound_tightness": self.bound_tightness,
            "nesting_depth": self.nesting_depth,
            "description_length": self.description_length,
        }


@dataclass
class _Node:
    priority: tuple
    node_id: int
    program: Program
    preds: tuple[Grid, ...]
    residual: JointResidual
    cost: int
    spec: Optional[RepresentationSpec]

    def __lt__(self, other: "_Node") -> bool:
        return (self.priority, self.node_id) < (other.priority, other.node_id)


@dataclass
class TaskResult:
    task_id: str
    split: str
    stage: str
    solved: bool
    runtime: float
    attempts: list[list[list[list[int]]]]
    programs: list[str]
    failure_category: str
    telemetry: dict[str, Any]
    task_stats: dict[str, Any]
    candidate_summaries: list[dict[str, Any]]
    test_exact: list[bool]


def _task_stats(task: ArcTask, specs: tuple[RepresentationSpec, ...]) -> dict[str, Any]:
    areas = []
    colors = set()
    entity_counts: dict[str, list[int]] = {}
    rel_counts: dict[str, list[int]] = {}
    amb = []
    for p in task.train:
        ih, iw = shape(p.input)
        oh, ow = shape(p.output)  # type: ignore[arg-type]
        areas.extend([ih * iw, oh * ow])
        colors |= set(colors_present(p.input)) | set(colors_present(p.output))  # type: ignore[arg-type]
        for spec in _STAT_SPECS:
            rep_i = build_representation(p.input, spec)
            entity_counts.setdefault(spec.spec_id, []).append(rep_i.n_objects)
            rel_counts.setdefault(spec.spec_id, []).append(rep_i.relation_count())
            amb.append(correspondence_ambiguity(rep_i.objects))
    inst = [instability_across(v) for v in entity_counts.values()] if entity_counts else [0.0]
    return {
        "n_train": task.n_train,
        "n_test": task.n_test,
        "max_grid_area": max(areas) if areas else 0,
        "median_grid_area": sorted(areas)[len(areas) // 2] if areas else 0,
        "n_colors": len(colors),
        "test_multiplicity": task.n_test,
        "representation_hypotheses": len(specs) if specs else 1,
        "entity_counts_by_spec": {k: sorted(v) for k, v in entity_counts.items()},
        "relation_counts_by_spec": {k: sorted(v) for k, v in rel_counts.items()},
        "correspondence_ambiguity_max": max(amb) if amb else 0,
        "representation_instability": max(inst) if inst else 0.0,
    }


def _score(residual: JointResidual, cost: int, cfg: SearchStageConfig) -> tuple:
    # Heuristic frontier order. Admissible bound is applied separately.
    if cfg.score_joint:
        r = residual.pixel_total
    else:
        r = min((p.pixel.mismatched_cells for p in residual.pairs), default=residual.pixel_total)
    h = 0 if residual.all_exact else MIN_OP_COST  # admissible remaining-cost
    return (r, cost + h, cost, -residual.n_exact)


def _apply_program(program: Program, grids: tuple[Grid, ...], cache: dict) -> Optional[tuple[Grid, ...]]:
    out: list[Grid] = []
    for g in grids:
        cur = g
        for op in program.ops:
            key = (op.serialize(), cur)
            if key in cache:
                nxt = cache[key]
            else:
                nxt = execute_op(op, cur)
                cache[key] = nxt
            if nxt is None:
                return None
            cur = nxt
        out.append(cur)
    return tuple(out)


def _structurally_distinct(a: Program, b: Program) -> bool:
    return a.names() != b.names()


def search_task(
    task: ArcTask,
    stage: str = "G",
    budget: Optional[SearchBudget] = None,
    cfg: Optional[SearchStageConfig] = None,
) -> TaskResult:
    cfg = cfg or stage_config(stage)
    budget = budget or SearchBudget()
    t0 = time.perf_counter()
    tel = SearchTelemetry()
    cache: dict = {}
    train_in = task.train_inputs()
    train_out = task.train_outputs()
    specs = cfg.object_specs
    tel.representation_hypotheses = max(1, len(specs))
    stats = _task_stats(task, specs)
    family = None

    root_preds = train_in
    # Expansion residual is pixel/shape only. Object residuals are diagnoses;
    # using them as the root dominant domain starves geometry under the op cap.
    root_res = joint_residual(root_preds, train_out, spec=None)
    root = _Node(
        priority=_score(root_res, 0, cfg),
        node_id=0,
        program=Program(()),
        preds=root_preds,
        residual=root_res,
        cost=0,
        spec=specs[0] if specs else None,
    )
    heap: list[_Node] = [root]
    tel.raw_states = 1
    tel.best_pixel_residual = root_res.pixel_total
    tel.best_n_exact = root_res.n_exact
    tel.frontier_peak = 1

    seen: set[bytes] = set()
    # Seed signature of the identity state.
    dummy_ops = generate_ops(task, root_preds, train_out, cfg, root_res.dominant_domain)
    family = legal_op_family(op.name for op in dummy_ops)
    if cfg.continuation_dedup:
        seen.add(continuation_signature(root_preds, cfg.stage, family))
        tel.unique_signatures = 1

    verified: list[CandidateRecord] = []
    partials: list[CandidateRecord] = []
    next_id = 1
    best_feasible_cost = 10**9
    expanded_with_ops = 0

    def maybe_record(node: _Node, rejection: str) -> None:
        rec = CandidateRecord(
            program=node.program,
            preds=node.preds,
            residual=node.residual,
            cost=node.cost,
            n_exact=node.residual.n_exact,
            rejection=rejection,
            node_id=node.node_id,
        )
        if node.residual.all_exact:
            verified.append(rec)
            elapsed = time.perf_counter() - t0
            if tel.time_to_first_exact_demo_solution is None:
                tel.time_to_first_exact_demo_solution = elapsed
            if tel.time_to_first_verified_test_candidate is None:
                tel.time_to_first_verified_test_candidate = elapsed
        elif node.residual.n_exact > 0:
            partials.append(rec)
            tel.consistency_prunes += 1
        tel.best_n_exact = max(tel.best_n_exact, node.residual.n_exact)
        if tel.best_pixel_residual is None or node.residual.pixel_total < tel.best_pixel_residual:
            tel.best_pixel_residual = node.residual.pixel_total

    if root.residual.all_exact:
        maybe_record(root, "identity")

    while heap:
        now = time.perf_counter()
        if now - t0 >= budget.max_seconds:
            tel.timed_out = True
            break
        if tel.nodes_expanded >= budget.max_nodes:
            tel.hit_node_limit = True
            break
        node = heapq.heappop(heap)
        tel.nodes_expanded += 1
        tel.nesting_depth = max(tel.nesting_depth, node.program.depth())

        if node.residual.all_exact:
            maybe_record(node, "")
            if node.cost < best_feasible_cost:
                best_feasible_cost = node.cost
            # Anytime: keep searching for a structurally distinct alternative.
            if len({v.program.serialize() for v in verified}) >= 2:
                # still allow a bit more search but bound-prune aggressively
                pass

        if node.program.depth() >= budget.max_depth:
            continue

        ops = generate_ops(task, node.preds, train_out, cfg, node.residual.dominant_domain)
        if family is None:
            family = legal_op_family(op.name for op in ops)
        expanded_with_ops += 1
        # Reserve exact generators and the finite geometry family before the
        # per-node cap. Residual-guided ops may follow; they must not starve D8.
        reserved_names = ("apply_colormap", "abs") + GEOM_UNARY
        reserved = [op for op in ops if op.name in reserved_names]
        rest = [op for op in ops if op.name not in reserved_names]
        cap = max(budget.max_ops_per_node, len(reserved))
        ops = (reserved + rest)[:cap]

        for op in ops:
            if time.perf_counter() - t0 >= budget.max_seconds:
                tel.timed_out = True
                break
            if op.name == "abs":
                from src.hrps.abstractions import active_library

                extra = active_library().cost(op.args[0])
            else:
                extra = OP_DEFS[op.name].cost
                if op.name == "apply_colormap" and op.args:
                    extra += 2 * len(op.args[0])
            child_cost = node.cost + extra
            child_prog = node.program.extend(op)
            # Admissible BnB on description length: h = 0 if this child were
            # already solved, else MIN_OP_COST. Never overestimates remaining cost.
            h_bound = MIN_OP_COST
            if best_feasible_cost < 10**9 and child_cost + h_bound >= best_feasible_cost:
                tel.bound_prunes += 1
                continue

            child_preds_l: list[Grid] = []
            failed = False
            changed = False
            for g in node.preds:
                key = (op.serialize(), g)
                if key in cache:
                    nxt = cache[key]
                else:
                    nxt = execute_op(op, g)
                    cache[key] = nxt
                if nxt is None:
                    failed = True
                    tel.executor_rejects += 1
                    break
                if nxt != g:
                    changed = True
                child_preds_l.append(nxt)
            if failed:
                continue
            if not changed:
                tel.noop_rejects += 1
                continue
            child_preds = tuple(child_preds_l)
            tel.nodes_generated += 1
            tel.raw_states += 1

            if cfg.continuation_dedup:
                sig = continuation_signature(child_preds, cfg.stage, family)
                if sig in seen:
                    tel.duplicate_states += 1
                    tel.dominance_prunes += 1
                    continue
                seen.add(sig)
                tel.unique_signatures = len(seen)

            spec = node.spec if node.spec is not None else (specs[0] if specs else None)
            # Inner-loop residual is pixel/shape only (exact). Object/relation
            # residuals are diagnoses, not required for expansion legality.
            child_res = joint_residual(child_preds, train_out, spec=None)
            child = _Node(
                priority=_score(child_res, child_cost, cfg),
                node_id=next_id,
                program=child_prog,
                preds=child_preds,
                residual=child_res,
                cost=child_cost,
                spec=spec,
            )
            next_id += 1
            if child_res.all_exact:
                maybe_record(child, "")
                if child_cost < best_feasible_cost:
                    best_feasible_cost = child_cost
            elif cfg.require_joint_solution and child_res.n_exact > 0 and child_res.n_exact < child_res.n_demos:
                tel.consistency_prunes += 1
            if len(heap) >= budget.max_frontier:
                tel.hit_frontier_limit = True
                break
            heapq.heappush(heap, child)
            if len(heap) > tel.frontier_peak:
                tel.frontier_peak = len(heap)
        if tel.timed_out or tel.hit_frontier_limit:
            break

    if not heap and not tel.timed_out and not tel.hit_node_limit and not tel.hit_frontier_limit:
        tel.enumerated_exhausted = True
    if cfg.continuation_dedup and tel.raw_states:
        tel.quotient_ratio = tel.unique_signatures / tel.raw_states
    tel.rss_bytes = _rss_bytes()

    verified.sort(key=lambda c: (c.cost, c.program.serialize()))
    # Unique programs
    uniq_v: list[CandidateRecord] = []
    seen_p = set()
    for c in verified:
        s = c.program.serialize()
        if s in seen_p:
            continue
        seen_p.add(s)
        uniq_v.append(c)

    attempt_progs: list[Program] = []
    incomplete_pool = sorted(
        partials
        + [
            CandidateRecord(
                root.program,
                root.preds,
                root.residual,
                0,
                root.residual.n_exact,
                "incomplete",
                0,
            )
        ],
        key=lambda c: (-c.n_exact, c.residual.pixel_total, c.cost),
    )
    if uniq_v:
        attempt_progs.append(uniq_v[0].program)
        tel.description_length = uniq_v[0].cost
        alt = next((c for c in uniq_v[1:] if _structurally_distinct(c.program, uniq_v[0].program)), None)
        if alt is not None:
            attempt_progs.append(alt.program)
    else:
        if incomplete_pool:
            attempt_progs.append(incomplete_pool[0].program)
            alt = next(
                (
                    c
                    for c in incomplete_pool[1:]
                    if _structurally_distinct(c.program, incomplete_pool[0].program)
                ),
                None,
            )
            if alt is not None:
                attempt_progs.append(alt.program)

    # Kaggle requires two slots. Duplicate the best eligible program rather
    # than injecting an unverified identity filler.
    if not attempt_progs:
        attempt_progs.append(Program(()))
    if len(attempt_progs) < 2:
        attempt_progs.append(attempt_progs[0])

    attempts: list[list[list[list[int]]]] = []
    programs_s: list[str] = []
    test_exact: list[bool] = []
    test_gts = task.test_outputs()
    all_test_ok = True
    for prog in attempt_progs[:2]:
        programs_s.append(prog.serialize())
        per_test = []
        for i, inp in enumerate(task.test_inputs()):
            pred = replay(prog, inp)
            per_test.append(to_lists(pred) if pred is not None else [[0]])
            gt = test_gts[i]
            ok = pred is not None and gt is not None and pred == gt
            if not ok:
                all_test_ok = False
        attempts.append(per_test)
    while len(attempts) < 2:
        attempts.append(attempts[0] if attempts else [[[0]] for _ in task.test])
        programs_s.append(programs_s[0] if programs_s else "identity")

    # Official pass@2: each test input is correct if either attempt matches.
    # Microscope `solved` additionally requires a jointly verified program
    # (eligible under the HRPS spec) rather than an unverified lucky grid.
    solved_tests_pass2 = []
    verified_tests = []
    for i in range(task.n_test):
        gt = test_gts[i]
        ok2 = False
        if gt is not None:
            for att in attempts:
                pred_l = att[i]
                if pred_l is not None and tuple(tuple(r) for r in pred_l) == gt:
                    ok2 = True
                    break
        solved_tests_pass2.append(ok2)
        if gt is None:
            verified_tests.append(False)
        else:
            pred0 = attempts[0][i] if attempts else None
            verified_tests.append(
                bool(uniq_v)
                and pred0 is not None
                and tuple(tuple(r) for r in pred0) == gt
            )
    hidden = any(g is None for g in test_gts)
    joint_ok = bool(uniq_v)
    if hidden:
        solved = joint_ok
    else:
        solved = joint_ok and all(verified_tests)
    pass2_solved = (True if hidden else all(solved_tests_pass2))
    test_exact = verified_tests
    tel_extra_pass2 = pass2_solved

    if uniq_v and tel.bound_value:
        # tightness: h / true remaining; at a solution remaining is 0 so report
        # min_op_cost / (solution_cost or 1) as a crude post-hoc ratio.
        tel.bound_tightness = round(MIN_OP_COST / max(uniq_v[0].cost, 1), 6)

    runtime = time.perf_counter() - t0
    if joint_ok and not hidden and not solved:
        failure = "consistency"
    else:
        failure = classify_failure(
            solved=solved,
            timed_out=tel.timed_out,
            hit_node_limit=tel.hit_node_limit,
            hit_frontier_limit=tel.hit_frontier_limit,
            enumerated_exhausted=tel.enumerated_exhausted,
            n_ops_generated=tel.nodes_generated,
            n_executor_rejects=tel.executor_rejects,
            n_partial_consistency=len(partials),
            n_exact_demos_best=tel.best_n_exact,
            n_demos=task.n_train,
            representation_instability=float(stats["representation_instability"]),
            used_object_ops=bool(cfg.object_specs),
        )
    cand_sum = [c.diagnostics() for c in (uniq_v[:4] or partials[:4])]
    tel_d = tel.as_dict()
    tel_d["runtime"] = runtime
    tel_d["joint_verified"] = joint_ok
    tel_d["pass2_solved"] = tel_extra_pass2
    tel_d["n_verified_programs"] = len(uniq_v)
    return TaskResult(
        task_id=task.task_id,
        split=task.split,
        stage=cfg.stage,
        solved=solved,
        runtime=runtime,
        attempts=attempts,
        programs=programs_s,
        failure_category=failure,
        telemetry=tel_d,
        task_stats=stats,
        candidate_summaries=cand_sum,
        test_exact=test_exact,
    )

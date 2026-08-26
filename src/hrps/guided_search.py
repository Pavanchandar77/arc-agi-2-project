"""Residual-guided Bond-L1 search with optional model proposals.

The verifier is exact. Frontier order is heuristic (pixel residual, then
description length). Extra test-time nodes are a compute budget, not an
admissible bound.

Kind: heuristic search + exact verification.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.hrps.bond_reward import rank_candidates, verifier_reward
from src.hrps.dsl import Program
from src.hrps.env import parse_program
from src.hrps.language import LANGUAGE_ID, bond_l1_budget, bond_l1_config
from src.hrps.model import FrozenOpenModel
from src.hrps.search import SearchBudget, search_task
from src.hrps.task import ArcTask


@dataclass
class GuidedSearchResult:
    task_id: str
    language: str
    programs: list[str]
    ranked: list[dict[str, Any]]
    joint_demo_exact: bool
    n_model_proposals: int = 0
    n_model_verified: int = 0
    search_solved: bool = False
    telemetry: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "language": self.language,
            "programs": self.programs,
            "ranked": self.ranked,
            "joint_demo_exact": self.joint_demo_exact,
            "n_model_proposals": self.n_model_proposals,
            "n_model_verified": self.n_model_verified,
            "search_solved": self.search_solved,
            "telemetry": self.telemetry,
        }


_PROGRAM_RE = re.compile(
    r"(?:rot90|rot180|rot270|flip_h|flip_v|transpose|anti_transpose|crop_fg|tile|upscale|downscale|"
    r"recolor|swap_colors|keep_color|apply_colormap|fill_holes|outline|gravity|isolate_largest|"
    r"isolate_smallest|recolor_smallest_to_largest_color|recolor_largest_to_smallest_color|"
    r"erase_smallest|erase_largest|recolor_all_fg_to_smallest_color|"
    r"recolor_nonsingleton_to_singleton_color|keep_least_frequent_color|keep_most_frequent_nonbg|"
    r"recolor_least_frequent_to_most_frequent|translate_fg|center_fg|left_half|right_half|"
    r"top_half|bottom_half)(?::[^\s|]+)?(?:\s*\|\s*[A-Za-z_]+(?::[^\s|]+)?)*"
)


def extract_program_texts(text: str) -> list[str]:
    found: list[str] = []
    try:
        blob = json.loads(text)
        if isinstance(blob, dict):
            prog = blob.get("program") or (blob.get("arguments") or {}).get("program")
            if isinstance(prog, str):
                found.append(prog)
    except Exception:
        pass
    for m in _PROGRAM_RE.finditer(text or ""):
        found.append(m.group(0).strip())
    out: list[str] = []
    seen: set[str] = set()
    for p in found:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out[:8]


def _verify_text(task: ArcTask, text: str, max_depth: int) -> Optional[Program]:
    prog, err = parse_program(text, max_depth=max_depth)
    if prog is None or err or not prog.ops:
        return None
    rec = verifier_reward(task, prog)
    return prog if rec["joint_demo_exact"] else None


def guided_search(
    task: ArcTask,
    *,
    model: Optional[FrozenOpenModel] = None,
    budget: Optional[SearchBudget] = None,
    max_proposals: int = 4,
) -> GuidedSearchResult:
    budget = budget or bond_l1_budget()
    cfg = bond_l1_config()
    verified: list[Program] = []
    n_prop = 0
    n_ok = 0
    if model is not None:
        prompt = (
            "Propose an HRPS DSL program for these demonstrations. "
            "Emit one JSON object {\"action\":\"propose_program\",\"arguments\":{\"program\":\"...\"}}.\n"
        )
        observe = []
        for i, p in enumerate(task.train):
            observe.append(f"DEMO {i} IN {p.input} OUT {p.output}")
        for step in range(max_proposals):
            turn = model.generate(
                prompt + "\n".join(observe),
                max_tokens=128,
                temperature=0.0,
                seed=step,
            )
            n_prop += 1
            for text in extract_program_texts(turn.text):
                prog = _verify_text(task, text, budget.max_depth)
                if prog is not None:
                    n_ok += 1
                    verified.append(prog)
    res = search_task(task, stage="L", budget=budget, cfg=cfg)
    for ser in res.programs:
        prog, err = parse_program(ser, max_depth=budget.max_depth)
        if prog is not None and not err:
            verified.append(prog)
    # Dedup
    uniq: dict[str, Program] = {}
    for p in verified:
        uniq[p.serialize()] = p
    ranked = rank_candidates(task, list(uniq.values()))
    joint = any(r["joint_demo_exact"] and not r["underconstraint_flags"] for r in ranked) or any(
        r["joint_demo_exact"] for r in ranked
    )
    return GuidedSearchResult(
        task_id=task.task_id,
        language=LANGUAGE_ID,
        programs=[r["program"] for r in ranked if r["joint_demo_exact"]],
        ranked=ranked,
        joint_demo_exact=bool(ranked and ranked[0]["joint_demo_exact"]),
        n_model_proposals=n_prop,
        n_model_verified=n_ok,
        search_solved=bool(res.solved),
        telemetry={
            "nodes_expanded": res.telemetry.get("nodes_expanded"),
            "failure": res.failure_category,
            "runtime": res.runtime,
            "n_verified_search": len(res.programs),
        },
    )

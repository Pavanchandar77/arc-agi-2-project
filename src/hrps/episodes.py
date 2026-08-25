"""HRPS reasoning episodes for Bond training.

Episodes are generated only from the official training split, excluding the
held-out diagnostic slice. Public evaluation is never read. Test gold never
appears in the observation stream; it is used only as a training-split
outcome tag (commit vs reject) and for the optional direct-answer target.

Kind: exact env replay; learned inventory of which traces become episodes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from src.hrps.abstractions import AbstractionLibrary, parse_program_text
from src.hrps.agent import SYSTEM_M0, SYSTEM_M2
from src.hrps.dsl import Program
from src.hrps.env import Action, HrpsEnv, gold_free_constraint_feedback, grid_to_compact, parse_program
from src.hrps.separability import DEFAULT_N, DEFAULT_OFFSET, held_out_training_ids
from src.hrps.task import DEFAULT_DATA_ROOT, ArcTask, load_task_file

G_TRACE_PATH = Path(__file__).resolve().parent.parent.parent / "artifacts" / "hrps_phase1_G_training" / "tasks.jsonl"

ACTION_SCHEMA: dict[str, Any] = {
    "actions": [
        "observe",
        "inspect",
        "apply",
        "residual",
        "hypothesize",
        "commit",
        "catalog",
        "answer",
    ],
    "inspect_targets": ["colors", "shapes", "objects", "catalog", "underconstraint"],
    "program": "serialized DSL, ops joined by ' | ', depth <= 3",
    "verifier": "joint pixel residual + gold-free underconstraint flags",
    "hidden": ["test_output", "public_evaluation"],
}

COMPETING_GEOM = {
    "rot180": "rot90",
    "rot90": "rot270",
    "rot270": "rot90",
    "flip_h": "flip_v",
    "flip_v": "flip_h",
    "transpose": "anti_transpose",
    "anti_transpose": "transpose",
}

FAMILY_HYPOTHESIS = {
    "rot90": "the grid is rotated 90 degrees",
    "rot180": "the grid is rotated 180 degrees",
    "rot270": "the grid is rotated 270 degrees",
    "flip_h": "the grid is reflected left-right",
    "flip_v": "the grid is reflected top-bottom",
    "transpose": "the grid is transposed",
    "anti_transpose": "the grid is anti-transposed",
    "crop_fg": "crop to the foreground bounding box",
    "left_half": "keep the left half",
    "right_half": "keep the right half",
    "top_half": "keep the top half",
    "bottom_half": "keep the bottom half",
    "tile": "tile the grid",
    "upscale": "upscale by integer blocks",
    "downscale": "downscale uniform blocks",
    "recolor": "recolor one source color",
    "swap_colors": "swap two colors",
    "keep_color": "keep one color and erase the rest",
    "apply_colormap": "apply a global colormap",
    "fill_holes": "fill enclosed background holes",
    "outline": "keep object outlines",
    "gravity": "pack non-background cells toward a side",
    "isolate_largest": "isolate the largest object",
    "isolate_smallest": "isolate the smallest object",
    "abs": "apply a named exact abstraction",
}


def _family(program_text: str) -> str:
    token = program_text.split("|")[0].strip().split(":")[0].strip()
    return token or "identity"


def _hypothesis(program_text: str) -> str:
    fam = _family(program_text)
    return FAMILY_HYPOTHESIS.get(fam, f"the transformation is a {fam} program")


@dataclass
class BondTurn:
    assistant: str
    feedback: str
    accepted: bool
    residual_pixel: Optional[int] = None
    joint_exact: Optional[bool] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "assistant": self.assistant,
            "feedback": self.feedback,
            "accepted": self.accepted,
            "residual_pixel": self.residual_pixel,
            "joint_exact": self.joint_exact,
        }


@dataclass
class BondEpisode:
    episode_id: str
    task_id: str
    kind: str
    split: str
    held_out: bool
    program: str
    joint_demo_exact: bool
    test_transfer: Optional[bool]
    turns: list[BondTurn] = field(default_factory=list)
    observe: str = ""
    catalog: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "split": self.split,
            "held_out": self.held_out,
            "program": self.program,
            "joint_demo_exact": self.joint_demo_exact,
            "test_transfer": self.test_transfer,
            "observe": self.observe,
            "catalog": self.catalog,
            "turns": [t.as_dict() for t in self.turns],
            "provenance": self.provenance,
        }

    def to_sft_messages(self, *, system: Optional[str] = None) -> list[dict[str, str]]:
        sys = system or (SYSTEM_M0 if self.kind == "direct_answer" else SYSTEM_M2)
        msgs = [{"role": "system", "content": sys}]
        if self.kind == "direct_answer":
            msgs.append({"role": "user", "content": self.observe})
            if self.turns:
                msgs.append({"role": "assistant", "content": self.turns[0].assistant})
            return msgs
        user = self.observe
        if self.catalog:
            user = user + "\n\nCATALOG:\n" + self.catalog
        msgs.append({"role": "user", "content": user})
        for turn in self.turns:
            msgs.append({"role": "assistant", "content": turn.assistant})
            msgs.append({"role": "user", "content": "HRPS:\n" + turn.feedback})
        # Drop the trailing environment turn so SFT ends on an assistant action.
        if msgs and msgs[-1]["role"] == "user" and len(msgs) > 2:
            msgs.pop()
        return msgs


def _episode_id(task_id: str, kind: str, program: str) -> str:
    raw = f"{task_id}|{kind}|{program}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _act(env: HrpsEnv, kind: str, payload: str = "") -> BondTurn:
    fb = env.step(Action(kind, payload, payload))
    residual = None
    joint = None
    if isinstance(fb.data.get("residual"), dict):
        residual = fb.data["residual"].get("pixel_total")
        joint = fb.data["residual"].get("all_exact")
    text = f"{kind.upper()} {payload}".strip()
    return BondTurn(
        assistant=text,
        feedback=fb.text,
        accepted=fb.accepted,
        residual_pixel=residual,
        joint_exact=joint,
    )


def _flags_after_apply(env: HrpsEnv, program: Program) -> list[str]:
    data = gold_free_constraint_feedback(env.task, program)
    return list(data.get("underconstraint_flags") or [])


def teacher_hrps_episode(
    task: ArcTask,
    program_text: str,
    *,
    test_transfer: Optional[bool],
    kind_hint: str,
    library: Optional[AbstractionLibrary] = None,
    enable_h: bool = False,
    include_competing: bool = True,
) -> Optional[BondEpisode]:
    """Replay a known program through the env as a teacher trajectory."""
    program, err = parse_program(program_text)
    if program is None or err or not program.ops:
        return None
    env = HrpsEnv(task, library=library, enable_h=enable_h)
    observe = env.observe()
    catalog = env.catalog_text()
    turns: list[BondTurn] = []
    turns.append(_act(env, "hypothesize", _hypothesis(program_text)))
    turns.append(_act(env, "inspect", "shapes"))
    turns.append(_act(env, "inspect", "colors"))

    first = program.ops[0].serialize()
    alt = COMPETING_GEOM.get(_family(program_text))
    if include_competing and alt and alt != first:
        t_wrong = _act(env, "apply", alt)
        turns.append(t_wrong)
        if not t_wrong.joint_exact:
            turns.append(
                _act(
                    env,
                    "hypothesize",
                    f"{alt} does not fit the demonstrations; residual remains; try {_hypothesis(program_text)}",
                )
            )

    ops = program.ops
    if len(ops) > 1:
        prefix = Program(ops[:1]).serialize()
        t_pre = _act(env, "apply", prefix)
        turns.append(t_pre)
        if not t_pre.joint_exact:
            turns.append(
                _act(
                    env,
                    "hypothesize",
                    "prefix is incomplete; compose the remaining operators",
                )
            )
    t_full = _act(env, "apply", program.serialize())
    turns.append(t_full)
    turns.append(_act(env, "inspect", "underconstraint"))
    flags = _flags_after_apply(env, program)
    kind = kind_hint
    if t_full.joint_exact and flags and test_transfer is False:
        kind = "underconstraint"
        turns.append(
            _act(
                env,
                "hypothesize",
                "jointly exact on demonstrations is underconstrained: "
                + "; ".join(flags)
                + ". Do not commit this as transferred.",
            )
        )
    elif t_full.joint_exact:
        turns.append(_act(env, "commit", program.serialize()))
        if include_competing and alt:
            kind = "competing_hypotheses" if kind_hint == "success_trajectory" else kind
    else:
        kind = "failed_hypothesis"
        turns.append(
            _act(
                env,
                "hypothesize",
                "program is not jointly exact; do not commit",
            )
        )

    return BondEpisode(
        episode_id=_episode_id(task.task_id, kind, program.serialize()),
        task_id=task.task_id,
        kind=kind,
        split=task.split,
        held_out=False,
        program=program.serialize(),
        joint_demo_exact=bool(t_full.joint_exact),
        test_transfer=test_transfer,
        turns=turns,
        observe=observe.text,
        catalog=catalog,
        provenance={
            "source": "hrps_teacher_replay",
            "kind_hint": kind_hint,
            "n_turns": len(turns),
        },
    )


def teacher_direct_episode(task: ArcTask, program_text: str, test_transfer: Optional[bool]) -> Optional[BondEpisode]:
    """M0-style target: emit the program's test-input image. Uses training-split labels only."""
    program, err = parse_program(program_text)
    if program is None or err or not program.ops:
        return None
    if not task.test:
        return None
    pred = None
    from src.hrps.dsl import replay

    pred = replay(program, task.test[0].input)
    gt = task.test[0].output
    if pred is None or gt is None or pred != gt:
        return None
    env = HrpsEnv(task)
    observe = env.observe()
    grid_text = grid_to_compact(pred)
    return BondEpisode(
        episode_id=_episode_id(task.task_id, "direct_answer", program.serialize()),
        task_id=task.task_id,
        kind="direct_answer",
        split=task.split,
        held_out=False,
        program=program.serialize(),
        joint_demo_exact=True,
        test_transfer=True,
        turns=[BondTurn(assistant=grid_text, feedback="", accepted=True)],
        observe=observe.text,
        catalog="",
        provenance={"source": "direct_from_verified_program"},
    )


def teacher_abstraction_episode(
    task: ArcTask,
    abs_name: str,
    library: AbstractionLibrary,
    test_transfer: Optional[bool],
) -> Optional[BondEpisode]:
    return teacher_hrps_episode(
        task,
        f"abs:{abs_name}",
        test_transfer=test_transfer,
        kind_hint="abstraction",
        library=library,
        enable_h=True,
        include_competing=False,
    )


def teacher_search_strategy_episode(task: ArcTask, row: dict[str, Any]) -> BondEpisode:
    env = HrpsEnv(task)
    observe = env.observe()
    catalog = env.catalog_text()
    turns = [
        _act(env, "inspect", "shapes"),
        _act(env, "inspect", "objects"),
        _act(
            env,
            "hypothesize",
            "no jointly exact program yet; request objects rather than committing identity",
        ),
    ]
    return BondEpisode(
        episode_id=_episode_id(task.task_id, "search_strategy", "identity"),
        task_id=task.task_id,
        kind="search_strategy",
        split=task.split,
        held_out=False,
        program="identity",
        joint_demo_exact=False,
        test_transfer=False,
        turns=turns,
        observe=observe.text,
        catalog=catalog,
        provenance={
            "source": "search_trace",
            "failure_category": row.get("failure_category"),
            "nodes_expanded": (row.get("telemetry") or {}).get("nodes_expanded"),
            "best_pixel_residual": (row.get("telemetry") or {}).get("best_pixel_residual"),
        },
    )


def _best_program(row: dict[str, Any]) -> str:
    programs = [p for p in (row.get("programs") or []) if p and p != "identity"]
    return programs[0] if programs else ""


def generate_from_trace_row(
    row: dict[str, Any],
    task: ArcTask,
    *,
    held_out_ids: set[str],
    library: Optional[AbstractionLibrary] = None,
    max_strategy: bool = False,
) -> list[BondEpisode]:
    if task.split != "training":
        return []
    if task.task_id in held_out_ids:
        return []
    out: list[BondEpisode] = []
    prog = _best_program(row)
    solved = bool(row.get("solved"))
    test_exact = row.get("test_exact") or []
    transferred = bool(test_exact) and all(test_exact)
    joint = bool((row.get("telemetry") or {}).get("joint_verified")) or solved

    if prog:
        hint = "success_trajectory" if solved else ("underconstraint" if joint and not transferred else "failed_hypothesis")
        ep = teacher_hrps_episode(
            task,
            prog,
            test_transfer=transferred if joint else False,
            kind_hint=hint,
            library=library,
            enable_h=False,
            include_competing=True,
        )
        if ep is not None:
            ep.provenance["g_row_solved"] = solved
            ep.provenance["g_failure"] = row.get("failure_category")
            out.append(ep)
        if solved and transferred:
            direct = teacher_direct_episode(task, prog, True)
            if direct is not None:
                out.append(direct)
        if library and solved:
            body_map = {a.body_serialize(): a.name for a in library.items}
            if prog in body_map:
                abs_ep = teacher_abstraction_episode(task, body_map[prog], library, transferred)
                if abs_ep is not None:
                    out.append(abs_ep)
    elif max_strategy:
        out.append(teacher_search_strategy_episode(task, row))
    return out


def load_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def generate_bond_episodes(
    *,
    trace_path: Optional[Path] = None,
    data_root: Optional[Path] = None,
    library: Optional[AbstractionLibrary] = None,
    offset: int = DEFAULT_OFFSET,
    n: int = DEFAULT_N,
    max_strategy_episodes: int = 15,
) -> list[BondEpisode]:
    held = set(held_out_training_ids(offset=offset, n=n, data_root=data_root))
    path = Path(trace_path) if trace_path is not None else G_TRACE_PATH
    if not path.is_file():
        return []
    rows = load_trace_rows(path)
    folder = (data_root or DEFAULT_DATA_ROOT) / "training"
    episodes: list[BondEpisode] = []
    n_strategy = 0
    for row in rows:
        if row.get("split") and row.get("split") != "training":
            continue
        tid = row.get("task_id")
        if not tid or tid in held:
            continue
        prog = _best_program(row)
        want_strategy = (
            n_strategy < max_strategy_episodes
            and not row.get("solved")
            and not prog
        )
        if not prog and not want_strategy:
            continue
        task_path = folder / f"{tid}.json"
        if not task_path.is_file():
            continue
        task = load_task_file(task_path, "training")
        produced = generate_from_trace_row(
            row,
            task,
            held_out_ids=held,
            library=library,
            max_strategy=want_strategy,
        )
        if want_strategy and any(e.kind == "search_strategy" for e in produced):
            n_strategy += 1
        episodes.extend(produced)
    return episodes


def assert_training_safe(episodes: Iterable[BondEpisode], held_out_ids: Iterable[str]) -> None:
    held = set(held_out_ids)
    for ep in episodes:
        if ep.held_out:
            raise ValueError(f"held-out episode leaked into training: {ep.task_id}")
        if ep.task_id in held:
            raise ValueError(f"held-out task id in Bond data: {ep.task_id}")
        if ep.split != "training":
            raise ValueError(f"non-training split in Bond data: {ep.split} {ep.task_id}")
        blob = ep.observe + "".join(t.feedback for t in ep.turns)
        if "TEST 0 OUTPUT" in blob or "TEST 1 OUTPUT" in blob:
            raise ValueError(f"test gold leaked into observations: {ep.task_id}")


def write_episodes(episodes: list[BondEpisode], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "episodes.jsonl"
    sft = out_dir / "sft.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh, sft.open("w", encoding="utf-8") as sh:
        for ep in episodes:
            fh.write(json.dumps(ep.as_dict()) + "\n")
            sh.write(json.dumps({"messages": ep.to_sft_messages(), "task_id": ep.task_id, "kind": ep.kind}) + "\n")
    kinds: dict[str, int] = {}
    for ep in episodes:
        kinds[ep.kind] = kinds.get(ep.kind, 0) + 1
    digest = hashlib.sha256(sft.read_bytes()).hexdigest() if sft.is_file() else None
    summary = {
        "n_episodes": len(episodes),
        "n_sft": len(episodes),
        "kinds": kinds,
        "task_ids": sorted({e.task_id for e in episodes}),
        "sft_sha256": digest,
        "action_schema": ACTION_SCHEMA,
    }
    (out_dir / "episodes_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

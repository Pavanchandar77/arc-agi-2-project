"""Strict Bond JSON action and observation schema.

The model may only emit one of the permitted HRPS actions. Unknown names,
malformed arguments, arbitrary Python, shell, filesystem, network, and
hidden-label access are rejected before anything executes.

Kind: exact validation. Mapping onto src.hrps.env is exact for permitted
actions; rejected actions never reach the executor.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.hrps.env import Action, EnvFeedback, LEGAL_OP_NAMES, parse_program

BOND_ACTIONS = (
    "inspect_objects",
    "inspect_relations",
    "propose_program",
    "execute_program",
    "compare_residuals",
    "revise_hypothesis",
    "reject_hypothesis",
    "commit_candidates",
)

ALLOWED_ACTIONS = frozenset(BOND_ACTIONS)

ARG_KEYS: dict[str, frozenset[str]] = {
    "inspect_objects": frozenset({"spec"}),
    "inspect_relations": frozenset({"spec"}),
    "propose_program": frozenset({"program", "note"}),
    "execute_program": frozenset({"program"}),
    "compare_residuals": frozenset({"candidate_id"}),
    "revise_hypothesis": frozenset({"text", "program"}),
    "reject_hypothesis": frozenset({"reason", "program"}),
    "commit_candidates": frozenset({"program"}),
}

REQUIRED_ARGS: dict[str, frozenset[str]] = {
    "propose_program": frozenset({"program"}),
    "execute_program": frozenset({"program"}),
    "revise_hypothesis": frozenset({"text"}),
    "reject_hypothesis": frozenset({"reason"}),
}

FORBIDDEN_PATTERNS = (
    r"\bimport\s+os\b",
    r"\bsubprocess\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"__import__",
    r"\bos\.system\b",
    r"\bsocket\b",
    r"\burllib\b",
    r"\brequests\b",
    r"\bopen\s*\(",
    r"\bpathlib\b",
    r"\bhttp[s]?://",
    r"TEST\s+\d+\s+OUTPUT",
    r"hidden_label",
    r"public_evaluation",
)

FORBIDDEN_ACTION_NAMES = frozenset(
    {
        "shell",
        "python",
        "exec",
        "eval",
        "write_file",
        "read_file",
        "http",
        "download",
        "install",
        "run",
        "code",
        "system",
    }
)

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["action"],
    "properties": {
        "action": {"enum": list(BOND_ACTIONS)},
        "arguments": {"type": "object"},
    },
    "additionalProperties": False,
    "forbidden": [
        "python",
        "shell",
        "filesystem",
        "network",
        "hidden_labels",
        "unrestricted_code",
    ],
}


@dataclass(frozen=True)
class ValidatedAction:
    action: str
    arguments: dict[str, Any]
    raw: str


@dataclass
class ParseResult:
    ok: bool
    action: Optional[ValidatedAction] = None
    error: str = ""
    env_action: Optional[Action] = None


def _contains_forbidden(text: str) -> Optional[str]:
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, text, flags=re.I):
            return f"forbidden_pattern:{pat}"
    return None


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    if not text or not isinstance(text, str):
        return []
    decoder = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(text):
        start = text.find("{", i)
        if start < 0:
            break
        try:
            blob, end = decoder.raw_decode(text[start:])
        except Exception:
            i = start + 1
            continue
        if isinstance(blob, dict):
            out.append(blob)
        i = start + end
    return out


def validate_action_dict(blob: dict[str, Any], *, raw: str = "") -> ParseResult:
    if not isinstance(blob, dict):
        return ParseResult(False, error="not_an_object")
    extra_top = set(blob) - {"action", "arguments"}
    if extra_top:
        return ParseResult(False, error=f"unknown_fields:{sorted(extra_top)}")
    name = blob.get("action")
    if not isinstance(name, str):
        return ParseResult(False, error="missing_action")
    name = name.strip()
    if name in FORBIDDEN_ACTION_NAMES:
        return ParseResult(False, error=f"forbidden_action:{name}")
    if name not in ALLOWED_ACTIONS:
        return ParseResult(False, error=f"unknown_action:{name}")
    args = blob.get("arguments", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return ParseResult(False, error="arguments_not_object")
    unknown = set(args) - ARG_KEYS[name]
    if unknown:
        return ParseResult(False, error=f"unknown_arguments:{sorted(unknown)}")
    missing = REQUIRED_ARGS.get(name, frozenset()) - set(args)
    if missing:
        return ParseResult(False, error=f"missing_arguments:{sorted(missing)}")
    forb = _contains_forbidden(raw or json.dumps(blob))
    if forb:
        return ParseResult(False, error=forb)
    if "program" in args:
        prog, err = parse_program(str(args["program"]))
        if prog is None or err:
            return ParseResult(False, error=err or "invalid_program")
        for op in prog.ops:
            if op.name not in LEGAL_OP_NAMES:
                return ParseResult(False, error=f"unknown_op:{op.name}")
    va = ValidatedAction(action=name, arguments=dict(args), raw=raw or json.dumps(blob, sort_keys=True))
    return ParseResult(True, action=va, env_action=to_env_action(va))


def parse_strict_action(text: str) -> ParseResult:
    """Parse exactly one Bond JSON action. Kind: exact reject of anything else."""
    if not text or not isinstance(text, str):
        return ParseResult(False, error="empty")
    forb = _contains_forbidden(text)
    if forb:
        return ParseResult(False, error=forb)
    blobs = _extract_json_objects(text)
    if not blobs:
        return ParseResult(False, error="no_json_action")
    return validate_action_dict(blobs[0], raw=text)


def to_env_action(va: ValidatedAction) -> Action:
    args = va.arguments
    if va.action == "inspect_objects":
        spec = str(args.get("spec") or "4c_zero")
        return Action("inspect", f"objects {spec}", va.raw)
    if va.action == "inspect_relations":
        spec = str(args.get("spec") or "4c_zero")
        return Action("inspect", f"relations {spec}", va.raw)
    if va.action == "propose_program":
        note = str(args.get("note") or "").strip()
        payload = f"propose {args['program']}" + (f" :: {note}" if note else "")
        return Action("hypothesize", payload, va.raw)
    if va.action == "execute_program":
        return Action("apply", str(args["program"]), va.raw)
    if va.action == "compare_residuals":
        return Action("residual", "", va.raw)
    if va.action == "revise_hypothesis":
        return Action("hypothesize", str(args["text"]), va.raw)
    if va.action == "reject_hypothesis":
        return Action("hypothesize", f"reject: {args['reason']}", va.raw)
    if va.action == "commit_candidates":
        return Action("commit", str(args.get("program") or ""), va.raw)
    return Action("unknown", va.action, va.raw)


def compact_observation(fb: EnvFeedback) -> dict[str, Any]:
    """Structured observation. No test gold. Compact enough for prompts."""
    data = dict(fb.data or {})
    residual = data.get("residual") if isinstance(data.get("residual"), dict) else None
    constraint = data.get("constraint") if isinstance(data.get("constraint"), dict) else data
    flags = []
    if isinstance(constraint, dict):
        flags = list(constraint.get("underconstraint_flags") or [])
    obs: dict[str, Any] = {
        "status": "ok" if fb.accepted else "rejected",
        "action": fb.action,
        "error": None if fb.accepted else data.get("error") or "rejected",
        "uses_test_labels": False,
    }
    if residual:
        obs["residual"] = {
            "pixel_total": residual.get("pixel_total"),
            "n_exact": residual.get("n_exact"),
            "n_demos": residual.get("n_demos"),
            "all_exact": residual.get("all_exact"),
            "shape_mismatch": residual.get("any_shape_mismatch"),
            "dominant_domain": residual.get("dominant_domain"),
        }
    if flags:
        obs["underconstraint_flags"] = flags
    if isinstance(constraint, dict):
        if constraint.get("test_unseen_input_colors"):
            obs["test_unseen_input_colors"] = constraint["test_unseen_input_colors"]
        if constraint.get("role_collision_colors"):
            obs["role_collision_colors"] = constraint["role_collision_colors"]
        if constraint.get("disjoint_per_demo_color_support") is not None:
            obs["disjoint_per_demo_color_support"] = constraint["disjoint_per_demo_color_support"]
    if "program" in data:
        obs["program"] = data["program"]
    if data.get("what") == "catalog":
        obs["catalog"] = True
    if "demos" in data and isinstance(data["demos"], list):
        # object/shape/color inspect: keep counts only
        obs["n_demos"] = len(data["demos"])
        if data["demos"] and "n_in" in data["demos"][0]:
            obs["object_counts"] = [
                {"i": d.get("i"), "n_in": d.get("n_in"), "n_out": d.get("n_out")} for d in data["demos"][:6]
            ]
        if data["demos"] and "in_hw" in data["demos"][0]:
            obs["shapes"] = [
                {"i": d.get("i"), "in_hw": d.get("in_hw"), "out_hw": d.get("out_hw")} for d in data["demos"][:6]
            ]
    if "relations" in data:
        obs["relations"] = data["relations"]
    obs["text"] = fb.text
    return obs


def teacher_line_to_json(assistant_line: str) -> Optional[dict[str, Any]]:
    """Project a legacy teacher line onto the Bond JSON schema."""
    line = (assistant_line or "").strip()
    if not line:
        return None
    upper = line.upper()
    if upper.startswith("INSPECT"):
        rest = line.split(None, 1)[1].strip().lower() if " " in line else "objects"
        token = rest.split()[0]
        if token in {"relations", "relation"}:
            return {"action": "inspect_relations", "arguments": {}}
        spec = "4c_zero"
        parts = rest.split()
        if len(parts) >= 2:
            spec = parts[1]
        return {"action": "inspect_objects", "arguments": {"spec": spec} if token == "objects" else {}}
    if upper.startswith("APPLY"):
        prog = line.split(None, 1)[1].strip() if " " in line else ""
        return {"action": "execute_program", "arguments": {"program": prog}}
    if upper.startswith("COMMIT"):
        prog = line.split(None, 1)[1].strip() if " " in line else ""
        return {"action": "commit_candidates", "arguments": {"program": prog} if prog else {}}
    if upper.startswith("HYPOTHESIZE"):
        text = line.split(None, 1)[1].strip() if " " in line else ""
        low = text.lower()
        if low.startswith("reject") or "underconstrained" in low or "do not commit" in low:
            return {"action": "reject_hypothesis", "arguments": {"reason": text}}
        if low.startswith("propose "):
            return {"action": "propose_program", "arguments": {"program": text.split(None, 1)[-1]}}
        return {"action": "revise_hypothesis", "arguments": {"text": text}}
    if upper.startswith("RESIDUAL"):
        return {"action": "compare_residuals", "arguments": {}}
    return None


SYSTEM_BOND_JSON = (
    "You are Bond, an active hypothesis-testing agent inside HRPS. "
    "Emit exactly one JSON object per turn with keys action and arguments. "
    "Permitted actions: inspect_objects, inspect_relations, propose_program, "
    "execute_program, compare_residuals, revise_hypothesis, reject_hypothesis, "
    "commit_candidates. "
    "HRPS executes exactly and returns residuals. Joint demonstration exactness "
    "does not imply test transfer. If underconstraint flags mention disjoint palettes, "
    "unseen test-input colors, or marker/blob role collision, reject the hypothesis "
    "instead of committing. No Python, no shell, no files, no network."
)

"""Neural proposal, symbolic verification.

A language model is good at guessing which composition of operations explains a
task, and bad at being right. The DSL executor is the opposite: it cannot guess
at all, but it can settle the question exactly. So the model proposes programs
and never answers directly, and the demonstrations decide which proposals live.

    model  ->  candidate programs
                    |
            parse and typecheck        unknown names never execute
                    |
            replay on every demo       exact equality, all pairs
                    |
            survivors only             a proposal that misses one demo is gone
                    |
            consensus on the test input

Three properties hold by construction, and are tested:

* **Test outputs are never consulted.** Verification reads ``task.train`` only.
  Nothing here can see the answer it is being scored against.
* **Only catalogued operators run.** Proposals are parsed into ``Op`` values
  whose names must exist in the DSL catalog; an unrecognised name is rejected
  rather than evaluated. No model output is ever passed to ``eval``.
* **Exactness is not relaxed.** A program survives only by reproducing every
  demonstration cell for cell.

The last stage is where this differs from sampling and voting. Candidates that
survive agree on all demonstrations yet can still diverge on the test input,
and that divergence is the only honest measure of uncertainty available. Votes
are counted among survivors, so every vote has already been paid for with a
proof on the demonstrations - unlike a vote among raw samples, which is only
evidence that a model is confident.

Kind: exact verification, heuristic ranking among survivors.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from src.hrps.dsl import OP_DEFS, DslType, Op, Program, replay
from src.hrps.grid import Grid, is_valid_grid
from src.hrps.task import ArcTask

# Abstraction ops index into a mutable runtime library, so a proposal naming one
# is not self-contained and its meaning depends on state the proposer cannot
# see. They are searchable but never proposable.
UNPROPOSABLE = frozenset({"abs"})

# ";" is deliberately NOT a pipeline separator: apply_colormap uses it to
# separate its own colour pairs ("1-2;3-4"), so treating it as a separator
# would split a colormap in half. The canonical serialization uses "|".
SEPARATORS = re.compile(r"\s*(?:\||->|=>|\bthen\b)\s*", flags=re.IGNORECASE)
_FENCE = re.compile(r"```(?:\w+)?\s*(.*?)\s*```", flags=re.DOTALL)
# A program line is made of operator tokens; anything with prose in it is not.
_PROGRAM_LINE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?::[0-9A-Za-z,;\-x]+)?"
                           r"(?:\s*(?:\||->|=>)\s*[A-Za-z_][A-Za-z0-9_]*(?::[0-9A-Za-z,;\-x]+)?)*$")

MAX_OPS = 8  # A proposal longer than this is not a hypothesis, it is a guess.


def _arity(name: str) -> int:
    """Number of arguments after the leading grid."""
    return max(0, len(OP_DEFS[name].in_types) - 1)


def _args_well_typed(name: str, args: tuple) -> bool:
    """Cheap range checks on the argument types the catalog declares."""
    types = OP_DEFS[name].in_types[1:]
    if len(args) != len(types):
        return False
    for value, dtype in zip(args, types):
        if dtype is DslType.COLORMAP:
            if not isinstance(value, tuple) or not value:
                return False
            for pair in value:
                if not (isinstance(pair, tuple) and len(pair) == 2):
                    return False
                if not all(isinstance(c, int) and 0 <= c <= 9 for c in pair):
                    return False
        elif dtype in (DslType.COLOR, DslType.BG):
            if not isinstance(value, int) or not 0 <= value <= 9:
                return False
        elif dtype is DslType.CONNECTIVITY:
            if value not in (4, 8):
                return False
        elif dtype is DslType.BOOL:
            if not isinstance(value, bool):
                return False
        elif dtype is DslType.INT:
            if isinstance(value, tuple):
                if not all(isinstance(x, int) for x in value):
                    return False
            elif not isinstance(value, int) or isinstance(value, bool):
                return False
    return True


def parse_program(text: str) -> Optional[Program]:
    """Parse one program, or None if it is not a valid, catalogued pipeline.

    Rejection is the common case and is never an error: a model proposing a
    misspelled or invented operator must fail closed, not approximately.
    """
    if not text or not isinstance(text, str):
        return None
    body = text.strip()
    if not body:
        return None
    if body.lower() in {"identity", "noop", "none"}:
        return Program(())
    tokens = [t for t in SEPARATORS.split(body) if t.strip()]
    if not tokens or len(tokens) > MAX_OPS:
        return None
    ops: list[Op] = []
    for token in tokens:
        token = token.strip().strip(",")
        if not token:
            return None
        name = token.split(":", 1)[0].strip()
        if name in UNPROPOSABLE or name not in OP_DEFS:
            return None
        try:
            op = Op.deserialize(token)
        except Exception:
            return None
        if op.name != name or len(op.args) != _arity(name):
            return None
        if not _args_well_typed(name, op.args):
            return None
        ops.append(op)
    return Program(tuple(ops))


def extract_candidates(text: str, *, limit: int = 64) -> list[str]:
    """Pull candidate program strings out of free-form model output.

    One per line, fenced blocks preferred. Numbering and bullets are stripped;
    prose lines are dropped rather than half-parsed.
    """
    if not text:
        return []
    blocks = _FENCE.findall(text)
    body = "\n".join(blocks) if blocks else text
    out: list[str] = []
    seen: set[str] = set()
    for raw in body.splitlines():
        line = raw.strip()
        line = re.sub(r"^(?:[-*+]|\d+[.)])\s*", "", line)  # bullets and numbering
        line = line.strip().strip("`").strip()
        if not line or not _PROGRAM_LINE.match(line):
            continue
        if line not in seen:
            seen.add(line)
            out.append(line)
        if len(out) >= limit:
            break
    return out


@dataclass(frozen=True)
class Verdict:
    """Why a program lived or died. Demonstrations only."""

    verified: bool
    n_demos: int
    n_matched: int
    failed_on: Optional[int] = None

    @property
    def is_total(self) -> bool:
        return self.verified and self.n_demos > 0 and self.n_matched == self.n_demos


def verify_program(program: Program, task: ArcTask) -> Verdict:
    """Replay on every demonstration and require exact equality on all of them.

    Reads ``task.train`` and nothing else. A task with no demonstrations cannot
    verify anything, so it returns unverified rather than vacuously true.
    """
    demos = [p for p in task.train if p.output is not None]
    if not demos:
        return Verdict(False, 0, 0)
    matched = 0
    for i, pair in enumerate(demos):
        try:
            got = replay(program, pair.input)
        except Exception:
            return Verdict(False, len(demos), matched, failed_on=i)
        if got is None or got != pair.output:
            return Verdict(False, len(demos), matched, failed_on=i)
        matched += 1
    return Verdict(True, len(demos), matched)


@dataclass
class ProposalReport:
    n_candidates: int = 0
    n_parsed: int = 0
    n_rejected: int = 0
    n_verified: int = 0
    n_distinct_outputs: int = 0
    consensus: float = 0.0  # share of survivors backing the chosen output
    winning_program: str = ""
    survivors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_candidates": self.n_candidates,
            "n_parsed": self.n_parsed,
            "n_rejected": self.n_rejected,
            "n_verified": self.n_verified,
            "n_distinct_outputs": self.n_distinct_outputs,
            "consensus": round(self.consensus, 3),
            "winning_program": self.winning_program,
            "survivors": self.survivors[:8],
        }


def propose_and_verify(
    candidates: Iterable[str],
    task: ArcTask,
    *,
    deadline: Optional[float] = None,
    max_survivors: int = 32,
) -> tuple[list[list[Optional[Grid]]], ProposalReport]:
    """Filter proposals through the demonstrations, then let survivors vote.

    Returns ``([attempt_1, attempt_2], report)``, each attempt holding one grid
    per test input, or None where no survivor produced a valid grid.
    """
    report = ProposalReport()
    test_inputs = task.test_inputs()
    survivors: list[Program] = []
    seen: set[str] = set()

    for text in candidates:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        report.n_candidates += 1
        program = parse_program(text)
        if program is None:
            report.n_rejected += 1
            continue
        key = program.serialize()
        if key in seen:
            continue
        seen.add(key)
        report.n_parsed += 1
        if verify_program(program, task).is_total:
            survivors.append(program)
            report.survivors.append(key)
            if len(survivors) >= max_survivors:
                break
    report.n_verified = len(survivors)
    if not survivors:
        return [[None] * len(test_inputs), [None] * len(test_inputs)], report

    # Every voter has already proved itself on the demonstrations. Weight by
    # simplicity so that, among equally proven explanations, the shorter one
    # carries more - a longer program agreeing by coincidence should not
    # outweigh a short one that generalises.
    per_test: list[list[Grid]] = []
    for i, grid_in in enumerate(test_inputs):
        tally: defaultdict[Grid, float] = defaultdict(float)
        for program in survivors:
            try:
                out = replay(program, grid_in)
            except Exception:
                continue
            if out is None or not is_valid_grid(out):
                continue
            tally[out] += 1.0 / (1.0 + program.cost())
        ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
        if i == 0:
            report.n_distinct_outputs = len(ranked)
            total = sum(v for _, v in ranked)
            if total > 0:
                report.consensus = ranked[0][1] / total
        per_test.append([g for g, _ in ranked[:2]])

    cheapest = min(survivors, key=lambda p: (p.cost(), p.depth(), p.serialize()))
    report.winning_program = cheapest.serialize()
    attempt_1 = [p[0] if p else None for p in per_test]
    attempt_2 = [(p[1] if len(p) > 1 else (p[0] if p else None)) for p in per_test]
    return [attempt_1, attempt_2], report

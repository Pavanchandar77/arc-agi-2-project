"""Sample programs from the model, let the demonstrations decide.

The model is asked for programs rather than grids, sampled several times for
genuinely different hypotheses, and every proposal is then filtered through the
demonstrations. What comes back is either an answer backed by a program that
reproduces every demonstration exactly, or nothing at all.

Sampling wide is cheap here in a way it is not for grid answers. A wrong grid
is indistinguishable from a right one without the answer key, so sampling more
grids buys only a popularity contest. A wrong program is detectable in
microseconds, so sampling more programs buys real coverage: the cost of a bad
proposal is one failed replay, and the benefit of a good one is a solved task.

Expert iteration falls out of this for free. A program the model proposes that
survives verification is, by the same standard used to build the original
training corpus, a correct label for that task - so it can be appended and
trained on, without anyone checking it by hand.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.hrps.grid import Grid
from src.hrps.program_prompt import build_inference_messages, build_training_example
from src.hrps.proposal import (
    ProposalReport,
    extract_candidates,
    parse_program,
    propose_and_verify,
    verify_program,
)
from src.hrps.task import ArcTask

# If the greedy sample cannot produce even one syntactically valid program, the
# model is not a program-emitting model and drawing again will not change that.
# Abandoning here keeps the fallback path nearly free instead of spending the
# whole sampling budget to learn the same thing eight times.
ABANDON_AFTER = 1


@dataclass
class ProgramSolveResult:
    attempts: list[list[Optional[Grid]]]
    report: ProposalReport
    n_samples: int = 0
    n_raw_candidates: int = 0
    seconds: float = 0.0

    @property
    def solved(self) -> bool:
        return self.report.n_verified > 0

    def as_dict(self) -> dict:
        out = self.report.as_dict()
        out.update(
            n_samples=self.n_samples,
            n_raw_candidates=self.n_raw_candidates,
            seconds=round(self.seconds, 2),
            solved=self.solved,
        )
        return out


def solve_by_proposal(
    solver,
    task: ArcTask,
    *,
    n_samples: int = 8,
    temperature: float = 0.8,
    max_new_tokens: int = 256,
    deadline: Optional[float] = None,
) -> ProgramSolveResult:
    """Ask the model for programs `n_samples` times, then verify them all.

    The first sample is greedy so the model's single best guess is always
    represented; the rest are sampled, because the point of drawing again is to
    get a different hypothesis rather than the same one twice.
    """
    started = time.perf_counter()
    messages = build_inference_messages(task)
    candidates: list[str] = []
    seen: set[str] = set()
    n_samples_done = 0

    n_parseable = 0
    for i in range(max(1, n_samples)):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        text = solver.complete(
            messages,
            temperature=0.0 if i == 0 else temperature,
            attempt=i,
            max_new_tokens=max_new_tokens,
        )
        n_samples_done += 1
        if text:
            for cand in extract_candidates(text):
                if cand in seen:
                    continue
                seen.add(cand)
                candidates.append(cand)
                if parse_program(cand) is not None:
                    n_parseable += 1
        if n_samples_done > ABANDON_AFTER and n_parseable == 0:
            break

    attempts, report = propose_and_verify(candidates, task, deadline=deadline)
    return ProgramSolveResult(
        attempts=attempts,
        report=report,
        n_samples=n_samples_done,
        n_raw_candidates=len(candidates),
        seconds=time.perf_counter() - started,
    )


def solve_step_by_step(
    solver,
    task: ArcTask,
    *,
    beam_width: int = 3,
    max_depth: int = 4,
    temperature: float = 0.7,
    max_new_tokens: int = 96,
    deadline: Optional[float] = None,
) -> ProgramSolveResult:
    """Execution-guided decoding, reported in the same shape as blind proposal.

    Each operator is run before the next is written, so the model composes
    against what actually happened rather than against what it assumed.
    """
    from src.hrps.execution_guided import search_execution_guided

    started = time.perf_counter()
    result = search_execution_guided(
        solver,
        task,
        beam_width=beam_width,
        max_depth=max_depth,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        deadline=deadline,
    )
    report = ProposalReport(
        n_candidates=result.report.n_ops_proposed,
        n_parsed=result.report.n_states_expanded,
        n_rejected=result.report.n_ops_rejected,
        n_verified=1 if result.solved else 0,
        n_distinct_outputs=1 if result.solved else 0,
        consensus=1.0 if result.solved else 0.0,
        winning_program=result.report.program,
        survivors=[result.report.program] if result.solved else [],
    )
    answers = result.answers()
    n_test = len(task.test_inputs())
    if not answers:
        answers = [None] * n_test
    return ProgramSolveResult(
        attempts=[list(answers), list(answers)],
        report=report,
        n_samples=result.report.n_model_calls,
        n_raw_candidates=result.report.n_ops_proposed,
        seconds=time.perf_counter() - started,
    )


def append_verified(
    task: ArcTask,
    programs: list[str],
    corpus: Path,
    *,
    known: Optional[set[str]] = None,
) -> int:
    """Append newly verified programs to the training corpus. Expert iteration.

    Each program is re-verified here rather than trusted from a caller's
    report, because this writes labels that later training will believe.
    Returns how many were actually appended.
    """
    corpus.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with corpus.open("a", encoding="utf-8") as handle:
        for text in programs:
            program = parse_program(text)
            if program is None:
                continue
            key = f"{task.task_id}\t{program.serialize()}"
            if known is not None and key in known:
                continue
            if not verify_program(program, task).is_total:
                continue
            handle.write(
                json.dumps(build_training_example(task, program.serialize())) + "\n"
            )
            if known is not None:
                known.add(key)
            written += 1
    return written

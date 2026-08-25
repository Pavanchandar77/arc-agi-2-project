"""Hierarchical Residual Program Synthesis (HRPS).

Phase 1 is the instrumented finite-DSL search microscope (A–G, then H).
Open-model elevation (M0–M3) lives in src.hrps.elevation. Bond training
(HRPS episodes → LoRA adapter → Bond+HRPS inference) lives in src.hrps.bond.
See src.hrps.conclusions for frozen Phase-1 statements.
"""

from src.hrps.conclusions import FROZEN_CONCLUSIONS, NEXT_CHANGE_POLICY
from src.hrps.dsl import Op, Program, operator_catalog, replay, stage_config
from src.hrps.grid import as_grid, grids_equal
from src.hrps.search import SearchBudget, search_task
from src.hrps.task import ArcTask, iter_split, load_task_file, parse_task

__all__ = [
    "ArcTask",
    "FROZEN_CONCLUSIONS",
    "NEXT_CHANGE_POLICY",
    "Op",
    "Program",
    "SearchBudget",
    "as_grid",
    "grids_equal",
    "iter_split",
    "load_task_file",
    "operator_catalog",
    "parse_task",
    "replay",
    "search_task",
    "stage_config",
]

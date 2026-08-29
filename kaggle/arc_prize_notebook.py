"""ARC Prize submission notebook — paste these cells into a Kaggle notebook.

Each cell marker below starts a new notebook cell. The whole thing runs offline
on CPU: no internet, no accelerator, no pip install.

Two ways to get the solver into the notebook:

  A. Bundle (no upload). Add `kaggle/arc_bundle.py` to the notebook as a
     Utility Script, or paste its contents into a cell. Cell 1 finds it.
  B. Dataset. Upload the repo as a Kaggle Dataset and attach it; cell 1
     finds `/kaggle/input/<anything>/src/kaggle_run.py` on its own.

Before submitting, confirm the notebook settings: Internet **off**,
Accelerator **None**, and the ARC Prize competition dataset attached.
"""

# %% [cell] 1. Load the solver
import json
import os
import sys
import time
from pathlib import Path

t_start = time.time()


def _load_solver():
    """Find the solver by bundle or by dataset, in that order."""
    for name in ("arc_bundle",):
        try:
            __import__(name)
            return f"bundle:{name}"
        except ImportError:
            pass
    candidates = [Path("/kaggle/input"), Path("/kaggle/working"), Path(".")]
    for base in candidates:
        if not base.is_dir():
            continue
        for hit in sorted(base.glob("*/src/kaggle_run.py")) + sorted(base.glob("src/kaggle_run.py")):
            root = str(hit.parent.parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return f"path:{root}"
    raise RuntimeError(
        "solver not found. Either attach kaggle/arc_bundle.py as a Utility Script, "
        "or upload the repo as a Dataset so that <dataset>/src/kaggle_run.py exists."
    )


print("solver source:", _load_solver())
from src.kaggle_run import build_config, find_challenges, run, validate_submission  # noqa: E402

print("challenges:", find_challenges())


# %% [cell] 2. Run
# Kaggle allows 12 hours. Stop at 11 so the commit and save always fit; the
# runner writes submission.json incrementally, so an early kill still leaves a
# complete file. Set ARC_TOTAL_SECONDS to override.
TOTAL_SECONDS = float(os.environ.get("ARC_TOTAL_SECONDS", 11 * 3600))
PER_TASK_SECONDS = 60.0

config = build_config(
    [
        "--output", "/kaggle/working/submission.json",
        "--total-seconds", str(TOTAL_SECONDS),
        "--per-task-seconds", str(PER_TASK_SECONDS),
        "--workers", str(os.cpu_count() or 4),
    ]
)
report = run(config)
print(json.dumps(report, indent=2))


# %% [cell] 3. Verify before submitting
# A missing task id scores the entire submission zero, so check explicitly
# rather than trusting the run report.
submission = json.loads(Path("/kaggle/working/submission.json").read_text())
expected = sorted(json.loads(Path(find_challenges()).read_text()))

problems = validate_submission(submission, expected)
print(f"tasks in submission: {len(submission)} / {len(expected)} expected")
print(f"schema problems: {problems if problems else 'none'}")
print(f"total notebook time: {(time.time() - t_start) / 60:.1f} min")
assert not problems, problems
assert len(submission) == len(expected), "submission is missing task ids"
print("submission.json is ready")

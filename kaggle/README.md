# Running on Kaggle

The offline solver is pure standard library. No internet, no GPU, no pip
install, no model weights. It runs inside the ARC Prize notebook constraints as
they actually are.

## Setup, in full

1. Create a new Kaggle notebook on the ARC Prize competition.
2. **Settings → Accelerator: None.** The offline path is CPU only; an
   accelerator just spends your GPU quota.
3. **Settings → Internet: Off.** Required for competition notebooks. Nothing
   here downloads anything, so this changes no behaviour.
4. Confirm the competition dataset is attached, so
   `/kaggle/input/<slug>/arc-agi_test_challenges.json` exists.
5. Get the solver into the notebook by one of the two routes below.
6. Paste the cells from `arc_prize_notebook.py`, then Run All, then
   **Save Version → Save & Run All (Commit)**, and submit the committed
   version's `submission.json`.

### Route A — bundle (nothing to upload)

`arc_bundle.py` is a single generated file that carries the whole solver. Add
it to the notebook as a Utility Script, or paste its contents into the first
cell. Cell 1 of the notebook picks it up automatically.

Regenerate it after any change to `src/`:

```bash
python scripts/build_kaggle_bundle.py
```

### Route B — dataset

Upload this repository as a Kaggle Dataset and attach it. The notebook finds
any `/kaggle/input/*/src/kaggle_run.py` on its own, so the dataset slug does
not matter.

## What the runner guarantees

These are the failure modes that cost people their submission, and what the
runner does about each:

| Risk | Handling |
|---|---|
| Notebook killed at the time limit | Global deadline defaults to 11h of the 12h allowance; `submission.json` is rewritten every 10 tasks, so a kill at any moment leaves a complete file |
| One task crashes the run | Every task is solved in a worker process; an exception becomes a placeholder entry |
| One task hangs forever | Cooperative deadline inside the solver, plus a SIGALRM hard stop in the worker, plus `pool.terminate()` at the global deadline |
| A task id missing from the output | Every id is written as a placeholder before solving starts, then overwritten with real answers |
| Malformed grid (ragged, >30, colour out of 0–9) | `validate_submission` checks the file and replaces any offending entry |
| Dataset path differs from expectation | `find_challenges` searches `/kaggle/input`, `input`, and `data` for both the 2025 and 2024 file names |

Cell 3 of the notebook re-validates the written file and asserts, so a bad
submission fails loudly in the notebook rather than silently on the
leaderboard.

## Local dry run

Point it at any local ARC split to reproduce the Kaggle behaviour exactly,
including scoring:

```bash
python -m src.kaggle_run \
  --challenges path/to/arc-agi_evaluation_challenges.json \
  --solutions  path/to/arc-agi_evaluation_solutions.json \
  --output /tmp/submission.json \
  --total-seconds 1200 --per-task-seconds 30 --workers 4
```

Useful flags: `--no-search` (solver bank only, seconds instead of minutes),
`--limit N` (first N tasks), `--workers 1` (serial, for debugging tracebacks).

## When you add the LLM

The offline solver is the floor, not the score. Two constraints shape how the
neural path has to be built here, and both are easy to discover too late:

* **Weights must arrive as a Kaggle Dataset.** Internet is off, so
  `from_pretrained("org/model")` cannot download anything. Upload the
  checkpoint as a Dataset and load it from `/kaggle/input/<slug>/`.
* **Test-time training has to fit the same wall clock.** Per-task adaptation
  plus generation, times the number of tasks, has to land inside the limit with
  room to spare. Budget it the way `kaggle_run` budgets the symbolic path:
  a global deadline, a per-task cap, and a fallback whenever a task overruns.

Keep the solver bank in front of the model. It is fast, and when it fires its
answer is verified against every demonstration rather than sampled, so it
should win any disagreement.

## Tuning the budget

`--per-task-seconds` caps any single task; the runner also computes a fair
share from the time and tasks remaining and takes the smaller of the two. So
raising the global budget cannot cause an overrun — it only lets slow tasks use
more of it. Note that the DSL search rarely converts extra time into extra
solves past roughly 30s per task; the solver bank finishes in well under a
second per task.

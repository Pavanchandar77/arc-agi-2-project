# Bond implementation report

The current repo is **not** a finished general HRPS overseer. It now has a **domain-neutral core** (`src/hrps/core.py`) plus an **ARC adapter** (`src/hrps/arc_adapter.py`) that preserves the existing exact DSL executor and verifier. ARC is the first proof domain. Generality is a future empirical result.

Public identity is **Bond**. The Qwen foundation is private provenance: keep those weights, do not delete them, and do not use the Qwen name as the runtime model.

This document separates **infrastructure** from **learned Bond results**. There is no trained Bond checkpoint in this environment.

## Thesis

Bond is not Qwen wrapped by a verifier. The intended system is:

```
Qwen foundation
  + HRPS-generated reasoning episodes
  + learned Bond adapter
  + active HRPS reasoning loop
  + exact executor and verifier
  = Bond-Qwen
```

The model inspects, hypothesizes, acts, reads exact residuals, revises, and commits. HRPS is the cognitive substrate. The verifier is exact feedback, not the answer.

## Module map

| Module | Role |
| --- | --- |
| `src/hrps/backend.py` | Foundation registry: `qwen1.5b_smoke`, `qwen3.5_4b`, configurable `inkling_small`. Hardware gate. Offline load. |
| `src/hrps/model.py` | HF client: adapter load, seed, top-p, `local_files_only`. |
| `src/hrps/schema.py` | Strict JSON actions. Rejects unknown names, ops, Python, shell, network, hidden labels. |
| `src/hrps/env.py` | Exact inspect/apply/residual/commit, including `inspect relations`. Gold-free observations. |
| `src/hrps/runner.py` | Active loop with call, token, and wall-clock budgets and interaction logs. |
| `src/hrps/episodes.py` | Teacher traces → SFT and JSON-action SFT. Hold-out / eval / gold leakage assertions. |
| `src/hrps/bond.py` | LoRA trainer CLI, four-way evaluator (JSON+CSV), `aabf363d` probe, manifest. |
| `src/hrps/dsl.py` / `residual.py` | Unchanged exact executor and verifier. |

## Four systems

| System | Meaning |
| --- | --- |
| `base_direct` | Frozen foundation, raw grids, no HRPS. |
| `base_hrps` | Same frozen foundation, JSON HRPS loop, no adapter. |
| `bond_direct` | Foundation + Bond adapter, raw grids. |
| `bond_hrps` | Adapter + active HRPS loop. |

Deltas: `Δ_train = bond_direct - base_direct`, `Δ_substrate = bond_hrps - bond_direct`.

## Foundations

| Id | HF id | Local | Final Bond? |
| --- | --- | --- | --- |
| `qwen1.5b_smoke` | `Qwen/Qwen2.5-1.5B-Instruct` | yes (8GB, CPU) | **no** — smoke only |
| `qwen3.5_4b` | `Qwen/Qwen3.5-4B` | no (train on remote GPU) | only after adapter training + four-way eval |
| `inkling_small` | env `HRPS_INKLING_ID` or `thinkingmachines/Inkling-Small` | no | not required for tests or 1.5B smoke |

This host has ~8GB RAM and no GPU. The software path is tested locally with scripted backends and the 1.5B **label**. Weights are not claimed to have been trained here.

## Commands

Local smoke (no torch required):

```
python -m src.hrps.bond smoke
```

Local 1.5B adapter training (needs torch; still not final Bond):

```
python -m src.hrps.bond train --foundation qwen1.5b_smoke --episodes artifacts/bond/sft_actions.jsonl --output-dir artifacts/bond --adapter models/bond_adapter --seed 42 --max-seq-length 2048 --learning-rate 2e-4 --epochs 3 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --holdout-spec training[400:440]
```

Remote Qwen3.5-4B training:

```
python -m src.hrps.bond train --foundation qwen3.5_4b --adapter models/bond_qwen35_4b --seed 42 --epochs 3 --lora-r 16 --lora-alpha 32
```

Held-out evaluation:

```
python -m src.hrps.bond eval --foundation qwen1.5b_smoke --n 40 --offset 400 --adapter models/bond_adapter
```

## Identity

- Runtime name with adapter: `Bond` (smoke adapter: `Bond-smoke`).
- Without adapter: `Bond adapter not found`. The bare foundation is `foundation`, not Bond.
- Merge writes `models/bond_merged/` and refuses to overwrite the foundation directory.
- Package layout: `bond_model/`, `bond_controller/`, `bond_interface/`, `bond_hrps/`, `bond_executor/`, `bond_verifier/`, `bond_manifest.json`.

```
python -m src.hrps.bond package
python -m src.hrps.bond bundle
python -m src.hrps.bond merge --foundation-dir models/foundation --adapter models/bond_qwen35_4b --merge-out models/bond_merged
python -m src.hrps.bond generate --scale train   # larger set; does not overwrite the pinned 62
```

## Honesty

- No adapter weights were produced in this runtime (`no_torch`).
- Scripted four-way eval validates the **pipeline**, not learned reasoning.
- Do not report a smoke 1.5B score as Bond capability.
- Do not report a Bond+HRPS score as a model gain unless `bond_direct` beats `base_direct` after adapter training.

## Experimental boundaries preserved

No new DSL operators. No public-evaluation tuning. No task-specific patches. No unrestricted code execution. `aabf363d` remains a gold-free underconstraint lesson, not a hard-coded solution. Hold-out `[400:440]` is excluded from training.

# Multi-Agent Mobile UI Testing Pipeline — Sustainability Benchmark

A 5-agent pipeline that processes ENRICO dataset screenshots through UI perception,
test case generation, metamorphic relation generation, test optimization, and
energy-aware suite reduction — benchmarked across multiple small language models
(SLMs) for energy/emissions cost using CodeCarbon.

## Pipeline overview

| Agent   | Task                                                                                                                                                                  | Model role                                          |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Agent 1 | UI Perception — describes screen components, layout, accessibility                                                                                                  | Vision-language model (VLM)                          |
| Agent 2 | Test Case Generation — functional/negative/accessibility test cases                                                                                                 | Text LLM                                             |
| Agent 3 | Metamorphic Relation Generation — 6-category taxonomy (INVARIANCE, MONOTONICITY, INPUT_TRANSFORMATION, VALIDATION_CONSISTENCY, INTERACTION_CONSISTENCY, ROBUSTNESS) | Text LLM + deterministic backfill code               |
| Agent 4 | Optimization/Prioritization — keep/reduce/deprioritize decision per MR                                                                                              | Text LLM + deterministic coverage-rule enforcement   |
| Agent 5 | Suite Reduction & Energy Savings Estimator — assigns each optimized MR an execution tier (always/sampled/optional/dropped) + cycle frequency                       | Text LLM, constrained by a hard-coded allowed-tier set per Agent 4 decision, with a deterministic fallback mapping if the LLM output fails to parse |

**Note on Agents 3, 4, and 5:** each layers deterministic post-processing on top of
the LLM call. Agent 3 has backfill functions that generate an entire metamorphic
relation in code if the LLM misses a category. Agent 4 has a coverage-rule
enforcement layer that can override the LLM's decision. Agent 5's energy/percentage
math is always computed in plain Python from the assigned tier — never by the LLM
— and falls back to a fixed `DECISION_TO_TIER_FALLBACK` mapping if its own LLM call
can't be parsed.

## Two ways to run this

`batch_runner.py` (the wrapper that looped `main.py` in batches) is gone — its
job is now built into `benchmark_pipeline.py` itself. `main.py` stays as a
simple, single-model backup option:

| | `main.py` | `benchmark_pipeline.py` |
| --- | --- | --- |
| Purpose | Simple, single-model backup — process the full dataset once, with Qwen | Compare energy/emissions across models |
| Models | Qwen only (hard-coded) | `qwen`, `phi35`, `internvl` (see `MODELS` list) |
| Resume support | Yes — checkpoints to `outputs/completed_screens.txt` | Yes — checkpoints to `outputs/<model>/run_N/completed_screens.txt`, per run |
| Batching | `--batch-size N` processes the next N unfinished images | `--batch-size N` per run, plus `--max-batches` to cap an invocation |
| Runs | Always a single pass | `--runs` timed repeats (default 3) per model |
| Output layout | Flat `outputs/` | Namespaced `outputs/<model_name>/run_N/` |
| Warmup stabilization | No | Yes — 300s wait after warmup before the first timed run, so leftover GPU/driver activity doesn't skew Run 1 |
| `--fresh` | Wipes `outputs/` entirely | Wipes `outputs/<model_name>/` and that model's emissions log only |

Use `main.py` when you just want one clean, resumable Qwen pass without the
multi-model machinery. Use `benchmark_pipeline.py` when you're comparing
models or want the finer-grained per-run/per-batch output structure.

```powershell
# benchmark_pipeline.py
python benchmark_pipeline.py --model qwen
python benchmark_pipeline.py --model phi35 --batch-size 100
python benchmark_pipeline.py --model internvl --batch-size 100 --max-batches 3   # quick test
python benchmark_pipeline.py --model qwen --runs 1
python benchmark_pipeline.py --model qwen --fresh
```

⚠️ **`--fresh` on `benchmark_pipeline.py` deletes `outputs/<model_name>/`
entirely, including its checkpoints and batch logs**, plus that model's
emissions log. **`--fresh` on `main.py` deletes all of `outputs/`, including
its checkpoint file.** Only pass either if you actually want to discard
progress and start over.

## Models benchmarked

| Model pair | Vision model            | Text model            | Params (approx)                  |
| ---------- | ------------------------ | ---------------------- | --------------------------------- |
| Qwen       | Qwen2-VL-7B-Instruct     | Qwen2.5-7B-Instruct    | 7B / 7B                           |
| Phi-3.5    | Phi-3.5-vision-instruct  | Phi-3.5-mini-instruct  | 4.2B / 3.8B                       |
| InternVL2  | InternVL2-8B             | internlm2_5-7b-chat    | 8B / 7B (matched pair, same lab)  |

All three are wired into `benchmark_pipeline.py`'s `MODELS` list — pick one with
`--model qwen` / `--model phi35` / `--model internvl`. Each does `--runs` timed
passes (default 3) across every screenshot in `images/`, with per-agent energy
(kWh), emissions (kg CO2), and duration logged via CodeCarbon.

## Repository structure

```
├── main.py                        # Simple single-model backup — full-dataset run with Qwen,
│                                   #   resumable via checkpoint + --batch-size, flat outputs/
├── benchmark_pipeline.py          # Multi-model comparison runner — resumable, batched, per-model outputs
├── breakeven_check.py             # Reads outputs from a main.py run; computes how many regression
│                                   #   cycles it takes for Agent 5's own generation cost to pay for
│                                   #   itself via its projected per-cycle savings
├── summarize_emissions.py         # Aggregates outputs/emissions_log_*.csv (from benchmark_pipeline.py)
│                                   #   into summary tables + chart
├── design_topics.csv              # Screen ID -> topic mapping (ENRICO metadata)
├── images/                        # Input screenshots — NOT tracked in git, populate this yourself
├── stages/
│   ├── ui_analysis.py             # Agent 1 — Qwen2-VL
│   ├── ui_analysis_phi.py         # Agent 1 — Phi-3.5-vision
│   ├── ui_analysis_internvl.py    # Agent 1 — InternVL2-8B
│   ├── test_generation.py         # Agent 2 (shared) — also Qwen's load_text_model()
│   ├── text_model_phi.py          # Agent 2/3/4/5 text model loader — Phi-3.5-mini
│   ├── text_model_internlm.py     # Agent 2/3/4/5 text model loader — internlm2_5-7b-chat
│   ├── metamorphic_testing.py     # Agent 3 (shared)
│   ├── optimization.py            # Agent 4 (shared)
│   └── optimization_reduction.py  # Agent 5 (shared)
├── test_agent1.py .. test_agent4.py  # Smoke tests, one agent (or agent chain) deeper each time
├── test_agent5.py                 # Smoke test: Agent 5 over outputs/optimized_relations/, any --model
└── outputs/                       # Not tracked in git — created by main.py / benchmark_pipeline.py
    ├── completed_screens.txt      # main.py's checkpoint file
    ├── emissions_log.csv          # main.py's single-model emissions log
    ├── emissions_log_<model>.csv  # benchmark_pipeline.py's per-model emissions log
    └── <model_name>/
        └── run_1 .. run_N/        # benchmark_pipeline.py per-run outputs
            ├── completed_screens.txt   # Checkpoint — which screens finished all 5 agents
            ├── batch_log.csv           # One row per batch: timing, images processed, status
            ├── next_batch_id.txt       # Persistent batch-id counter (survives Ctrl+C/restart)
            └── testcases_master.csv, metamorphic_relations_master.csv, etc.
```

**Note:** `stages/test_generation.py`'s `load_text_model()` doubles as both the
Agent 2 stage file and Qwen's text-model loader — Phi/InternVL have their own
dedicated `text_model_<name>.py` loader instead.

## Why two virtual environments

Qwen requires a modern `transformers` version. Phi-3.5-vision, Phi-3.5-mini,
InternVL2-8B, and internlm2_5-7b-chat all rely on custom `trust_remote_code=True`
modeling code written against an older `transformers` internal API that was never
updated by the model authors — running them under a modern `transformers` version
raises cache-API or attribute errors (`DynamicCache` / `all_tied_weights_keys`
depending on the model). Pinning `transformers==4.46.1` in a separate venv resolves
this for Phi-3.5-vision, Phi-3.5-mini, and InternVL2-8B.

| Venv           | Used for            | Key package versions                          |
| -------------- | -------------------- | --------------------------------------------- |
| `.venv`        | Qwen                  | `transformers==5.8.1`, `torch==2.11.0+cu128`  |
| `.venv-vision` | Phi-3.5, InternVL2    | `transformers==4.46.1`, `torch==2.11.0+cu128` |

## Setup

### 1. Qwen environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch transformers accelerate pillow qwen-vl-utils json_repair codecarbon matplotlib pandas
```

### 2. Phi-3.5 / InternVL2 environment

```powershell
python -m venv .venv-vision
.\.venv-vision\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.46.1"
pip install accelerate pillow json_repair codecarbon einops timm sentencepiece protobuf matplotlib pandas
```

Place your ENRICO screenshots in an `images/` folder at the repo root before
running anything — it's gitignored, so it won't come from `git clone`.

## Running the full dataset (simple backup: `main.py`, Qwen)

```powershell
# process the next 1000 not-yet-completed images
python main.py --batch-size 1000

# process everything still remaining
python main.py

# start completely over — wipes outputs/, including the checkpoint (use with care)
python main.py --fresh
```

Each run prints how many screens are already done, how many remain, and picks up
exactly where the last run left off via `outputs/completed_screens.txt`. A screen
is only marked complete after **all 5 agents** succeed on it — a failure mid-screen
leaves it unmarked so it's retried on the next run.

After a run, check the generation-vs-savings tradeoff:

```powershell
python breakeven_check.py
```

## Comparing models (`benchmark_pipeline.py`)

Each model must be run **separately, from its own venv** — Qwen and the other
models can't coexist in the same Python environment (see above).

```powershell
# Qwen — from .venv
python benchmark_pipeline.py --model qwen

# Phi-3.5 — from .venv-vision
python benchmark_pipeline.py --model phi35

# InternVL2 — from .venv-vision
python benchmark_pipeline.py --model internvl
```

By default this does `--runs` (3) timed passes over every image in `images/`,
each run processed in one batch, with a 60-second cooldown between runs and a
300-second stabilization wait after warmup before Run 1 starts. All agents use
greedy decoding (`do_sample=False`), so the runs measure energy/timing
stability, not output variance.

To process in chunks instead (recommended for large datasets, since it's
checkpointed after every batch):

```powershell
python benchmark_pipeline.py --model qwen --batch-size 200
```

Stop it (Ctrl+C) and re-run the exact same command any time — it picks up from
`outputs/qwen/run_N/completed_screens.txt` rather than starting over.

For a quick smoke test without committing to a full run:

```powershell
python benchmark_pipeline.py --model qwen --batch-size 50 --max-batches 1
```

### Smoke-testing a single agent (any model)

```powershell
python test_agent1.py --model qwen        # Agent 1 only
python test_agent2.py --model phi35       # through Agent 2
python test_agent3.py --model internvl    # through Agent 3
python test_agent4.py --model qwen        # through Agent 4
python test_agent5.py --model qwen        # Agent 5 over an existing outputs/optimized_relations/ run
```

## Summarizing results across models

Once you've benchmarked the models you want to compare:

```powershell
python summarize_emissions.py --by-screen
```

Auto-discovers every `outputs/emissions_log_*.csv`, merges them, and produces:

- `outputs/emissions_summary_by_agent.csv` — mean/std energy, emissions, duration per model x agent
- `outputs/emissions_summary_by_run.csv` — total per model x run (sanity check across runs)
- `outputs/emissions_summary_by_screen.csv` — per-screen breakdown (with `--by-screen`)
- `outputs/emissions_energy_by_agent.png` — grouped bar chart

Flags: `--log <path>` (repeatable) to point at specific logs, `--log-dir` to change
where it searches (default `outputs/`), `--out` to change where summaries are written.
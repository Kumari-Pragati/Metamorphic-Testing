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

The repo has **two separate entry points** — pick based on what you're doing:

| | `main.py` | `benchmark_pipeline.py` |
| --- | --- | --- |
| Purpose | Process the full dataset once, with Qwen | Compare energy/emissions across models |
| Models | Qwen only (hard-coded) | `qwen`, `phi35` (see `MODELS` list — InternVL2 has its own `stages/ui_analysis_internvl.py` and `stages/text_model_internlm.py` but isn't wired into this script's `MODELS` list yet) |
| Resume support | **Yes** — checkpoints to `outputs/completed_screens.txt`, skips already-completed screens on the next run | **No** — every invocation wipes `outputs/<model_name>/` and that model's emissions log and starts over, by design |
| Batching | `--batch-size N` processes the next N unfinished images | Not batched — always runs the full `images/` folder, `--runs` times |

⚠️ **`main.py --fresh` deletes `outputs/` entirely, including the checkpoint file.**
Only pass `--fresh` if you actually want to discard all progress and start over —
never combine it with `--batch-size` out of habit.

## Models benchmarked (via `benchmark_pipeline.py`)

| Model pair | Vision model            | Text model            | Params (approx)                  |
| ---------- | ------------------------ | ---------------------- | --------------------------------- |
| Qwen       | Qwen2-VL-7B-Instruct     | Qwen2.5-7B-Instruct    | 7B / 7B                           |
| Phi-3.5    | Phi-3.5-vision-instruct  | Phi-3.5-mini-instruct  | 4.2B / 3.8B                       |
| InternVL2  | InternVL2-8B             | internlm2_5-7b-chat    | 8B / 7B (matched pair, same lab) — stage files exist but not yet in `benchmark_pipeline.py`'s `MODELS` list |

`benchmark_pipeline.py` runs each configured model **5 timed passes** (`--runs` to
override) across every screenshot in `images/`, with per-agent energy (kWh),
emissions (kg CO2), and duration logged via CodeCarbon.

## Repository structure

```
├── main.py                        # Full-dataset run with Qwen, resumable via checkpoint + --batch-size
├── benchmark_pipeline.py          # Multi-model comparison runner (qwen, phi35) — no resume, wipes outputs each run
├── breakeven_check.py             # Reads outputs from a main.py run; computes how many regression
│                                   #   cycles it takes for Agent 5's own generation cost to pay for
│                                   #   itself via its projected per-cycle savings
├── summarize_emissions.py         # Aggregates outputs/emissions_log_*.csv (from benchmark_pipeline.py)
│                                   #   into summary tables + chart
├── design_topics.csv              # Screen ID -> topic mapping (ENRICO metadata)
├── images/                        # Input screenshots — NOT tracked in git, populate this yourself
├── images_done/                   # Tracked sample images left in git history — not read by either script
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
    └── <model_name>/run_1 .. run_5/   # benchmark_pipeline.py per-run outputs
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

## Running the full dataset (recommended: `main.py`, Qwen)

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
```

Every invocation **wipes and rebuilds** `outputs/<model_name>/` and
`outputs/emissions_log_<model_name>.csv` from scratch — there's no resume here,
so let it finish once you start it. Each run does 5 timed passes (`--runs` to
change) across every image in `images/`, with a 60-second cooldown between runs.
All agents use greedy decoding (`do_sample=False`), so the 5 runs measure
energy/timing stability, not output variance.

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
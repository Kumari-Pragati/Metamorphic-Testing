# benchmark_pipeline.py
#
# Runs the 5-agent pipeline for ONE model (qwen / phi35 / internvl), across
# NUM_RUNS repeats of the full image set, with per-run checkpointing so you
# can Ctrl+C and resume, optional batching, and per-agent emissions tracking.
# Replaces the old main.py + batch_runner.py — everything they did is in here.
#
# Usage:
#   python benchmark_pipeline.py --model qwen
#       Process all images, 3 runs (default), each run in one batch.
#
#   python benchmark_pipeline.py --model phi35 --batch-size 100
#       Same, but each run is processed 100 images at a time, checkpointed
#       after every batch — safe to Ctrl+C and re-run to resume.
#
#   python benchmark_pipeline.py --model internvl --batch-size 100 --max-batches 3
#       Quick test: only run 3 batches this invocation, then stop (even if
#       the run isn't finished). Re-run the same command to continue.
#
#   python benchmark_pipeline.py --model qwen --runs 1
#       Only do a single pass over the dataset instead of the default 3.
#
#   python benchmark_pipeline.py --model qwen --fresh
#       Wipe outputs/qwen/ and its emissions log, then start over from
#       scratch (normally the script auto-resumes instead).
#
# Run once per model, each from that model's own venv (see the MODELS
# comment below for why qwen/phi35/internvl can't share a venv).

import argparse
import os
import csv
import time
import shutil
import traceback
from codecarbon import EmissionsTracker

# ─── Config ───────────────────────────────────────────────────────────────────

IMAGES_DIR   = "images/"
TOPICS_CSV   = "design_topics.csv"

# Number of full pipeline runs (repeats) per model
NUM_RUNS = 3

# Cooldown between runs in seconds — lets GPU thermals/power stabilize.
# Only applied before a run that's actually about to do work (see below).
COOLDOWN_SECONDS = 60

# Wait after warmup (before the first timed run) in seconds. Warmup itself
# triggers a burst of GPU/driver activity (kernel compilation, memory
# allocator warm-up, clock ramp-up) that can bleed into the first timed run
# and skew its energy numbers if you go straight from warmup into Run 1.
# This just lets things settle first. Not applied again between runs —
# COOLDOWN_SECONDS already handles that.
WARMUP_STABILIZE_SECONDS = 300

# Models to benchmark — add more entries here as you test new SLMs.
# Each entry is a dict with:
#   "name"        → short label used in emissions CSV and output dirs
#   "vlm_module"  → dotted import path for the Agent 1 module
#   "llm_module"  → dotted import path for the shared text model loader
#
# IMPORTANT: qwen, phi35, and internvl each require DIFFERENT transformers
# versions and cannot all be installed in the same venv:
#   - Phi-3.5-vision's remote code only works on transformers<=4.46.x
#     (never patched for newer cache APIs)
#   - InternVL2-8B / InternLM2.5 are the same mid-2024 generation of custom
#     trust_remote_code releases as Phi-3.5 — treat a modern transformers
#     version as a likely incompatibility risk too. If you hit a cache/
#     attribute error, pin transformers similarly (check the model card /
#     GitHub issues for the exact working version, since this can shift).
#   - Also confirm internlm2_5-7b-chat's tokenizer actually has a working
#     chat_template before running the full benchmark — its warmup() will
#     raise a clear error if not; see stages/text_model_internlm.py.
# This script runs ONE model per invocation — pick it with --model,
# and run this script once from each model's own venv.
MODELS = [
    {
        "name":       "qwen",
        "vlm_module": "stages.ui_analysis",
        "llm_module": "stages.test_generation",  # load_text_model lives here
    },
    {
        "name":       "phi35",
        "vlm_module": "stages.ui_analysis_phi",
        "llm_module": "stages.text_model_phi",
    },
    {
        "name":       "internvl",
        "vlm_module": "stages.ui_analysis_internvl",
        "llm_module": "stages.text_model_internlm",
    },
]

MODEL_NAMES = [m["name"] for m in MODELS]

MASTER_FILE_NAMES = [
    "testcases_master.csv",
    "metamorphic_relations_master.csv",
    "optimized_relations_master.csv",
    "reduced_suite_master.csv",
    "energy_savings_summary.csv",
]

BATCH_LOG_COLUMNS = [
    "batch_id", "model_name", "run_number", "batch_size_requested",
    "started_at", "ended_at", "duration_sec",
    "completed_before", "completed_after", "images_processed_this_batch",
    "total_images", "status",
]

# ─── CLI args ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Run the 5-agent pipeline benchmark for ONE model, processing "
                "each run's images in resumable, checkpointed batches. Run this "
                "script once per model, each from that model's own venv. Safe "
                "to stop (Ctrl+C) and re-run at any time — it resumes from "
                "whatever the checkpoint says is left."
)
parser.add_argument(
    "--model",
    required=True,
    choices=MODEL_NAMES,
    help=f"Which model config to benchmark. Options: {MODEL_NAMES}",
)
parser.add_argument(
    "--runs",
    type=int,
    default=NUM_RUNS,
    help=f"Number of timed repeats over the full dataset (default: {NUM_RUNS})",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=None,
    help="Images processed per batch within a run. Omit to process all "
         "remaining images for a run in a single batch (old behavior, just "
         "now checkpointed/resumable).",
)
parser.add_argument(
    "--max-batches",
    type=int,
    default=None,
    help="Cap on number of batches PER RUN this invocation will do (e.g. for "
         "a quick test). If a run doesn't finish because of this cap, the "
         "script stops rather than starting the next repeat on an "
         "incomplete one — re-run the same command to continue.",
)
parser.add_argument(
    "--fresh",
    action="store_true",
    help="Wipe this model's entire outputs/<model>/ tree (all runs, "
         "checkpoints, batch logs) and its emissions log, then start over.",
)
args = parser.parse_args()

NUM_RUNS = args.runs
model_cfg = next(m for m in MODELS if m["name"] == args.model)
model_name = model_cfg["name"]

MODEL_DIR = f"outputs/{model_name}"

# Each model gets its OWN emissions log file so that benchmarking one model
# never overwrites or wipes out another model's results. summarize_emissions.py
# automatically finds and merges every emissions_log_*.csv file.
EMISSIONS_LOG     = f"outputs/emissions_log_{model_name}.csv"
EMISSIONS_COLUMNS = [
    "model_name", "run_number", "screen_id", "agent",
    "energy_kwh", "emissions_kg_co2", "total_emissions_kg_co2", "duration_seconds",
    "ram_power_w", "cpu_power_w", "gpu_power_w",
    "ram_energy_kwh", "cpu_energy_kwh", "gpu_energy_kwh",
    "region", "country_name", "country_iso_code",
]


def _init_emissions_log():
    os.makedirs("outputs", exist_ok=True)
    with open(EMISSIONS_LOG, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EMISSIONS_COLUMNS).writeheader()


def _log_emissions(run_number: int, screen_id: str, agent: str,
                    tracker: EmissionsTracker, duration_s: float):
    emissions_kg = tracker.stop()
    data         = tracker.final_emissions_data
    energy_kwh   = data.energy_consumed if data else 0.0

    with open(EMISSIONS_LOG, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EMISSIONS_COLUMNS).writerow({
            "model_name":             model_name,
            "run_number":             run_number,
            "screen_id":              screen_id,
            "agent":                  agent,
            "energy_kwh":             round(energy_kwh, 6),
            "emissions_kg_co2":       round(emissions_kg or 0.0, 8),
            "total_emissions_kg_co2": round(emissions_kg or 0.0, 8),
            "duration_seconds":       round(duration_s, 2),
            "ram_power_w":            round(data.ram_power, 4)         if data else 0.0,
            "cpu_power_w":            round(data.cpu_power, 4)         if data else 0.0,
            "gpu_power_w":            round(data.gpu_power, 4)         if data else 0.0,
            "ram_energy_kwh":         round(data.ram_energy, 6)        if data else 0.0,
            "cpu_energy_kwh":         round(data.cpu_energy, 6)        if data else 0.0,
            "gpu_energy_kwh":         round(data.gpu_energy, 6)        if data else 0.0,
            "region":                 (data.region or "unknown")        if data else "unknown",
            "country_name":           (data.country_name or "unknown")  if data else "unknown",
            "country_iso_code":       (data.country_iso_code or "unknown") if data else "unknown",
        })

    print(f"   🌱 {agent} — {round(emissions_kg or 0.0, 8)} kg CO2 | "
          f"{round(energy_kwh, 6)} kWh | {round(duration_s, 2)}s | "
          f"GPU: {round(data.gpu_power, 2) if data else 0.0}W | "
          f"region: {data.region if data else 'unknown'}")


# ─── Dynamic import helper ────────────────────────────────────────────────────

def _import(module_path: str, attr: str):
    """Import a single attribute from a dotted module path."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


# ─── Per-run checkpoint / resume / rollback (adapted from main.py) ───────────

def _checkpoint_path(run_out_dir: str) -> str:
    return f"{run_out_dir}/completed_screens.txt"


def _load_checkpoint(run_out_dir: str) -> set:
    path = _checkpoint_path(run_out_dir)
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def _mark_completed(run_out_dir: str, screen_id: str):
    with open(_checkpoint_path(run_out_dir), "a", encoding="utf-8") as f:
        f.write(screen_id + "\n")


def _rollback_incomplete_screens(run_idx: int, run_out_dir: str, completed: set):
    """
    Remove rows left behind by a screen that started but never reached
    _mark_completed() for this run (e.g. Ctrl+C mid-pipeline). Safe to run
    every startup: a screen_id found in these files that is NOT in this
    run's checkpoint is by definition partial, since completion is only
    recorded after all 5 agents succeed. Also backs out that screen's
    emissions rows for this run_number so reprocessing it doesn't
    double-count energy or CO2.
    """
    # This run's own master CSVs — every row in these belongs to run_idx,
    # so we just filter by whether the screen made it into the checkpoint.
    for fname in MASTER_FILE_NAMES:
        path = f"{run_out_dir}/{fname}"
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if not fieldnames or "screen_id" not in fieldnames:
            continue
        kept = [r for r in rows if r.get("screen_id") in completed]
        dropped = len(rows) - len(kept)
        if dropped:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(kept)
            print(f"🧹 Rolled back {dropped} partial row(s) from {path}")

    # The shared emissions log holds rows for every run — only touch rows
    # matching this run_number.
    if os.path.exists(EMISSIONS_LOG):
        with open(EMISSIONS_LOG, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if fieldnames and "screen_id" in fieldnames and "run_number" in fieldnames:
            kept = [
                r for r in rows
                if not (r.get("run_number") == str(run_idx) and r.get("screen_id") not in completed)
            ]
            dropped = len(rows) - len(kept)
            if dropped:
                with open(EMISSIONS_LOG, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(kept)
                print(f"🧹 Rolled back {dropped} partial emissions row(s) for run {run_idx}")


# ─── Per-run batch log + persistent batch-id counter (from batch_runner.py) ──

def _ensure_batch_log(run_out_dir: str):
    path = f"{run_out_dir}/batch_log.csv"
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=BATCH_LOG_COLUMNS).writeheader()


def _log_batch(run_out_dir: str, row: dict):
    path = f"{run_out_dir}/batch_log.csv"
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=BATCH_LOG_COLUMNS).writerow(row)


def _next_batch_id(run_out_dir: str) -> int:
    """Persistent, restart-safe batch counter, scoped to this run."""
    path = f"{run_out_dir}/next_batch_id.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            n = int((f.read().strip() or "1"))
    else:
        n = 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(n + 1))
    return n


# ─── Fresh start: wipe this model's entire output tree ───────────────────────

if args.fresh:
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
        print(f"🗑️  --fresh: cleared {MODEL_DIR}/ entirely")
    if os.path.exists(EMISSIONS_LOG):
        os.remove(EMISSIONS_LOG)
        print(f"🗑️  --fresh: cleared {EMISSIONS_LOG}")

os.makedirs(MODEL_DIR, exist_ok=True)

is_first_time_for_model = not os.path.exists(EMISSIONS_LOG)
if is_first_time_for_model:
    _init_emissions_log()
    print(f"🌱 Emissions log initialized → {EMISSIONS_LOG}")
else:
    print(f"🌱 Resuming — appending to existing {EMISSIONS_LOG}")

# ─── Load topic mapping ───────────────────────────────────────────────────────

id_to_topic = {}
if os.path.exists(TOPICS_CSV):
    with open(TOPICS_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                id_to_topic[row[0].strip()] = row[1].strip()
    print(f"✅ Loaded {len(id_to_topic)} topic mappings")
else:
    print(f"⚠️  {TOPICS_CSV} not found — topics will be 'unknown'")

all_image_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.endswith((".png", ".jpg", ".jpeg"))
])
total_images = len(all_image_files)
print(f"\n📁 Found {total_images} images to process per run")

# ─── Import stage functions shared across all models ─────────────────────────
# These don't change between models — only the model/tokenizer passed in changes.

from stages.test_generation     import generate_test_cases, save_test_cases, append_to_master_csv
from stages.metamorphic_testing import generate_metamorphic_relations, save_mr_data, append_mr_to_master_csv
from stages.optimization        import optimize_metamorphic_relations, save_optimized_mr_data, append_optimized_mr_to_master
from stages.optimization_reduction import generate_reduced_suite, save_reduced_suite, append_reduced_to_master, append_savings_summary

# ─── Load the selected model ──────────────────────────────────────────────────

vlm_module  = model_cfg["vlm_module"]
llm_module  = model_cfg["llm_module"]

print(f"\n{'='*60}")
print(f"🤖 Starting benchmark: {model_name}  ({NUM_RUNS} runs)")
print(f"   VLM: {vlm_module}  |  LLM: {llm_module}")
print(f"{'='*60}")

# ── Load VLM (Agent 1) ─────────────────────────────────────────────────
load_vlm_fn   = _import(vlm_module, "load_model")
analyze_ui_fn = _import(vlm_module, "analyze_ui")
save_ui_fn    = _import(vlm_module, "save_ui_data")
warmup_vlm_fn = _import(vlm_module, "warmup") if hasattr(
    __import__(vlm_module, fromlist=[""]), "warmup"
) else None

print(f"\nLoading VLM ({model_name})...")
vision_model, processor = load_vlm_fn()

# ── Load LLM (Agents 2/3/4/5) ──────────────────────────────────────────
load_llm_fn   = _import(llm_module, "load_text_model")
warmup_llm_fn = _import(llm_module, "warmup") if hasattr(
    __import__(llm_module, fromlist=[""]), "warmup"
) else None

print(f"\nLoading LLM ({model_name})...")
text_model, text_tokenizer = load_llm_fn()

# ── Warmup pass (NOT timed, NOT tracked — this is setup cost) ──────────
print(f"\n🔥 Running warmup passes for {model_name}...")
if warmup_vlm_fn:
    warmup_vlm_fn(vision_model, processor)
if warmup_llm_fn:
    warmup_llm_fn(text_model, text_tokenizer)
print(f"✅ Warmup done")

# ── Post-warmup stabilization wait (NOT timed, NOT tracked) ─────────────
if WARMUP_STABILIZE_SECONDS > 0:
    print(f"⏳ Stabilizing for {WARMUP_STABILIZE_SECONDS}s after warmup before timed runs begin...")
    time.sleep(WARMUP_STABILIZE_SECONDS)
print(f"✅ Ready to process {NUM_RUNS} run(s)\n")

# ─── Process a single image through all 5 agents for one run ────────────────

def _process_image(run_idx: int, run_out_dir: str, image_file: str) -> bool:
    """Returns True if the screen completed all 5 agents successfully."""
    full_path = os.path.join(IMAGES_DIR, image_file)
    screen_id = os.path.splitext(image_file)[0]
    topic     = id_to_topic.get(screen_id, "unknown")

    print(f"\n  ── {model_name} | run {run_idx} | {image_file} (topic: {topic}) ──")

    try:
        # Agent 1 — Perception
        tracker = EmissionsTracker(
            project_name=f"agent1_{model_name}_{run_idx}_{screen_id}",
            output_dir="outputs", log_level="error", save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        ui_data          = analyze_ui_fn(full_path, vision_model, processor)
        ui_data["topic"] = topic
        save_ui_fn(ui_data, out_dir=f"{run_out_dir}/ui_analysis")
        _log_emissions(run_idx, screen_id, "Agent1_Perception", tracker, time.time() - _t0)
        print(f"  ✅ Agent 1 complete")

        # Agent 2 — Test Case Generation
        tracker = EmissionsTracker(
            project_name=f"agent2_{model_name}_{run_idx}_{screen_id}",
            output_dir="outputs", log_level="error", save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        tc_data              = generate_test_cases(ui_data, text_model, text_tokenizer)
        tc_data["screen_id"] = screen_id
        tc_data["topic"]     = topic
        _log_emissions(run_idx, screen_id, "Agent2_Generation", tracker, time.time() - _t0)
        print(f"  ✅ Agent 2 — {len(tc_data.get('test_cases', []))} test cases")

        save_test_cases(tc_data, out_dir=f"{run_out_dir}/testcases")
        append_to_master_csv(tc_data, master_path=f"{run_out_dir}/testcases_master.csv")

        # Agent 3 — Metamorphic Testing
        tracker = EmissionsTracker(
            project_name=f"agent3_{model_name}_{run_idx}_{screen_id}",
            output_dir="outputs", log_level="error", save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        mr_data = generate_metamorphic_relations(tc_data, text_model, text_tokenizer)
        _log_emissions(run_idx, screen_id, "Agent3_Metamorphic", tracker, time.time() - _t0)
        print(f"  ✅ Agent 3 — {len(mr_data.get('metamorphic_relations', []))} MRs")

        save_mr_data(mr_data, out_dir=f"{run_out_dir}/metamorphic_relations")
        append_mr_to_master_csv(mr_data, master_path=f"{run_out_dir}/metamorphic_relations_master.csv")

        # Agent 4 — Optimization
        tracker = EmissionsTracker(
            project_name=f"agent4_{model_name}_{run_idx}_{screen_id}",
            output_dir="outputs", log_level="error", save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        opt_data = optimize_metamorphic_relations(mr_data, text_model, text_tokenizer)
        _log_emissions(run_idx, screen_id, "Agent4_Optimization", tracker, time.time() - _t0)
        print(f"  ✅ Agent 4 — {len(opt_data.get('optimized_relations', []))} optimized MRs")

        save_optimized_mr_data(opt_data, out_dir=f"{run_out_dir}/optimized_relations")
        append_optimized_mr_to_master(opt_data, master_path=f"{run_out_dir}/optimized_relations_master.csv")

        # Agent 5 — Suite Reduction & Energy Savings
        tracker = EmissionsTracker(
            project_name=f"agent5_{model_name}_{run_idx}_{screen_id}",
            output_dir="outputs", log_level="error", save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        reduced_data = generate_reduced_suite(opt_data, text_model, text_tokenizer)
        _log_emissions(run_idx, screen_id, "Agent5_Reduction", tracker, time.time() - _t0)

        save_reduced_suite(reduced_data, out_dir=f"{run_out_dir}/reduced_suite")
        append_reduced_to_master(reduced_data, master_path=f"{run_out_dir}/reduced_suite_master.csv")
        append_savings_summary(reduced_data, summary_path=f"{run_out_dir}/energy_savings_summary.csv")

        print(f"  ✅ Pipeline complete for {screen_id}")
        return True

    except Exception as e:
        traceback.print_exc()
        print(f"  ❌ Failed on {image_file}: {e} — NOT marked complete, will retry next run of this script")
        return False


# ─── Main loop: runs, each processed in resumable batches ───────────────────

prev_run_did_work = False

for run_idx in range(1, NUM_RUNS + 1):
    run_out_dir = f"{MODEL_DIR}/run_{run_idx}"
    os.makedirs(run_out_dir, exist_ok=True)
    _ensure_batch_log(run_out_dir)

    completed = _load_checkpoint(run_out_dir)
    _rollback_incomplete_screens(run_idx, run_out_dir, completed)
    completed = _load_checkpoint(run_out_dir)  # re-read in case rollback matters downstream

    if total_images > 0 and len(completed) >= total_images:
        print(f"\n✅ Run {run_idx}/{NUM_RUNS}: already fully completed "
              f"({len(completed)}/{total_images}) — skipping.")
        prev_run_did_work = False
        continue

    if run_idx > 1 and prev_run_did_work:
        print(f"\n⏳ Cooldown {COOLDOWN_SECONDS}s before run {run_idx}...")
        time.sleep(COOLDOWN_SECONDS)

    print(f"\n── Run {run_idx}/{NUM_RUNS} | Model: {model_name} "
          f"| {len(completed)}/{total_images} already done ──")

    batches_this_invocation = 0
    stopped_early = False

    while True:
        completed = _load_checkpoint(run_out_dir)
        remaining = [f for f in all_image_files if os.path.splitext(f)[0] not in completed]
        if not remaining:
            break

        if args.max_batches is not None and batches_this_invocation >= args.max_batches:
            print(f"\n⏹  Reached --max-batches={args.max_batches} for run {run_idx}. "
                  f"Stopping. ({len(completed)}/{total_images} done, "
                  f"{len(remaining)} remaining — re-run to continue.)")
            stopped_early = True
            break

        batch_id = _next_batch_id(run_out_dir)
        this_batch_size = len(remaining) if args.batch_size is None else min(args.batch_size, len(remaining))
        batch_images = remaining[:this_batch_size]
        batches_this_invocation += 1
        prev_run_did_work = True

        print(f"\n=== Batch {batch_id} (run {run_idx}) — processing {len(batch_images)} images "
              f"({len(completed)}/{total_images} done so far) ===")

        started = time.time()
        started_str = time.strftime("%Y-%m-%d %H:%M:%S")
        done_before = len(completed)
        batch_failed = False

        for image_file in batch_images:
            screen_id = os.path.splitext(image_file)[0]
            ok = _process_image(run_idx, run_out_dir, image_file)
            if ok:
                _mark_completed(run_out_dir, screen_id)
            else:
                batch_failed = True

        ended = time.time()
        ended_str = time.strftime("%Y-%m-%d %H:%M:%S")
        done_after = len(_load_checkpoint(run_out_dir))

        _log_batch(run_out_dir, {
            "batch_id":                    batch_id,
            "model_name":                  model_name,
            "run_number":                  run_idx,
            "batch_size_requested":        this_batch_size,
            "started_at":                  started_str,
            "ended_at":                    ended_str,
            "duration_sec":                round(ended - started, 1),
            "completed_before":            done_before,
            "completed_after":             done_after,
            "images_processed_this_batch": done_after - done_before,
            "total_images":                total_images,
            "status":                      "partial_failures" if batch_failed else "ok",
        })

        print(f"=== Batch {batch_id} finished: {done_after - done_before} images processed "
              f"in {round(ended - started, 1)}s | status={'partial_failures' if batch_failed else 'ok'} ===")

    if stopped_early:
        print(f"\n⏸ Run {run_idx} left incomplete on purpose (--max-batches). "
              f"Stopping before starting further runs — re-run the same command to continue.")
        break

    print(f"\n✅ Run {run_idx}/{NUM_RUNS} complete — all {total_images} images processed.")

print(f"\n✅ {model_name} benchmark session finished")
print(f"🌱 Emissions log saved → {EMISSIONS_LOG}")
print(f"\nNext: run another model from its own venv with:")
other = [m for m in MODEL_NAMES if m != model_name]
if other:
    print(f"   python benchmark_pipeline.py --model {other[0]}")
print(f"\nThen merge + summarize all with:")
print(f"   python summarize_emissions.py")
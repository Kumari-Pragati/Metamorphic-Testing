# main.py
#
# Runs the full 5-agent pipeline using Qwen across images in images/.
# Supports batching and resume: run this script multiple times with
# --batch-size to process the dataset in chunks across separate sessions.
# Already-completed screens are automatically skipped on the next run.
#
# Usage:
#   python main.py --batch-size 200        (process next 200 unprocessed images)
#   python main.py                          (process ALL remaining unprocessed images)
#   python main.py --fresh                  (wipe all outputs/checkpoint, start over)
#   python main.py --fresh --batch-size 200 (start over, process first 200)

import argparse
import os
import csv
import time
import shutil
import traceback
from codecarbon import EmissionsTracker
from stages.ui_analysis        import load_model, analyze_ui, save_ui_data
from stages.test_generation    import load_text_model, generate_test_cases, save_test_cases, append_to_master_csv
from stages.metamorphic_testing import (
    generate_metamorphic_relations,
    save_mr_data,
    append_mr_to_master_csv,
)
from stages.optimization import (
    optimize_metamorphic_relations,
    save_optimized_mr_data,
    append_optimized_mr_to_master,
)
from stages.optimization_reduction import (
    generate_reduced_suite,
    save_reduced_suite,
    append_reduced_to_master,
    append_savings_summary,
)

IMAGES_DIR = "images/"
TOPICS_CSV = "design_topics.csv"
CHECKPOINT_FILE = "outputs/completed_screens.txt"

# ─── CLI args ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Run the 5-agent pipeline, optionally in batches with resume.")
parser.add_argument(
    "--batch-size",
    type=int,
    default=None,
    help="Number of NEW (not-yet-completed) images to process this run. "
         "Omit to process all remaining images.",
)
parser.add_argument(
    "--fresh",
    action="store_true",
    help="Wipe the checkpoint and all output CSVs/emissions log before starting, "
         "as if running for the first time.",
)
parser.add_argument(
    "--batch-id",
    type=int,
    default=None,
    help="If provided, this run's results are ALSO written to "
         "outputs/batches/batch_<id>/ (per-batch CSVs + a batch_summary.csv), "
         "in addition to appending to the combined master files in outputs/. "
         "Each batch-id is expected to be used exactly once — its files are "
         "freshly created (not appended) at the start of the run.",
)
args = parser.parse_args()

# ─── Emissions log columns ────────────────────────────────────────────────────
EMISSIONS_LOG = "outputs/emissions_log.csv"
EMISSIONS_COLUMNS = [
    "screen_id", "agent", "energy_kwh", "emissions_kg_co2",
    "total_emissions_kg_co2", "duration_seconds",
    "ram_power_w", "cpu_power_w", "gpu_power_w",
    "ram_energy_kwh", "cpu_energy_kwh", "gpu_energy_kwh",
    "region", "country_name", "country_iso_code",
]

MASTER_FILES = [
    "outputs/testcases_master.csv",
    "outputs/metamorphic_relations_master.csv",
    "outputs/optimized_relations_master.csv",
    "outputs/reduced_suite_master.csv",
    "outputs/energy_savings_summary.csv",
]

# ─── Per-batch output setup (only active when --batch-id is passed) ──────────
BATCH_ID  = args.batch_id
BATCH_DIR = f"outputs/batches/batch_{BATCH_ID:03d}" if BATCH_ID is not None else None

if BATCH_DIR:
    os.makedirs(BATCH_DIR, exist_ok=True)

    BATCH_EMISSIONS_LOG = f"{BATCH_DIR}/emissions_log.csv"
    BATCH_SCREENS_FILE  = f"{BATCH_DIR}/screens.txt"
    BATCH_SUMMARY_FILE  = f"{BATCH_DIR}/batch_summary.csv"
    BATCH_MASTER_PATHS = {
        "testcases": f"{BATCH_DIR}/testcases_master.csv",
        "mrs":       f"{BATCH_DIR}/metamorphic_relations_master.csv",
        "optimized": f"{BATCH_DIR}/optimized_relations_master.csv",
        "reduced":   f"{BATCH_DIR}/reduced_suite_master.csv",
        "savings":   f"{BATCH_DIR}/energy_savings_summary.csv",
    }

    # A batch-id is expected to be used exactly once — start its files fresh
    # each run rather than appending, so a re-run with the same id can't
    # silently mix two different batches' data together.
    for _p in [BATCH_EMISSIONS_LOG, BATCH_SCREENS_FILE, BATCH_SUMMARY_FILE, *BATCH_MASTER_PATHS.values()]:
        if os.path.exists(_p):
            os.remove(_p)

    with open(BATCH_EMISSIONS_LOG, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EMISSIONS_COLUMNS).writeheader()

    _batch_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
else:
    BATCH_EMISSIONS_LOG = None
    BATCH_SCREENS_FILE  = None
    BATCH_SUMMARY_FILE  = None
    BATCH_MASTER_PATHS  = {}


def _init_emissions_log():
    os.makedirs("outputs", exist_ok=True)
    if os.path.exists(EMISSIONS_LOG):
        os.remove(EMISSIONS_LOG)
    with open(EMISSIONS_LOG, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EMISSIONS_COLUMNS).writeheader()


def _log_emissions(screen_id: str, agent: str, tracker: EmissionsTracker, duration_s: float):
    emissions_kg = tracker.stop()
    data = tracker.final_emissions_data

    energy_kwh = data.energy_consumed if data else 0.0

    row = {
        "screen_id":              screen_id,
        "agent":                  agent,
        "energy_kwh":             round(energy_kwh, 6),
        "emissions_kg_co2":       round(emissions_kg or 0.0, 8),
        "total_emissions_kg_co2": round(emissions_kg or 0.0, 8),
        "duration_seconds":       round(duration_s, 2),
        "ram_power_w":            round(data.ram_power, 4)        if data else 0.0,
        "cpu_power_w":            round(data.cpu_power, 4)        if data else 0.0,
        "gpu_power_w":            round(data.gpu_power, 4)        if data else 0.0,
        "ram_energy_kwh":         round(data.ram_energy, 6)       if data else 0.0,
        "cpu_energy_kwh":         round(data.cpu_energy, 6)       if data else 0.0,
        "gpu_energy_kwh":         round(data.gpu_energy, 6)       if data else 0.0,
        "region":                 (data.region or "unknown")       if data else "unknown",
        "country_name":           (data.country_name or "unknown") if data else "unknown",
        "country_iso_code":       (data.country_iso_code or "unknown") if data else "unknown",
    }

    with open(EMISSIONS_LOG, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EMISSIONS_COLUMNS).writerow(row)

    if BATCH_EMISSIONS_LOG:
        with open(BATCH_EMISSIONS_LOG, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EMISSIONS_COLUMNS).writerow(row)

    print(f"   🌱 {agent} — {round(emissions_kg or 0.0, 8)} kg CO2 | "
          f"{round(energy_kwh, 6)} kWh | {round(duration_s, 2)}s | "
          f"GPU: {round(data.gpu_power, 2) if data else 0.0}W | "
          f"region: {data.region if data else 'unknown'}")


def _write_batch_summary(screens_processed: list):
    """Write one summary row for this batch by re-reading its own emissions log."""
    if not BATCH_SUMMARY_FILE:
        return

    total_energy = 0.0
    total_emissions = 0.0
    total_duration = 0.0
    if os.path.exists(BATCH_EMISSIONS_LOG):
        with open(BATCH_EMISSIONS_LOG, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                total_energy    += float(r.get("energy_kwh") or 0.0)
                total_emissions += float(r.get("emissions_kg_co2") or 0.0)
                total_duration  += float(r.get("duration_seconds") or 0.0)

    with open(BATCH_SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "batch_id", "screens_processed", "total_energy_kwh",
            "total_emissions_kg_co2", "total_duration_seconds",
            "started_at", "ended_at",
        ])
        w.writeheader()
        w.writerow({
            "batch_id":               BATCH_ID,
            "screens_processed":      len(screens_processed),
            "total_energy_kwh":       round(total_energy, 6),
            "total_emissions_kg_co2": round(total_emissions, 8),
            "total_duration_seconds": round(total_duration, 2),
            "started_at":             _batch_started_at,
            "ended_at":               time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    print(f"🌱 Batch {BATCH_ID} summary saved → {BATCH_SUMMARY_FILE}")


def _load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def _mark_completed(screen_id: str):
    os.makedirs("outputs", exist_ok=True)
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(screen_id + "\n")


def _rollback_incomplete_screens(completed: set):
    """
    Remove any rows left behind by a screen that started but never reached
    _mark_completed() (e.g. Ctrl+C mid-pipeline). Safe to run every startup:
    any screen_id found in these files that is NOT in the checkpoint is by
    definition partial, since completion is only recorded after all 5 agents
    succeed. This also backs out that screen's emissions/energy rows so
    reprocessing it doesn't double-count energy or CO2.
    """
    files_and_key = [
        (EMISSIONS_LOG, "screen_id"),
        ("outputs/testcases_master.csv", "screen_id"),
        ("outputs/metamorphic_relations_master.csv", "screen_id"),
        ("outputs/optimized_relations_master.csv", "screen_id"),
        ("outputs/reduced_suite_master.csv", "screen_id"),
        ("outputs/energy_savings_summary.csv", "screen_id"),
    ]

    for path, key in files_and_key:
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if not fieldnames or key not in fieldnames:
            continue
        kept = [r for r in rows if r.get(key) in completed]
        dropped = len(rows) - len(kept)
        if dropped:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(kept)
            print(f"🧹 Rolled back {dropped} partial row(s) from {path} "
                  f"(incomplete screen from a previous interrupted run)")


# ─── Fresh start: wipe everything ─────────────────────────────────────────────

if args.fresh:
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
        print("🗑️  --fresh: cleared outputs/ entirely")
    os.makedirs("outputs", exist_ok=True)

completed_screens = _load_checkpoint()
_rollback_incomplete_screens(completed_screens)
is_first_run = not os.path.exists(EMISSIONS_LOG)

if is_first_run:
    for master_path in MASTER_FILES:
        if os.path.exists(master_path):
            os.remove(master_path)
    _init_emissions_log()
    print("🌱 Emissions log initialized (first run)")
else:
    print(f"🌱 Resuming — {len(completed_screens)} screens already completed, appending to existing logs")

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

# ─── Load vision model once (Agent 1) — NOT tracked, this is setup cost ──────
print("\nLoading vision model (Agent 1)...")
vision_model, processor = load_model()

# ─── Load text model once (Agents 2, 3, 4, 5 share it) — NOT tracked ────────
print("\nLoading text model (Agents 2 / 3 / 4 / 5)...")
text_model, text_tokenizer = load_text_model()

# ─── Determine which images still need processing ────────────────────────────
all_image_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.endswith((".png", ".jpg", ".jpeg"))
])

remaining = [
    f for f in all_image_files
    if os.path.splitext(f)[0] not in completed_screens
]

print(f"\n📁 {len(all_image_files)} total images | {len(completed_screens)} already done | {len(remaining)} remaining")

if args.batch_size is not None:
    image_files = remaining[:args.batch_size]
    print(f"📦 Batch mode: processing {len(image_files)} of {len(remaining)} remaining images this run")
else:
    image_files = remaining
    print(f"📦 Processing all {len(image_files)} remaining images this run")

if not image_files:
    print("\n✅ Nothing left to process — all images already completed.")
    raise SystemExit(0)

# ─── Process images ────────────────────────────────────────────────────────────

batch_screens_done = []

for idx, image_file in enumerate(image_files, start=1):
    full_path = os.path.join(IMAGES_DIR, image_file)
    screen_id = os.path.splitext(image_file)[0]
    topic     = id_to_topic.get(screen_id, "unknown")

    print(f"\n── [{idx}/{len(image_files)}] {image_file} (topic: {topic}) ──")

    try:
        # ── Agent 1 — Perception ──────────────────────────────────────────
        tracker = EmissionsTracker(
            project_name=f"agent1_{screen_id}",
            output_dir="outputs",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        ui_data          = analyze_ui(full_path, vision_model, processor)
        ui_data["topic"] = topic
        json_path, txt_path = save_ui_data(ui_data)
        _log_emissions(screen_id, "Agent1_Perception", tracker, time.time() - _t0)
        print(f"✅ Agent 1 complete — UI data saved")

        # ── Agent 2 — Test Case Generation ────────────────────────────────
        tracker = EmissionsTracker(
            project_name=f"agent2_{screen_id}",
            output_dir="outputs",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        tc_data              = generate_test_cases(ui_data, text_model, text_tokenizer)
        tc_data["screen_id"] = screen_id
        tc_data["topic"]     = topic
        _log_emissions(screen_id, "Agent2_Generation", tracker, time.time() - _t0)
        print(f"✅ Agent 2 (Generation) — {len(tc_data.get('test_cases', []))} test cases")

        save_test_cases(tc_data)
        append_to_master_csv(tc_data)
        if BATCH_DIR:
            append_to_master_csv(tc_data, master_path=BATCH_MASTER_PATHS["testcases"])

        # ── Agent 3 — Metamorphic Testing ─────────────────────────────────
        tracker = EmissionsTracker(
            project_name=f"agent3_{screen_id}",
            output_dir="outputs",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        mr_data = generate_metamorphic_relations(tc_data, text_model, text_tokenizer)
        _log_emissions(screen_id, "Agent3_Metamorphic", tracker, time.time() - _t0)
        print(f"✅ Agent 3 (Metamorphic) — {len(mr_data.get('metamorphic_relations', []))} MRs")

        save_mr_data(mr_data)
        append_mr_to_master_csv(mr_data)
        if BATCH_DIR:
            append_mr_to_master_csv(mr_data, master_path=BATCH_MASTER_PATHS["mrs"])

        # ── Agent 4 — Optimization ─────────────────────────────────────────
        tracker = EmissionsTracker(
            project_name=f"agent4_{screen_id}",
            output_dir="outputs",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        opt_data = optimize_metamorphic_relations(mr_data, text_model, text_tokenizer)
        _log_emissions(screen_id, "Agent4_Optimization", tracker, time.time() - _t0)
        print(f"✅ Agent 4 (Optimization) — {len(opt_data.get('optimized_relations', []))} optimized MRs")

        save_optimized_mr_data(opt_data)
        append_optimized_mr_to_master(opt_data)
        if BATCH_DIR:
            append_optimized_mr_to_master(opt_data, master_path=BATCH_MASTER_PATHS["optimized"])

        # ── Agent 5 — Suite Reduction & Energy Savings ─────────────────────
        tracker = EmissionsTracker(
            project_name=f"agent5_{screen_id}",
            output_dir="outputs",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        reduced_data = generate_reduced_suite(opt_data, text_model, text_tokenizer)
        _log_emissions(screen_id, "Agent5_Reduction", tracker, time.time() - _t0)

        save_reduced_suite(reduced_data)
        append_reduced_to_master(reduced_data)
        append_savings_summary(reduced_data)
        if BATCH_DIR:
            append_reduced_to_master(reduced_data, master_path=BATCH_MASTER_PATHS["reduced"])
            append_savings_summary(reduced_data, summary_path=BATCH_MASTER_PATHS["savings"])

        # ── Mark this screen done — only after ALL agents succeeded ────────
        _mark_completed(screen_id)
        if BATCH_DIR:
            batch_screens_done.append(screen_id)
            with open(BATCH_SCREENS_FILE, "a", encoding="utf-8") as f:
                f.write(screen_id + "\n")
        print(f"✅ Pipeline complete for {screen_id} ({idx}/{len(image_files)} this run)")

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Failed on {image_file}: {e} — NOT marked complete, will retry next run")
        continue

remaining_after = len(remaining) - len(image_files)
print(f"\n✅ Batch finished — {len(image_files)} images processed this run")
if remaining_after > 0:
    print(f"📦 {remaining_after} images still remaining — run again with the same command to continue")
else:
    print("🎉 All images in the dataset have now been processed")
print(f"🌱 Emissions log saved → {EMISSIONS_LOG}")

if BATCH_DIR:
    _write_batch_summary(batch_screens_done)
    print(f"📦 Batch {BATCH_ID} outputs saved → {BATCH_DIR}/")
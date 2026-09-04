# batch_runner.py
#
# Runs main.py repeatedly in fixed-size batches until every image in
# images/ has been processed, logging each batch's progress to
# outputs/batch_log.csv. Never passes --fresh — safe to stop (Ctrl+C)
# and re-run at any time; it just resumes where the checkpoint left off.
#
# Each batch is also given a persistent --batch-id, read from and saved
# back to outputs/next_batch_id.txt. This is what main.py uses to name
# its outputs/batches/batch_<id>/ folder. Using a file (rather than a
# local loop counter) means batch numbering survives Ctrl+C / restarts:
# if you've already completed batches 1-3 and restart this script, the
# next batch is correctly labeled batch 4, not batch 1 again — so it
# can never overwrite an earlier batch's per-batch output folder.
#
# Usage:
#   python batch_runner.py                    (default: 100 per batch)
#   python batch_runner.py --batch-size 50     (50 per batch, ~29 batches for 1460)
#   python batch_runner.py --batch-size 100 --max-batches 3   (stop after 3 batches, e.g. for a quick test)
import argparse
import csv
import os
import subprocess
import sys
import time

IMAGES_DIR = "images/"
CHECKPOINT_FILE = "outputs/completed_screens.txt"
BATCH_LOG = "outputs/batch_log.csv"
NEXT_BATCH_ID_FILE = "outputs/next_batch_id.txt"

parser = argparse.ArgumentParser(description="Run main.py in tracked batches until the dataset is fully processed.")
parser.add_argument("--batch-size", type=int, default=100, help="Images processed per batch (default: 100).")
parser.add_argument("--max-batches", type=int, default=None, help="Optional cap on number of batches this run (for testing).")
args = parser.parse_args()


def total_images() -> int:
    if not os.path.isdir(IMAGES_DIR):
        print(f"ERROR: {IMAGES_DIR} not found. Run this from the project root.")
        sys.exit(1)
    return len([
        f for f in os.listdir(IMAGES_DIR)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])


def completed_count() -> int:
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return len([line for line in f if line.strip()])


def ensure_log_header():
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists(BATCH_LOG):
        with open(BATCH_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "batch_id", "batch_size_requested", "started_at", "ended_at",
                "duration_sec", "completed_before", "completed_after",
                "images_processed_this_batch", "total_images", "status",
            ])


def log_batch(row):
    with open(BATCH_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def next_batch_id() -> int:
    """Persistent, restart-safe batch counter (see module docstring)."""
    os.makedirs("outputs", exist_ok=True)
    if os.path.exists(NEXT_BATCH_ID_FILE):
        with open(NEXT_BATCH_ID_FILE, "r", encoding="utf-8") as f:
            n = int((f.read().strip() or "1"))
    else:
        n = 1
    with open(NEXT_BATCH_ID_FILE, "w", encoding="utf-8") as f:
        f.write(str(n + 1))
    return n


def main():
    total = total_images()
    ensure_log_header()
    runs_this_session = 0
    while True:
        done = completed_count()
        remaining = total - done
        if remaining <= 0:
            print(f"\nAll {total} images completed. Nothing left to do.")
            break
        if args.max_batches is not None and runs_this_session >= args.max_batches:
            print(f"\nReached --max-batches={args.max_batches}. Stopping. "
                  f"({done}/{total} done, {remaining} remaining — re-run to continue.)")
            break

        batch_id = next_batch_id()
        runs_this_session += 1
        this_batch_size = min(args.batch_size, remaining)
        print(f"\n=== Batch {batch_id} — requesting {this_batch_size} images "
              f"({done}/{total} done so far) ===")

        started = time.time()
        started_str = time.strftime("%Y-%m-%d %H:%M:%S")
        result = subprocess.run(
            [sys.executable, "main.py",
             "--batch-size", str(this_batch_size),
             "--batch-id", str(batch_id)],
        )
        ended = time.time()
        ended_str = time.strftime("%Y-%m-%d %H:%M:%S")

        done_after = completed_count()
        status = "ok" if result.returncode == 0 else f"exit_code_{result.returncode}"
        log_batch([
            batch_id, this_batch_size, started_str, ended_str,
            round(ended - started, 1), done, done_after,
            done_after - done, total, status,
        ])
        print(f"=== Batch {batch_id} finished: {done_after - done} images processed "
              f"in {round(ended - started, 1)}s | status={status} "
              f"| per-batch files: outputs/batches/batch_{batch_id:03d}/ ===")

        if result.returncode != 0:
            print("\nmain.py exited with an error — stopping the batch runner. "
                  "Fix the issue and re-run this script; it will resume from the checkpoint. "
                  "Note: the batch-id already consumed for this failed attempt will not be reused; "
                  "the next run gets a fresh batch-id, which is correct since the failed batch's "
                  "partial per-batch files should be inspected, not silently continued into.")
            sys.exit(1)


if __name__ == "__main__":
    main()
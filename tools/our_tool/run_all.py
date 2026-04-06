#!/usr/bin/env python3
"""
run_all.py — Run the enhanced resolver on every snippet in /gists.

Usage (inside the container):
    python run_all.py -g /gists -m gemma2 -b http://ollama-our-tool:11434 -l 5 -r 0
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import time


def find_snippets(gists_dir: str) -> list:
    snippets = []
    for root, dirs, files in os.walk(gists_dir):
        for f in files:
            if f == "snippet.py":
                snippets.append(os.path.join(root, f))
    return sorted(snippets)


def already_solved(snippet_path: str) -> bool:
    """
    Check if this snippet already has a passing output.
    Reads output_data_X.X.yml and looks for error_type: None.
    Uses regex to handle spacing variations.
    """
    folder = os.path.dirname(snippet_path)
    yml_files = glob.glob(f"{folder}/output_data_*.yml")
    for yml in yml_files:
        try:
            text = open(yml).read()
            if re.search(r'error_type:\s*None', text):
                return True
        except Exception:
            pass
    return False


def run_snippet(snippet_path: str, args) -> dict:
    start = time.time()
    cmd = [
        "python", "/app/test_executor.py",
        "-f", snippet_path,
        "-m", args.model,
        "-b", args.base,
        "-l", str(args.loop),
        "-r", str(args.range),
        "-t", str(args.temp),
    ]

    folder     = os.path.dirname(snippet_path)
    snippet_id = folder.split("/")[-1]

    print(f"\n{'='*60}")
    print(f"[{snippet_id}] Starting...")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            timeout=args.timeout,
            capture_output=False,
        )
        # Wait for YML to finish flushing to disk before checking
        time.sleep(2)
        elapsed = time.time() - start
        solved  = already_solved(snippet_path)

        return {
            "snippet_id":      snippet_id,
            "snippet_path":    snippet_path,
            "return_code":     result.returncode,
            "solved":          solved,
            "elapsed_seconds": round(elapsed, 1),
            "status":          "solved" if solved else "failed",
            "error":           "",
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"[{snippet_id}] TIMEOUT after {args.timeout}s")
        return {
            "snippet_id":      snippet_id,
            "snippet_path":    snippet_path,
            "return_code":     -1,
            "solved":          False,
            "elapsed_seconds": round(elapsed, 1),
            "status":          "timeout",
            "error":           f"Exceeded {args.timeout}s timeout",
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"[{snippet_id}] EXCEPTION: {e}")
        return {
            "snippet_id":      snippet_id,
            "snippet_path":    snippet_path,
            "return_code":     -2,
            "solved":          False,
            "elapsed_seconds": round(elapsed, 1),
            "status":          "exception",
            "error":           str(e),
        }


def write_summary(results: list, output_path: str):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Summary] Written to {output_path}")


def print_stats(results: list):
    total      = len(results)
    solved     = sum(1 for r in results if r["solved"])
    failed     = sum(1 for r in results if r["status"] == "failed")
    timeout    = sum(1 for r in results if r["status"] == "timeout")
    skipped    = sum(1 for r in results if r["status"] == "skipped")
    total_time = sum(r["elapsed_seconds"] for r in results)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"  Total snippets : {total}")
    print(f"  Solved         : {solved}  ({100*solved/max(total,1):.1f}%)")
    print(f"  Failed         : {failed}")
    print(f"  Timeout        : {timeout}")
    print(f"  Skipped (done) : {skipped}")
    print(f"  Total time     : {total_time/60:.1f} min")
    print("="*60)


def process_args():
    parser = argparse.ArgumentParser(description="Run enhanced resolver on all snippets")
    parser.add_argument("-g", "--gists",    type=str,   default="/gists")
    parser.add_argument("-b", "--base",     type=str,   default="http://ollama-our-tool:11434")
    parser.add_argument("-m", "--model",    type=str,   default="gemma2")
    parser.add_argument("-t", "--temp",     type=float, default=0.7)
    parser.add_argument("-l", "--loop",     type=int,   default=5)
    parser.add_argument("-r", "--range",    type=int,   default=0)
    parser.add_argument("--timeout",        type=int,   default=1200)
    parser.add_argument("--skip-solved",    action="store_true", default=True)
    parser.add_argument("--no-skip",        action="store_true")
    parser.add_argument("--limit",          type=int,   default=0)
    parser.add_argument("--start-from",     type=str,   default="")
    return parser.parse_args()


def main():
    args = process_args()

    print(f"[Batch] Scanning {args.gists} for snippets...")
    snippets = find_snippets(args.gists)
    print(f"[Batch] Found {len(snippets)} snippets")

    if args.limit > 0:
        snippets = snippets[:args.limit]
        print(f"[Batch] Limited to first {args.limit} snippets")

    if args.start_from:
        for i, s in enumerate(snippets):
            if args.start_from in s:
                snippets = snippets[i:]
                print(f"[Batch] Resuming from {args.start_from} ({len(snippets)} remaining)")
                break

    results      = []
    summary_path = os.path.join(args.gists, "results_summary.csv")
    skip_solved  = args.skip_solved and not args.no_skip

    for i, snippet_path in enumerate(snippets):
        folder     = os.path.dirname(snippet_path)
        snippet_id = folder.split("/")[-1]
        progress   = f"[{i+1}/{len(snippets)}]"

        if skip_solved and already_solved(snippet_path):
            print(f"{progress} Skipping {snippet_id} (already solved)")
            results.append({
                "snippet_id":      snippet_id,
                "snippet_path":    snippet_path,
                "return_code":     0,
                "solved":          True,
                "elapsed_seconds": 0,
                "status":          "skipped",
                "error":           "",
            })
            continue

        print(f"\n{progress} Processing {snippet_id}")
        result = run_snippet(snippet_path, args)
        results.append(result)

        # Write rolling summary after every snippet — never lose data
        write_summary(results, summary_path)

        # Print rolling stats every 10 snippets
        if (i + 1) % 10 == 0:
            solved_so_far = sum(1 for r in results if r["solved"])
            print(f"\n[Rolling] {solved_so_far}/{i+1} solved ({100*solved_so_far/max(i+1,1):.1f}%)")

    print_stats(results)
    write_summary(results, summary_path)


if __name__ == "__main__":
    main()


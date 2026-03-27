"""
run.py
-------
Entry Point — CLI equivalent of PLLM's test_executor.py

Usage (inside the container):
  python run.py -f /gists/0a2ac74d800a2eff9540/snippet.py \
                -m gemma2 \
                -b http://ollama:11434 \
                -l 5 \
                -p 3.8

Results are appended to a single CSV file (default: /gists/results.csv)
rather than writing a separate JSON file per snippet — making it easy to
aggregate results across the whole dataset.

CSV columns match PLLM's YAML output structure so comparison is straightforward:
  snippet_path, python_version, status, iterations, elapsed_seconds,
  used_knowledge_graph, requirements, final_error_type
"""

import argparse
import csv
import json
import os
import sys
import time

from agent import run_agent


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "snippet_path",
    "python_version",
    "status",
    "iterations",
    "elapsed_seconds",
    "used_knowledge_graph",
    "requirements",        # JSON-encoded dict, e.g. '{"numpy": "1.21.0"}'
    "final_error_type",    # last error seen before success/failure
]


def append_to_csv(result: dict, csv_path: str) -> None:
    """
    Append one result row to the shared CSV file.
    Creates the file with a header row if it doesn't exist yet.
    Uses append mode so parallel/sequential runs accumulate cleanly.
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "snippet_path":         result["snippet_path"],
            "python_version":       result["python_version"],
            "status":               result["status"],
            "iterations":           result["iterations"],
            "elapsed_seconds":      result["elapsed_seconds"],
            "used_knowledge_graph": result["used_knowledge_graph"],
            "requirements":         json.dumps(result.get("requirements", {})),
            "final_error_type":     result.get("final_error_type", ""),
        })


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PLLM++ — Agentic Python Dependency Resolver"
    )
    parser.add_argument(
        "-f", "--file",
        required=True,
        help="Full path to the Python snippet, e.g. /gists/.../snippet.py"
    )
    parser.add_argument(
        "-m", "--model",
        default="gemma2",
        help="Ollama model name (default: gemma2)"
    )
    parser.add_argument(
        "-b", "--base",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "-l", "--loop",
        type=int,
        default=5,
        help="Maximum LLM retry iterations (default: 5)"
    )
    parser.add_argument(
        "-p", "--python",
        default="3.8",
        help="Initial Python version guess (default: 3.8)"
    )
    parser.add_argument(
        "-o", "--output",
        default="/gists/results.csv",
        help="Path to the shared CSV results file (default: /gists/results.csv)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    start_time = time.time()

    print(f"[run.py] Starting PLLM++ agent on: {args.file}")

    final_state = run_agent(
        snippet_path    = args.file,
        python_version  = args.python,
        model           = args.model,
        ollama_base_url = args.base,
        max_iterations  = args.loop,
    )

    elapsed = round(time.time() - start_time, 2)

    result = {
        "snippet_path":         args.file,
        "python_version":       final_state["python_version"],
        "status":               final_state["status"],
        "iterations":           final_state["iteration"],
        "elapsed_seconds":      elapsed,
        "used_knowledge_graph": final_state.get("used_kg", False),
        "requirements":         final_state.get("requirements", {}),
        # Pull the last error type out of error_info if present
        "final_error_type":     final_state.get("error_info", {}).get("error_type", ""),
    }

    # ── Print summary to stdout ──────────────────────────────────────────
    print(json.dumps(result, indent=2))

    # ── Append to shared CSV (no per-snippet JSON files) ────────────────
    append_to_csv(result, args.output)
    print(f"[run.py] Result appended to {args.output}")

    sys.exit(0 if final_state["status"] == "success" else 1)


if __name__ == "__main__":
    main()

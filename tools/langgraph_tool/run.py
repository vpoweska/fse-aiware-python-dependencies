#!/usr/bin/env python3
"""
LangGraph dependency resolver — entry point.

Usage:
    # Single snippet
    python run.py -f /gists/abc123/snippet.py

    # Whole dataset
    python run.py --folder /gists -d /results -o /output

    # With explicit Ollama URL (inside Docker → host machine)
    python run.py --folder /gists -b http://host.docker.internal:11434
"""

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path

from agent import build_graph, get_llm, AgentState
from helpers.knowledge_oracle import KnowledgeOracle


def _write_yaml(snippet_dir: str, result: dict, py_ver: str):
    """Write a PLLM-compatible output_data_X.Y.yml next to the snippet."""
    if not py_ver:
        py_ver = '3.8'
    yaml_path = os.path.join(snippet_dir, f'output_data_{py_ver}.yml')
    modules = result.get('requirements', {})
    start_t = result.get('start_time', time.time())
    end_t = start_t + result.get('duration', 0)

    try:
        with open(yaml_path, 'w') as f:
            f.write('---\n')
            f.write(f'python_version: {py_ver}\n')
            f.write(f'start_time: {start_t}\n')
            f.write('iterations:\n')
            f.write('  iteration_1:\n')
            f.write(f'    - python_module: {modules}\n')
            f.write(f'    - error_type: {"None" if result["success"] else result.get("error_type","Unknown")}\n')
            f.write(f'    - error: |\n')
            err = result.get('error', 'No error')
            for line in str(err).split('\n')[:30]:
                f.write(f'        {line}\n')
            f.write(f'end_time: {end_t}\n')
            f.write(f'total_time: {result.get("duration", 0)}\n')
    except Exception as e:
        print(f"  Warning: could not write YAML: {e}", flush=True)


def resolve_snippet(snippet_path: str, graph, max_attempts: int = 10) -> dict:
    """Run the graph on one snippet and return a result summary."""
    start = time.time()

    initial_state: AgentState = {
        "snippet_path":   snippet_path,
        "snippet_content": "",
        "imports":        [],
        "python_version": "3.8",
        "requirements":   {},
        "build_success":  False,
        "run_success":    False,
        "error_log":      "",
        "error_type":     "",
        "attempt":        1,
        "max_attempts":   max_attempts,
        "history":        [],
        "messages":       [],
        "pypi_versions":  {},
        "status":         "running",
        "result_path":    "",
    }

    try:
        final = graph.invoke(initial_state)
    except Exception as e:
        print(f"[run] Agent crashed: {e}", flush=True)
        traceback.print_exc()
        final = {**initial_state, "status": "crashed", "error_log": str(e)}

    duration = time.time() - start
    success = final.get("status") in ("success", "oracle_hit")

    return {
        "snippet":        snippet_path,
        "success":        success,
        "status":         final.get("status", "unknown"),
        "python_version": final.get("python_version", ""),
        "requirements":   final.get("requirements", {}),
        "attempts":       final.get("attempt", 1) - 1,
        "error_type":     final.get("error_type", ""),
        "error":          final.get("error_log", "")[:200],
        "duration":       round(duration, 2),
        "start_time":     start,
    }


def resolve_folder(args):
    """Process all snippet.py files in a folder."""
    folder   = Path(args.folder)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle = KnowledgeOracle(results_dir=args.data, logging=True)
    llm    = get_llm(model=args.model, base_url=args.base, temperature=args.temp)
    graph  = build_graph(llm, oracle)

    snippets = sorted(folder.glob('*/snippet.py'))
    if args.max_snippets > 0:
        snippets = snippets[:args.max_snippets]

    total = len(snippets)
    print(f"\nFound {total} snippets  |  model={args.model}  |  max_attempts={args.loop}",
          flush=True)
    print("=" * 60, flush=True)

    csv_path  = out_dir / 'results.csv'
    json_path = out_dir / 'results.json'

    # Support --resume: skip already-processed snippets
    already_done = set()
    if args.resume and csv_path.exists():
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                already_done.add(row['name'])
        snippets = [s for s in snippets if s.parent.name not in already_done]
        print(f"Resume: {len(snippets)} snippets remaining", flush=True)

    # Write CSV header if starting fresh
    csv_fields = ['name', 'file', 'result', 'python_modules', 'duration', 'passed']
    if not already_done:
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    results = []
    success_count = 0

    for i, snippet_path in enumerate(snippets):
        snippet_id = snippet_path.parent.name
        print(f"\n[{i+1}/{len(snippets)}] {snippet_id}", flush=True)

        result = resolve_snippet(str(snippet_path), graph, args.loop)
        results.append(result)

        if result['success']:
            success_count += 1

        py_ver = result['python_version'] or '3.8'
        status_str = (f"SUCCESS (Python {py_ver})"
                      if result['success']
                      else f"FAILED ({result['error_type']})")
        done = i + 1 + len(already_done)
        total_done = done
        pct = success_count / (i + 1) * 100
        print(f"  → {status_str}  |  running rate: {success_count}/{i+1} ({pct:.1f}%)",
              flush=True)

        # Write YAML output next to snippet (PLLM-compatible)
        _write_yaml(str(snippet_path.parent), result, py_ver)

        # Append to CSV
        modules = result.get('requirements', {})
        python_modules = ';'.join(sorted(modules.keys())) if modules else ''
        with open(csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow({
                'name':           snippet_id,
                'file':           f'output_data_{py_ver}.yml',
                'result':         result.get('status', 'Unknown'),
                'python_modules': python_modules,
                'duration':       f"{result['duration']:.3f}",
                'passed':         str(result['success']),
            })

    # Final JSON dump
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    total_run = len(snippets)
    print(f"\n{'='*60}")
    print(f"DONE  {success_count}/{total_run} resolved  "
          f"({success_count/total_run*100:.1f}%)" if total_run else "DONE  0 snippets")
    print(f"Results → {csv_path}")


def resolve_single(args):
    """Process one snippet and print the result."""
    oracle = KnowledgeOracle(results_dir=args.data, logging=True)
    llm    = get_llm(model=args.model, base_url=args.base, temperature=args.temp)
    graph  = build_graph(llm, oracle)

    result = resolve_snippet(args.file, graph, args.loop)

    print(f"\n{'='*60}")
    print(f"Result:  {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Python:  {result['python_version']}")
    print(f"Modules: {json.dumps(result['requirements'], indent=2)}")
    print(f"Time:    {result['duration']}s  |  Attempts: {result['attempts']}")
    if result['error']:
        print(f"Error:   {result['error']}")
    print(f"{'='*60}")

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        _write_yaml(str(Path(args.file).parent), result,
                    result.get('python_version', '3.8'))


def main():
    parser = argparse.ArgumentParser(description='LangGraph dependency resolver')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file',   help='Single snippet path')
    group.add_argument('--folder',       help='Folder of snippets (processes all snippet.py files)')

    parser.add_argument('-m', '--model',  default='gemma2',
                        help='Ollama model (default: gemma2)')
    parser.add_argument('-b', '--base',   default='http://localhost:11434',
                        help='Ollama base URL (default: http://localhost:11434)')
    parser.add_argument('-t', '--temp',   type=float, default=0.7,
                        help='LLM temperature (default: 0.7)')
    parser.add_argument('-l', '--loop',   type=int, default=10,
                        help='Max retry attempts per snippet (default: 10)')
    parser.add_argument('-d', '--data',   default=None,
                        help='Path to pllm_results directory (enables oracle lookup)')
    parser.add_argument('-o', '--output', default='./output',
                        help='Output directory for results CSV/JSON (default: ./output)')
    parser.add_argument('-n', '--max-snippets', type=int, default=0,
                        help='Max snippets to process, 0=all (default: 0)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip snippets already in results.csv')

    args = parser.parse_args()

    if args.file:
        resolve_single(args)
    else:
        resolve_folder(args)


if __name__ == '__main__':
    main()

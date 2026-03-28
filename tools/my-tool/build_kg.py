"""
build_kg.py

One-time Knowledge Graph Builder

Run this ONCE before starting your test run to populate the SQLite knowledge
graph from the historical training data (pllm_results/, pyego-results/, readpy-results/.).

The DB is written to /gists/knowledge_graph.db, inside the mounted volume,
so it persists across container restarts and never needs to be rebuilt unless
you want to refresh it with new training data.

Usage (inside the container):
  python build_kg.py

Optional arguments:
  python build_kg.py --db /gists/knowledge_graph.db
  python build_kg.py --results /gists/pllm_results /gists/pyego-results
  python build_kg.py --stats       # just print DB stats, don't rebuild
"""

import argparse
import glob
import os
import sqlite3
import sys
import tarfile

sys.path.insert(0, "/app")

from helpers.knowledge_graph import (
    build_db,
    init_db,
    DEFAULT_DB_PATH,
    RESULT_DIRS,
)

def extract_archives(search_dir: str = "/gists") -> None:
    """
    Find any .tar.gz archives in `search_dir` and extract them in place,
    but only if the extracted folder doesn't already exist.
    """
    archives = glob.glob(os.path.join(search_dir, "*.tar.gz"))

    if not archives:
        print(f"[build_kg] No .tar.gz archives found in {search_dir}")
        return

    for archive_path in sorted(archives):
        folder_name = os.path.basename(archive_path).replace(".tar.gz", "")
        extract_to  = os.path.join(search_dir, folder_name)

        if os.path.isdir(extract_to):
            print(f"[build_kg] Already extracted: {folder_name}/ — skipping")
            continue

        print(f"[build_kg] Extracting {os.path.basename(archive_path)} …")
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=search_dir)
            print(f"[build_kg] Extracted → {extract_to}")
        except Exception as e:
            print(f"[build_kg] Failed to extract {archive_path}: {e}")


def print_stats(db_path: str) -> None:
    if not os.path.isfile(db_path):
        print(f"[Stats] No database found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) as n FROM working_packages").fetchone()["n"]
    packages = conn.execute("SELECT COUNT(DISTINCT package) as n FROM working_packages").fetchone()["n"]
    py_versions = conn.execute("SELECT COUNT(DISTINCT python_version) as n FROM working_packages").fetchone()["n"]

    print(f"\n[Stats] Knowledge graph at: {db_path}")
    print(f"  Total records   : {total:,}")
    print(f"  Unique packages : {packages:,}")
    print(f"  Python versions : {py_versions}")

    print("\n  Top 10 most-seen packages:")
    rows = conn.execute(
        "SELECT package, COUNT(*) as n FROM working_packages GROUP BY package ORDER BY n DESC LIMIT 10"
    ).fetchall()
    for row in rows:
        print(f"    {row['package']:<30} {row['n']:>5} records")

    print("\n  Records per Python version:")
    rows = conn.execute(
        "SELECT python_version, COUNT(*) as n FROM working_packages GROUP BY python_version ORDER BY python_version"
    ).fetchall()
    for row in rows:
        print(f"    Python {row['python_version']:<8} {row['n']:>5} records")

    conn.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the PLLM++ knowledge graph from historical training data"
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to write the SQLite DB (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--results",
        nargs="+",
        default=RESULT_DIRS,
        help="One or more directories containing YAML result files"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print DB statistics only — do not rebuild"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing DB and rebuild from scratch"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.stats:
        print_stats(args.db)
        return

    if args.force and os.path.isfile(args.db):
        print(f"[build_kg] --force: deleting existing DB at {args.db}")
        os.remove(args.db)

    if os.path.isfile(args.db):
        print(f"[build_kg] Database already exists at {args.db}")
        print("[build_kg] Nothing to do. Use --force to rebuild, or --stats to inspect.")
        print_stats(args.db)
        return

    gists_dir = os.path.dirname(args.db)   # e.g. /gists
    extract_archives(search_dir=gists_dir)
    print(f"\n[build_kg] Building knowledge graph …")
    print(f"  DB path     : {args.db}")
    print(f"  Result dirs : {args.results}\n")

    inserted = build_db(result_dirs=args.results, db_path=args.db)

    print(f"\n[build_kg] Done — {inserted:,} records inserted.")
    print_stats(args.db)


if __name__ == "__main__":
    main()

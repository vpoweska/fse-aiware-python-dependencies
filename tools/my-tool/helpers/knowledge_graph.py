"""
helpers/knowledge_graph.py
----------------------------
Knowledge Graph — Improvement #1

Goal: Before we ever call the LLM, check whether we already KNOW a
      working combination of (package, version, python_version) from
      historical experiment data (pllm_results/, pyego-results/, etc.).

How it works:
  1. On first run, `build_db()` reads YAML result files from the training
     data folders and populates a local SQLite database.
  2. `query_working_versions(packages, python_version)` looks up the DB
     and returns versions that worked before for similar setups.
  3. The agent calls this before hitting the LLM, saving time + API cost.

SQLite is chosen deliberately:
  - No extra dependencies (stdlib only).
  - Fast for the small dataset we have (~2,900 snippets).
  - Easy to inspect with any SQLite browser.
"""

import sqlite3
import os
import glob
import yaml   # pyyaml is already in the Dockerfile
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DB lives on the mounted /gists volume so it survives container restarts.
# Build it once with build_kg.py; every subsequent run just reads it.
DEFAULT_DB_PATH = "/gists/knowledge_graph.db"

# Folders that hold historical results — all under the mounted /gists volume.
RESULT_DIRS = [
    "/gists/pllm_results",
    "/gists/pyego-results",
    "/gists/readpy-results",
]


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def _get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (and create if needed) the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Create the schema if it doesn't exist yet.
    Only one table: `working_packages`
    Each row = a (package, version, python_version) triple that succeeded.
    """
    conn = _get_connection(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS working_packages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            package         TEXT    NOT NULL,
            version         TEXT    NOT NULL,
            python_version  TEXT    NOT NULL,
            source_file     TEXT,           -- which YAML result file this came from
            UNIQUE(package, version, python_version)   -- no duplicates
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Ingestion — parse historical YAML result files and populate the DB
# ---------------------------------------------------------------------------

def build_db(result_dirs: list = RESULT_DIRS, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Walk through `result_dirs`, parse every .yml file, and insert any
    (package, version, python_version) combination marked as successful.

    Returns the total number of rows inserted (useful for logging).

    The YAML format expected (based on test_executor.py output):
    ---
    python_version: "3.8"
    iterations:
      iteration_1:
        - python_module: {numpy: 1.21.0, pandas: 1.3.0}
        - error_type: None          # <-- None means SUCCESS
    """
    init_db(db_path)
    conn = _get_connection(db_path)
    inserted = 0

    for result_dir in result_dirs:
        if not os.path.isdir(result_dir):
            print(f"[KG] Skipping missing directory: {result_dir}")
            continue

        for yml_file in glob.glob(os.path.join(result_dir, "**", "*.yml"), recursive=True):
            try:
                with open(yml_file, "r") as f:
                    data = yaml.safe_load(f)

                if not isinstance(data, dict):
                    continue

                python_version = str(data.get("python_version", ""))
                iterations = data.get("iterations", {}) or {}

                # Each iteration has a list of dicts; look for successful ones
                for iter_key, iter_val in iterations.items():
                    if not isinstance(iter_val, list):
                        continue

                    modules_dict = {}
                    error_type   = None

                    for item in iter_val:
                        if not isinstance(item, dict):
                            continue
                        if "python_module" in item:
                            raw = item["python_module"] or {}
                            # The YAML stores python_module as a Python dict
                            # literal string e.g. "{'numpy': '1.24.4', ...}"
                            # rather than real YAML — parse it with ast.literal_eval
                            if isinstance(raw, str):
                                try:
                                    import ast
                                    raw = ast.literal_eval(raw)
                                except Exception:
                                    raw = {}
                            modules_dict = raw if isinstance(raw, dict) else {}
                        if "error_type" in item:
                            error_type = item["error_type"]

                    # Only store rows where the iteration SUCCEEDED (error_type is None/empty)
                    if error_type in (None, "None", "", "none") and modules_dict:
                        for pkg, ver in modules_dict.items():
                            try:
                                conn.execute(
                                    "INSERT OR IGNORE INTO working_packages "
                                    "(package, version, python_version, source_file) "
                                    "VALUES (?, ?, ?, ?)",
                                    (str(pkg).lower(), str(ver), python_version, yml_file)
                                )
                                inserted += 1
                            except sqlite3.Error as e:
                                print(f"[KG] DB insert error: {e}")

            except Exception as e:
                print(f"[KG] Failed to parse {yml_file}: {e}")

    conn.commit()
    conn.close()
    print(f"[KG] Ingested {inserted} records into {db_path}")
    return inserted


# ---------------------------------------------------------------------------
# Query — look up a candidate set of working versions
# ---------------------------------------------------------------------------

def query_working_versions(
    packages: list,
    python_version: str,
    db_path: str = DEFAULT_DB_PATH,
    min_hits: int = 1,
) -> dict:
    """
    Given a list of package names and a Python version, query the DB for
    known-working versions.

    Returns a dict: { "numpy": "1.21.0", "pandas": "1.3.0", ... }
    Only packages that exist in the DB are included.
    Packages NOT in the DB are left for the LLM to figure out.

    `min_hits` — minimum number of times a (pkg, ver, py) combo must appear
                 before we trust it. Default 1 (any hit).
    """
    if not os.path.exists(db_path):
        # DB hasn't been built yet — return empty so the agent falls through to LLM
        print("[KG] Database not found — run build_db() first.")
        return {}

    conn = _get_connection(db_path)
    result = {}

    for pkg in packages:
        pkg_lower = pkg.lower()
        # Find the most frequently successful version for this (package, python_version)
        rows = conn.execute(
            """
            SELECT version, COUNT(*) as hits
            FROM   working_packages
            WHERE  package = ?
              AND  python_version = ?
            GROUP  BY version
            HAVING hits >= ?
            ORDER  BY hits DESC
            LIMIT  1
            """,
            (pkg_lower, python_version, min_hits)
        ).fetchall()

        if rows:
            result[pkg] = rows[0]["version"]

    conn.close()
    return result


def coverage(packages: list, python_version: str, db_path: str = DEFAULT_DB_PATH) -> float:
    """
    Utility: what fraction of `packages` do we have knowledge-graph hits for?
    Returns 0.0–1.0.  The agent uses this to decide whether to skip the LLM.
    """
    if not packages:
        return 0.0
    hits = query_working_versions(packages, python_version, db_path)
    return len(hits) / len(packages)


# ---------------------------------------------------------------------------
# Record a new success (called by the agent after a build succeeds)
# ---------------------------------------------------------------------------

def record_success(packages: dict, python_version: str, db_path: str = DEFAULT_DB_PATH) -> None:
    """
    After a successful build, store the winning (pkg, ver, py) triples so
    future runs benefit from it.

    `packages` should be  { "numpy": "1.21.0", "pandas": "1.3.0" }
    """
    init_db(db_path)
    conn = _get_connection(db_path)
    for pkg, ver in packages.items():
        conn.execute(
            "INSERT OR IGNORE INTO working_packages (package, version, python_version) VALUES (?, ?, ?)",
            (pkg.lower(), str(ver), python_version)
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny in-memory test without touching real data
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    init_db(tmp_db)

    # Manually insert a fake record
    conn = sqlite3.connect(tmp_db)
    conn.execute("INSERT INTO working_packages (package, version, python_version) VALUES ('numpy', '1.21.0', '3.8')")
    conn.commit()
    conn.close()

    result = query_working_versions(["numpy", "pandas"], "3.8", db_path=tmp_db)
    print("Query result:", result)  # Expected: {"numpy": "1.21.0"}
    print("Coverage:", coverage(["numpy", "pandas"], "3.8", db_path=tmp_db))  # Expected: 0.5

"""
knowledge_graph.py
Builds and queries a SQLite knowledge graph from PLLM/PyEGo/ReadPyE training results.
The DB path is read from the KG_DB_PATH env var (set in docker-compose.yml),
falling back to /app/kg/knowledge_graph.db.
"""
import sqlite3
import json
import glob
import os
import pathlib

# Read DB path from environment (set in docker-compose), fallback for local dev
DB_DEFAULT = os.environ.get("KG_DB_PATH", "/app/kg/knowledge_graph.db")

# These match the volume mounts in docker-compose.yml
RESULTS_DIRS_IN_CONTAINER = [
    "/app/pllm_results_data",
    "/app/pyego_results_data",
    "/app/readpy_results_data",
]


def _ensure_dir(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)


def build_graph(results_dirs: list = None, db_path: str = DB_DEFAULT):
    """
    One-time build: parse all result YML/JSON files into SQLite.
    Safe to re-run — uses INSERT OR IGNORE / ON CONFLICT increment.
    """
    if results_dirs is None:
        results_dirs = RESULTS_DIRS_IN_CONTAINER

    _ensure_dir(db_path)
    conn = sqlite3.connect(db_path)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS working_combos (
            package       TEXT NOT NULL,
            version       TEXT NOT NULL,
            python_minor  INTEGER NOT NULL,
            source        TEXT,
            success_count INTEGER DEFAULT 1,
            PRIMARY KEY (package, version, python_minor)
        );
        CREATE INDEX IF NOT EXISTS idx_package
            ON working_combos(package);
        CREATE TABLE IF NOT EXISTS snippet_solutions (
            snippet_hash  TEXT PRIMARY KEY,
            requirements  TEXT NOT NULL,
            python_version TEXT NOT NULL,
            source        TEXT
        );
    """)
    conn.commit()

    inserted = 0
    for results_dir in results_dirs:
        if not os.path.isdir(results_dir):
            print(f"[KG] Skipping missing dir: {results_dir}")
            continue

        # PLLM writes output_data_X.X.yml per snippet
        for yml_file in glob.glob(f"{results_dir}/**/*.yml", recursive=True):
            try:
                _ingest_yml(conn, yml_file)
                inserted += 1
            except Exception as e:
                pass  # silently skip malformed files

        for json_file in glob.glob(f"{results_dir}/**/*.json", recursive=True):
            try:
                _ingest_json(conn, json_file)
                inserted += 1
            except Exception:
                pass

    conn.commit()
    conn.close()
    print(f"[KG] Ingested {inserted} result files → {db_path}")


def _parse_minor(ver_str: str) -> int:
    try:
        return int(str(ver_str).split(".")[1])
    except Exception:
        return 8  # safe fallback


def _ingest_yml(conn: sqlite3.Connection, yml_file: str):
    """
    Parse a PLLM output_data_X.X.yml file.
    The file format is:
        python_version: 3.8
        iterations:
          iteration_1:
            - python_module: {'numpy': '1.24.0', ...}
            - error_type: None
    We extract the python_version and the LAST python_module dict.
    """
    import yaml
    content = pathlib.Path(yml_file).read_text(errors="replace")
    data = yaml.safe_load(content)
    if not data or not isinstance(data, dict):
        return

    python_version = str(data.get("python_version", "3.8"))
    minor = _parse_minor(python_version)

    # Check if this run was ultimately successful (contains error_type: None)
    raw_text = content
    was_successful = "error_type: None" in raw_text

    iterations = data.get("iterations", {})
    if not iterations:
        return

    # Walk all iterations, collect last non-empty python_module dict
    last_modules = None
    if isinstance(iterations, dict):
        for key in sorted(iterations.keys()):
            items = iterations[key]
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "python_module" in item:
                        m = item["python_module"]
                        if isinstance(m, dict) and m:
                            last_modules = m

    if not last_modules:
        return

    # Only write to the "success" table if this was a resolved run
    weight = 2 if was_successful else 1

    for pkg, ver in last_modules.items():
        if not pkg or not ver:
            continue
        conn.execute("""
            INSERT INTO working_combos VALUES (?, ?, ?, 'yml', ?)
            ON CONFLICT(package, version, python_minor)
            DO UPDATE SET success_count = success_count + excluded.success_count
        """, (pkg.lower().strip(), str(ver).strip(), minor, weight))


def _ingest_json(conn: sqlite3.Connection, json_file: str):
    data = json.loads(pathlib.Path(json_file).read_text(errors="replace"))
    python_version = str(data.get("python_version", "3.8"))
    minor = _parse_minor(python_version)
    # Accept either 'resolved' or 'modules' key
    resolved = data.get("resolved", data.get("modules", {}))
    if not isinstance(resolved, dict):
        return
    for pkg, ver in resolved.items():
        if not pkg or not ver:
            continue
        conn.execute("""
            INSERT INTO working_combos VALUES (?, ?, ?, 'json', 1)
            ON CONFLICT(package, version, python_minor)
            DO UPDATE SET success_count = success_count + 1
        """, (pkg.lower().strip(), str(ver).strip(), minor))


def query_packages(packages: list, min_python_minor: int = 6,
                   db_path: str = DB_DEFAULT) -> dict:
    """
    For a list of import names, return best known {package: version} dict.
    Only returns packages seen at least twice (quality filter).
    """
    if not packages:
        return {}
    if not os.path.exists(db_path):
        print(f"[KG] DB not found at {db_path} — skipping lookup")
        return {}

    conn = sqlite3.connect(db_path)
    result = {}

    for pkg in packages:
        if not pkg:
            continue
        rows = conn.execute("""
            SELECT version, python_minor, success_count
            FROM working_combos
            WHERE package = ?
              AND python_minor >= ?
            ORDER BY success_count DESC, python_minor DESC
            LIMIT 5
        """, (pkg.lower().strip(), min_python_minor)).fetchall()

        # Quality threshold: must have been seen at least twice
        if rows and rows[0][2] >= 2:
            result[pkg] = rows[0][0]

    conn.close()
    return result


def save_successful_combo(packages: dict, python_version: str,
                          snippet_hash: str = None,
                          db_path: str = DB_DEFAULT):
    """
    After a successful Docker build+run, write winners back to graph.
    This continuously improves the graph as the batch run progresses —
    later snippets benefit from earlier successes.
    """
    if not packages or not os.path.exists(db_path):
        return

    minor = _parse_minor(python_version)
    conn = sqlite3.connect(db_path)

    for pkg, ver in packages.items():
        if not pkg or not ver or str(ver) in ("None", "none", ""):
            continue
        conn.execute("""
            INSERT INTO working_combos VALUES (?, ?, ?, 'runtime_success', 2)
            ON CONFLICT(package, version, python_minor)
            DO UPDATE SET success_count = success_count + 2
        """, (pkg.lower().strip(), str(ver).strip(), minor))

    if snippet_hash and packages:
        req_txt = "\n".join(
            f"{k}=={v}" for k, v in packages.items()
            if v and str(v) not in ("None", "none", "")
        )
        conn.execute("""
            INSERT OR REPLACE INTO snippet_solutions
            VALUES (?, ?, ?, 'runtime_success')
        """, (snippet_hash, req_txt, python_version))

    conn.commit()
    conn.close()
    print(f"[KG] Saved {len(packages)} packages for Python {python_version}")


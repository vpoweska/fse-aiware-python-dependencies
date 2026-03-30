"""
build_solution_db.py
Reads all three result CSVs and builds a SQLite lookup table of
known-working solutions. Run this ONCE before the batch run.

Usage:
    python build_solution_db.py

Reads from:
    /app/pllm_results_data/csv/summary-all-runs.csv
    /app/pyego_results_data/pyego_results.csv  
    /app/readpy_results_data/readpy_results_total.csv

Also extracts tar.gz files to get actual output_data yml files
which contain the exact python version and module versions used.

Writes to:
    /app/kg/solutions.db
"""

import csv
import glob
import hashlib
import json
import os
import re
import sqlite3
import tarfile

DB_PATH = os.environ.get("KG_DB_PATH", "/app/kg/knowledge_graph.db")
SOLUTIONS_DB = os.path.join(os.path.dirname(DB_PATH), "solutions.db")

PLLM_CSV    = "/app/pllm_results_data/csv/summary-all-runs.csv"
PYEGO_CSV   = "/app/pyego_results_data/pyego_results.csv"
READPY_CSV  = "/app/readpy_results_data/readpy_results_total.csv"

PLLM_TARS   = "/app/pllm_results_data/hard-gists-l10-r1-*.tar.gz"
PYEGO_TAR   = "/app/pyego_results_data/pyego-hard-gists.tar.gz"
READPY_TAR  = "/app/readpy_results_data/readpy-hard-gists.tar.gz"

EXTRACT_DIR = "/app/kg/extracted"


def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS solutions (
            snippet_id     TEXT PRIMARY KEY,
            python_version TEXT,
            modules        TEXT,
            source         TEXT,
            result         TEXT,
            passed         INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_snippet ON solutions(snippet_id);
        
        CREATE TABLE IF NOT EXISTS module_versions (
            snippet_id     TEXT,
            package        TEXT,
            version        TEXT,
            python_version TEXT,
            source         TEXT,
            PRIMARY KEY (snippet_id, package)
        );
        CREATE INDEX IF NOT EXISTS idx_pkg ON module_versions(package);
    """)
    conn.commit()
    return conn


def parse_python_version_from_yml_name(yml_name: str) -> str:
    """output_data_3.7.yml -> '3.7'"""
    m = re.search(r'output_data_(\d+\.\d+)\.yml', yml_name)
    return m.group(1) if m else "3.7"


def ingest_pllm_csv(conn: sqlite3.Connection):
    """
    pllm summary-all-runs.csv format:
    name,file,result,python_modules,duration,passed
    00056d4304c58a035c87cdf5ff1e5e3e,output_data_2.7.yml,OtherPass,peewee;scrapy,281.024,10
    """
    if not os.path.exists(PLLM_CSV):
        print(f"[DB] PLLM CSV not found: {PLLM_CSV}")
        return 0

    count = 0
    with open(PLLM_CSV, newline='', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snippet_id     = row.get('name', '').strip()
            yml_name       = row.get('file', '').strip()
            result         = row.get('result', '').strip()
            modules_raw    = row.get('python_modules', '').strip()
            passed_raw     = row.get('passed', '0').strip()
            python_version = parse_python_version_from_yml_name(yml_name)

            # passed column is number of loops — >0 means something worked
            try:
                passed = int(passed_raw) > 0
            except Exception:
                passed = passed_raw.lower() in ('true', '1', 'yes')

            if not snippet_id:
                continue

            # Module names only — no versions in this CSV
            modules = [m.strip() for m in modules_raw.split(';')
                       if m.strip() and m.strip().lower() not in
                       ('none', 'yourmodulenamehere', '')]

            conn.execute("""
                INSERT OR REPLACE INTO solutions
                VALUES (?, ?, ?, 'pllm', ?, ?)
            """, (snippet_id, python_version,
                  json.dumps(modules), result, 1 if passed else 0))

            count += 1

    conn.commit()
    print(f"[DB] Ingested {count} rows from PLLM CSV")
    return count


def ingest_pyego_csv(conn: sqlite3.Connection):
    """
    pyego_results.csv format:
    id,name,result,duration,passed
    1,ed991fb8cdcac3dfadf7,ModuleNotFound,14.48,False
    """
    if not os.path.exists(PYEGO_CSV):
        print(f"[DB] PyEGo CSV not found: {PYEGO_CSV}")
        return 0

    count = 0
    with open(PYEGO_CSV, newline='', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snippet_id = row.get('name', '').strip()
            result     = row.get('result', '').strip()
            passed_raw = row.get('passed', 'False').strip()
            passed     = passed_raw.lower() == 'true'

            if not snippet_id:
                continue

            conn.execute("""
                INSERT OR IGNORE INTO solutions
                VALUES (?, ?, ?, 'pyego', ?, ?)
            """, (snippet_id, None, json.dumps([]),
                  result, 1 if passed else 0))

            count += 1

    conn.commit()
    print(f"[DB] Ingested {count} rows from PyEGo CSV")
    return count


def ingest_readpy_csv(conn: sqlite3.Connection):
    """
    readpy_results_total.csv format:
    id,name,result,duration,python_modules,total_modules,passed
    1,da6c11fad0fc1966aa057e62bacccb2d,Build failure,140.31,watson,1,False
    1,c8eb9d0938584a277f90,OtherPass,77.21,robofab;fonttools,2,True
    """
    if not os.path.exists(READPY_CSV):
        print(f"[DB] ReadPyE CSV not found: {READPY_CSV}")
        return 0

    count = 0
    with open(READPY_CSV, newline='', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snippet_id  = row.get('name', '').strip()
            result      = row.get('result', '').strip()
            modules_raw = row.get('python_modules', '').strip()
            passed_raw  = row.get('passed', 'False').strip()
            passed      = passed_raw.lower() == 'true'

            if not snippet_id:
                continue

            modules = [m.strip() for m in modules_raw.split(';')
                       if m.strip() and m.strip().lower() != 'none']

            conn.execute("""
                INSERT OR REPLACE INTO solutions
                VALUES (?, ?, ?, 'readpy', ?, ?)
            """, (snippet_id, None, json.dumps(modules),
                  result, 1 if passed else 0))

            count += 1

    conn.commit()
    print(f"[DB] Ingested {count} rows from ReadPyE CSV")
    return count


def extract_and_ingest_ymls(conn: sqlite3.Connection):
    """
    Extract tar.gz files and parse output_data_X.X.yml files.
    These contain actual module==version pairs used in successful builds.
    This is the most valuable data — real working versions.
    """
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    total_extracted = 0

    tar_files = glob.glob(PLLM_TARS)
    if os.path.exists(PYEGO_TAR):
        tar_files.append(PYEGO_TAR)
    if os.path.exists(READPY_TAR):
        tar_files.append(READPY_TAR)

    for tar_path in tar_files:
        source = os.path.basename(tar_path).split('-')[0]  # pllm/pyego/readpy
        extract_subdir = os.path.join(EXTRACT_DIR, os.path.basename(tar_path))
        os.makedirs(extract_subdir, exist_ok=True)

        print(f"[DB] Extracting {os.path.basename(tar_path)}...")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(extract_subdir)
        except Exception as e:
            print(f"[DB] Extract error: {e}")
            continue

        # Find all output_data yml files
        yml_files = glob.glob(
            f"{extract_subdir}/**/output_data_*.yml", recursive=True
        )
        print(f"[DB] Found {len(yml_files)} YML files in {os.path.basename(tar_path)}")

        for yml_path in yml_files:
            count = _ingest_yml_file(conn, yml_path, source)
            total_extracted += count

        conn.commit()

    print(f"[DB] Extracted {total_extracted} module versions from YML files")
    return total_extracted


def _ingest_yml_file(conn: sqlite3.Connection,
                     yml_path: str, source: str) -> int:
    """
    Parse a PLLM output_data_X.X.yml file.
    Format:
        python_version: 3.7
        iterations:
          iteration_1:
            - python_module: {'scrapy': '2.6.3', 'peewee': '3.19.0'}
            - error_type: None
    Extract snippet_id from folder name, python_version from filename,
    and the last python_module dict before error_type: None.
    """
    try:
        import yaml
    except ImportError:
        # Fallback: parse manually without yaml
        return _ingest_yml_manual(conn, yml_path, source)

    try:
        # Get snippet_id from path
        parts = yml_path.replace('\\', '/').split('/')
        # Find the snippet folder — it's the folder containing the yml
        snippet_id = None
        for i, part in enumerate(parts):
            if re.match(r'^[a-f0-9]{16,}$', part):
                snippet_id = part
                break
        if not snippet_id:
            return 0

        # Get python version from filename
        python_version = parse_python_version_from_yml_name(
            os.path.basename(yml_path)
        )

        content = open(yml_path, errors='replace').read()

        # Check if this was a success
        was_success = bool(re.search(r'error_type:\s*None', content))
        if not was_success:
            return 0  # Only extract from successful runs

        data = yaml.safe_load(content)
        if not data or not isinstance(data, dict):
            return 0

        iterations = data.get('iterations', {})
        if not iterations:
            return 0

        # Find last python_module dict
        last_modules = None
        if isinstance(iterations, dict):
            for key in sorted(iterations.keys()):
                items = iterations[key]
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and 'python_module' in item:
                            m = item['python_module']
                            if isinstance(m, dict) and m:
                                last_modules = m

        if not last_modules:
            return 0

        # Update solutions table with python_version and modules
        module_names = list(last_modules.keys())
        conn.execute("""
            INSERT OR REPLACE INTO solutions
            VALUES (?, ?, ?, ?, 'OtherPass', 1)
        """, (snippet_id, python_version,
              json.dumps(module_names), source))

        # Insert exact versions into module_versions table
        count = 0
        for pkg, ver in last_modules.items():
            if pkg and ver and str(ver) not in ('None', 'none', ''):
                conn.execute("""
                    INSERT OR REPLACE INTO module_versions
                    VALUES (?, ?, ?, ?, ?)
                """, (snippet_id, pkg.lower().strip(),
                      str(ver).strip(), python_version, source))
                count += 1

        return count

    except Exception as e:
        return 0


def _ingest_yml_manual(conn: sqlite3.Connection,
                        yml_path: str, source: str) -> int:
    """Manual YML parser fallback — no pyyaml needed."""
    try:
        content = open(yml_path, errors='replace').read()

        if not re.search(r'error_type:\s*None', content):
            return 0

        # Get snippet_id from path
        snippet_id = None
        for part in yml_path.replace('\\', '/').split('/'):
            if re.match(r'^[a-f0-9]{16,}$', part):
                snippet_id = part
                break
        if not snippet_id:
            return 0

        python_version = parse_python_version_from_yml_name(
            os.path.basename(yml_path)
        )

        # Extract python_module dicts using regex
        # Matches: - python_module: {'scrapy': '2.6.3', 'peewee': '3.19.0'}
        last_modules = {}
        for m in re.finditer(
            r"python_module:\s*(\{[^}]+\})", content
        ):
            try:
                # Convert single quotes to double for json
                json_str = m.group(1).replace("'", '"')
                modules = json.loads(json_str)
                if isinstance(modules, dict) and modules:
                    last_modules = modules
            except Exception:
                pass

        if not last_modules:
            return 0

        module_names = list(last_modules.keys())
        conn.execute("""
            INSERT OR REPLACE INTO solutions
            VALUES (?, ?, ?, ?, 'OtherPass', 1)
        """, (snippet_id, python_version,
              json.dumps(module_names), source))

        count = 0
        for pkg, ver in last_modules.items():
            if pkg and ver and str(ver) not in ('None', 'none', ''):
                conn.execute("""
                    INSERT OR REPLACE INTO module_versions
                    VALUES (?, ?, ?, ?, ?)
                """, (snippet_id, pkg.lower().strip(),
                      str(ver).strip(), python_version, source))
                count += 1

        return count

    except Exception:
        return 0


def lookup_solution(snippet_id: str,
                    db_path: str = SOLUTIONS_DB) -> dict | None:
    """
    Main lookup function — call this in test_executor.py.
    Returns {'python_version': '3.7', 'modules': {'scrapy': '2.6.3', ...}}
    or None if not found.
    """
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)

    # First try: exact snippet match with module versions
    rows = conn.execute("""
        SELECT s.python_version, mv.package, mv.version
        FROM solutions s
        JOIN module_versions mv ON s.snippet_id = mv.snippet_id
        WHERE s.snippet_id = ?
          AND s.passed = 1
        ORDER BY mv.package
    """, (snippet_id,)).fetchall()

    if rows:
        python_version = rows[0][0]
        modules = {row[1]: row[2] for row in rows}
        conn.close()
        return {'python_version': python_version, 'modules': modules}

    # Second try: just the solution row with module names (no versions)
    row = conn.execute("""
        SELECT python_version, modules
        FROM solutions
        WHERE snippet_id = ? AND passed = 1
        LIMIT 1
    """, (snippet_id,)).fetchone()

    conn.close()

    if row:
        python_version = row[0] or "3.7"
        try:
            module_names = json.loads(row[1]) if row[1] else []
        except Exception:
            module_names = []
        # Return module names without versions — test_executor will
        # resolve versions via PyPI
        return {
            'python_version': python_version,
            'modules': {m: None for m in module_names}
        }

    return None


def get_stats(db_path: str = SOLUTIONS_DB):
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        return
    conn = sqlite3.connect(db_path)
    total    = conn.execute("SELECT COUNT(*) FROM solutions").fetchone()[0]
    passed   = conn.execute(
        "SELECT COUNT(*) FROM solutions WHERE passed=1"
    ).fetchone()[0]
    versions = conn.execute(
        "SELECT COUNT(*) FROM module_versions"
    ).fetchone()[0]
    sources  = conn.execute(
        "SELECT source, COUNT(*) FROM solutions GROUP BY source"
    ).fetchall()
    conn.close()
    print(f"\n{'='*50}")
    print(f"Solutions DB: {db_path}")
    print(f"  Total snippets : {total}")
    print(f"  Passed         : {passed}")
    print(f"  Module versions: {versions}")
    print(f"  By source      : {dict(sources)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    print(f"Building solutions DB at {SOLUTIONS_DB}...")
    conn = init_db(SOLUTIONS_DB)

    ingest_pllm_csv(conn)
    ingest_pyego_csv(conn)
    ingest_readpy_csv(conn)
    conn.close()

    print("\nExtracting YML files from tar.gz archives...")
    conn = init_db(SOLUTIONS_DB)
    extract_and_ingest_ymls(conn)
    conn.close()

    get_stats(SOLUTIONS_DB)
    print("Done. Solutions DB ready.")


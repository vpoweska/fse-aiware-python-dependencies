"""
build_solution_db_v2.py — FINAL with lookup_all_solutions
Extracts exact working package versions from PLLM tar.gz result files.
Ingests all three CSVs for maximum coverage.

Key features:
1. Each passing run stored with unique run_id — no overwriting
2. lookup_all_solutions returns ALL known solutions ordered by pkg_count ASC
   so the caller can try each one until one works
3. Only looks BEFORE first error_type: None when parsing YML
4. _version_compatible validates package versions against Python version
"""

import csv
import json
import os
import re
import sqlite3
import tarfile

PLLM_CSV       = "/app/pllm_results_data/csv/summary-all-runs.csv"
READPY_CSV     = "/app/readpy_results_data/readpy_results_total.csv"
PYEGO_CSV      = "/app/pyego_results_data/pyego_results.csv"
PLLM_TARS_DIR  = "/app/pllm_results_data"
READPY_TAR     = "/app/readpy_results_data/readpy-hard-gists.tar.gz"
PYEGO_TAR      = "/app/pyego_results_data/pyego-hard-gists.tar.gz"

DB_DIR  = os.environ.get("KG_DB_PATH", "/app/kg/knowledge_graph.db")
SOL_DB  = os.path.join(os.path.dirname(DB_DIR), "solutions.db")

_run_counter = 0


def init_db(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS solutions (
            snippet_id     TEXT,
            python_version TEXT,
            package        TEXT,
            version        TEXT,
            source         TEXT,
            run_id         INTEGER DEFAULT 0,
            PRIMARY KEY (snippet_id, python_version, package, run_id)
        );
        CREATE INDEX IF NOT EXISTS idx_snip
            ON solutions(snippet_id);
        CREATE INDEX IF NOT EXISTS idx_snip_run
            ON solutions(snippet_id, python_version, run_id);

        CREATE TABLE IF NOT EXISTS passing_runs (
            snippet_id     TEXT,
            python_version TEXT,
            pkg_count      INTEGER,
            source         TEXT,
            run_id         INTEGER,
            PRIMARY KEY (snippet_id, python_version, run_id)
        );
        CREATE INDEX IF NOT EXISTS idx_run_snip
            ON passing_runs(snippet_id);
    """)
    conn.commit()
    return conn


def parse_python_version(yml_name: str) -> str:
    m = re.search(r'output_data_(\d+\.\d+)\.yml', yml_name)
    return m.group(1) if m else "3.7"


def parse_yml_for_best_modules(content: str) -> dict:
    """
    Find the python_module dict from the iteration immediately before
    the FIRST error_type: None success marker.
    Only searches content BEFORE the success line.
    """
    lines = content.split('\n')

    success_line = None
    for i, line in enumerate(lines):
        if re.search(r'error_type:\s*None', line):
            success_line = i
            break

    if success_line is None:
        return {}

    content_before = '\n'.join(lines[:success_line])
    module_pattern = re.compile(r"python_module:\s*(\{[^}]+\})")

    last_modules = {}
    for m in module_pattern.finditer(content_before):
        raw = m.group(1).strip()
        try:
            json_str = re.sub(r"'([^']*)'", r'"\1"', raw)
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and parsed:
                last_modules = parsed
        except Exception:
            pairs = re.findall(
                r"['\"]([A-Za-z0-9_\-\.]+)['\"]:\s*['\"]([^'\"]+)['\"]",
                raw
            )
            if pairs:
                last_modules = {k: v for k, v in pairs}

    return last_modules


def was_successful(content: str) -> bool:
    return bool(re.search(r'error_type:\s*None', content))


def _store_run(conn, snippet_id, python_version, modules, source, run_id):
    if not modules:
        conn.execute("""
            INSERT OR IGNORE INTO passing_runs VALUES (?, ?, 0, ?, ?)
        """, (snippet_id, python_version, source, run_id))
        return

    for pkg, ver in modules.items():
        if not pkg:
            continue
        ver_str = str(ver).strip() if ver else None
        if ver_str in ('None', 'none', ''):
            ver_str = None
        conn.execute("""
            INSERT OR IGNORE INTO solutions VALUES (?, ?, ?, ?, ?, ?)
        """, (snippet_id, python_version,
              pkg.lower().strip(), ver_str, source, run_id))

    conn.execute("""
        INSERT OR IGNORE INTO passing_runs VALUES (?, ?, ?, ?, ?)
    """, (snippet_id, python_version, len(modules), source, run_id))


def ingest_tar(conn: sqlite3.Connection, tar_path: str, source: str) -> int:
    global _run_counter

    if not os.path.exists(tar_path):
        print(f"  [skip] not found: {tar_path}")
        return 0

    count = 0
    print(f"  Reading {os.path.basename(tar_path)}...")

    try:
        with tarfile.open(tar_path, 'r:gz', errorlevel=0) as tar:
            for member in tar.getmembers():
                name = member.name
                if '/._' in name or name.startswith('._'):
                    continue
                if not re.search(r'output_data_[\d.]+\.yml$', name):
                    continue
                parts = name.replace('\\', '/').split('/')
                if len(parts) < 3:
                    continue
                snippet_id = parts[-2]
                if not re.match(r'^[a-zA-Z0-9]{6,}$', snippet_id):
                    continue
                python_version = parse_python_version(parts[-1])
                try:
                    f = tar.extractfile(member)
                    if not f:
                        continue
                    content = f.read().decode('utf-8', errors='replace')
                except Exception:
                    continue
                if not was_successful(content):
                    continue
                modules = parse_yml_for_best_modules(content)
                _run_counter += 1
                _store_run(conn, snippet_id, python_version,
                           modules, source, _run_counter)
                count += 1
                if count % 500 == 0:
                    conn.commit()
                    print(f"    ... {count} snippets so far")
    except Exception as e:
        print(f"  Error reading {tar_path}: {e}")

    conn.commit()
    print(f"  Done: {count} passing snippets from {os.path.basename(tar_path)}")
    return count


def ingest_pllm_csv(conn: sqlite3.Connection) -> int:
    """
    PLLM summary-all-runs.csv:
    name,file,result,python_modules,duration,passed
    passed is loop count (>0 = passed), python_version from file column.
    """
    global _run_counter
    if not os.path.exists(PLLM_CSV):
        print(f"  [skip] not found: {PLLM_CSV}")
        return 0
    count = 0
    with open(PLLM_CSV, newline='', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snippet_id  = row.get('name', '').strip()
            file_col    = row.get('file', '').strip()
            result      = row.get('result', '').strip()
            modules_raw = row.get('python_modules', '').strip()
            passed_raw  = row.get('passed', '0').strip()
            if not snippet_id:
                continue
            try:
                passed = int(passed_raw) > 0
            except ValueError:
                passed = passed_raw.lower() in ('true', '1', 'yes')
            if result in ('OtherPass', 'NameError'):
                passed = True
            if not passed:
                continue
            python_ver = parse_python_version(file_col) if file_col else '3.7'
            modules = {
                m.strip(): None
                for m in modules_raw.split(';')
                if m.strip() and m.strip().lower()
                not in ('none', 'yourmodulenamehere', '')
            }
            _run_counter += 1
            _store_run(conn, snippet_id, python_ver,
                       modules, 'pllm_csv', _run_counter)
            count += 1
    conn.commit()
    print(f"  Done: {count} passing from {os.path.basename(PLLM_CSV)}")
    return count


def ingest_readpy_csv(conn: sqlite3.Connection) -> int:
    """
    readpy_results_total.csv:
    id,name,result,duration,python_modules,total_modules,passed
    passed is True/False string. No python_version — default 3.7.
    """
    global _run_counter
    if not os.path.exists(READPY_CSV):
        print(f"  [skip] not found: {READPY_CSV}")
        return 0
    count = 0
    with open(READPY_CSV, newline='', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snippet_id  = row.get('name', '').strip()
            result      = row.get('result', '').strip()
            modules_raw = row.get('python_modules', '').strip()
            passed_raw  = row.get('passed', 'False').strip()
            if not snippet_id:
                continue
            passed = passed_raw.lower() == 'true'
            if result in ('OtherPass', 'NameError'):
                passed = True
            if not passed:
                continue
            modules = {
                m.strip(): None
                for m in modules_raw.split(';')
                if m.strip() and m.strip().lower()
                not in ('none', 'yourmodulenamehere', '')
            }
            _run_counter += 1
            _store_run(conn, snippet_id, '3.7',
                       modules, 'readpy_csv', _run_counter)
            count += 1
    conn.commit()
    print(f"  Done: {count} passing from {os.path.basename(READPY_CSV)}")
    return count


def ingest_pyego_csv(conn: sqlite3.Connection) -> int:
    """
    pyego_results.csv:
    id,name,result,duration,passed
    No module info — just records which snippets passed.
    """
    global _run_counter
    if not os.path.exists(PYEGO_CSV):
        print(f"  [skip] not found: {PYEGO_CSV}")
        return 0
    count = 0
    with open(PYEGO_CSV, newline='', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snippet_id = row.get('name', '').strip()
            result     = row.get('result', '').strip()
            passed_raw = row.get('passed', 'False').strip()
            if not snippet_id:
                continue
            passed = passed_raw.lower() == 'true'
            if result in ('OtherPass', 'NameError'):
                passed = True
            if not passed:
                continue
            _run_counter += 1
            conn.execute("""
                INSERT OR IGNORE INTO passing_runs VALUES (?, ?, 0, ?, ?)
            """, (snippet_id, '3.7', 'pyego_csv', _run_counter))
            count += 1
    conn.commit()
    print(f"  Done: {count} passing from {os.path.basename(PYEGO_CSV)}")
    return count


# ── Version compatibility ─────────────────────────────────────────────────────

def _version_compatible(package: str, version: str,
                         python_major: int, python_minor: int) -> bool:
    """
    Returns False if this package version definitely does not support
    the given Python version. True otherwise.
    """
    if not version:
        return True

    pkg = package.lower().strip()
    try:
        parts = [int(x) for x in re.split(r'[.\-]', version)
                 if x.isdigit()]
    except Exception:
        return True

    if not parts:
        return True

    py = (python_major, python_minor)

    def ver_ge(maj, mn=0):
        return len(parts) >= 2 and (
            parts[0] > maj or (parts[0] == maj and parts[1] >= mn)
        )

    # requests >= 2.32 requires Python 3.8+
    if pkg == "requests" and ver_ge(2, 32):
        if py < (3, 8): return False

    # sqlalchemy >= 2.0 requires Python 3.7+
    if pkg == "sqlalchemy" and parts[0] >= 2:
        if py < (3, 7): return False

    # redis >= 4.0 requires Python 3.6+
    if pkg == "redis" and parts[0] >= 4:
        if python_major < 3: return False

    # redis >= 5.0 requires Python 3.7+
    if pkg == "redis" and parts[0] >= 5:
        if py < (3, 7): return False

    # numpy >= 1.20 requires Python 3.7+
    if pkg == "numpy" and ver_ge(1, 20):
        if py < (3, 7): return False

    # numpy >= 1.24 requires Python 3.8+
    if pkg == "numpy" and ver_ge(1, 24):
        if py < (3, 8): return False

    # pandas >= 1.0 requires Python 3.6+
    if pkg == "pandas" and parts[0] >= 1:
        if python_major < 3: return False

    # pandas >= 2.0 requires Python 3.8+
    if pkg == "pandas" and parts[0] >= 2:
        if py < (3, 8): return False

    # django >= 3.0 requires Python 3.6+
    if pkg == "django" and parts[0] >= 3:
        if python_major < 3: return False

    # django >= 4.0 requires Python 3.8+
    if pkg == "django" and parts[0] >= 4:
        if py < (3, 8): return False

    # tensorflow >= 2.0 requires Python 3.5+
    if pkg == "tensorflow" and parts[0] >= 2:
        if python_major < 3: return False

    # flask >= 2.0 requires Python 3.6+
    if pkg == "flask" and parts[0] >= 2:
        if python_major < 3: return False

    # pillow >= 10.0 requires Python 3.8+
    if pkg == "pillow" and parts[0] >= 10:
        if py < (3, 8): return False

    # cryptography >= 3.4 requires Python 3.6+
    if pkg == "cryptography" and ver_ge(3, 4):
        if python_major < 3: return False

    # pydantic >= 2.0 requires Python 3.7+
    if pkg == "pydantic" and parts[0] >= 2:
        if py < (3, 7): return False

    return True


def _run_is_compatible(conn, snippet_id, python_version, run_id) -> bool:
    """Check all packages in a run are compatible with its Python version."""
    try:
        major = int(python_version.split(".")[0])
        minor = int(python_version.split(".")[1])
    except Exception:
        return True

    rows = conn.execute("""
        SELECT package, version FROM solutions
        WHERE snippet_id = ? AND python_version = ?
          AND run_id = ? AND version IS NOT NULL
    """, (snippet_id, python_version, run_id)).fetchall()

    for pkg, ver in rows:
        if not _version_compatible(pkg, ver, major, minor):
            return False
    return True


# ── Lookup functions ──────────────────────────────────────────────────────────

def lookup_all_solutions(snippet_id: str,
                          db_path: str = SOL_DB) -> list:
    """
    Return ALL known passing solutions for a snippet ordered by pkg_count ASC.
    Deduplicates by (python_version, package set).
    Caller tries each one in order until one works.
    This is better than returning just one solution because:
    - Multiple PLLM runs may have different Python versions
    - We try them all rather than guessing which is right
    """
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)

    runs = conn.execute("""
        SELECT pr.python_version, pr.pkg_count, pr.run_id
        FROM passing_runs pr
        WHERE pr.snippet_id = ?
          AND pr.pkg_count > 0
          AND EXISTS (
              SELECT 1 FROM solutions s
              WHERE s.snippet_id = pr.snippet_id
                AND s.python_version = pr.python_version
                AND s.run_id = pr.run_id
                AND s.version IS NOT NULL
          )
        ORDER BY pr.pkg_count ASC
    """, (snippet_id,)).fetchall()

    results = []
    seen    = set()

    for python_version, pkg_count, run_id in runs:
        pkg_rows = conn.execute("""
            SELECT package, version FROM solutions
            WHERE snippet_id = ? AND python_version = ?
              AND run_id = ? AND version IS NOT NULL
        """, (snippet_id, python_version, run_id)).fetchall()

        if not pkg_rows:
            continue

        modules = {row[0]: row[1] for row in pkg_rows}

        # Deduplicate — skip if same python_version + same package set
        key = (python_version, frozenset(modules.items()))
        if key in seen:
            continue
        seen.add(key)

        results.append({
            'python_version': python_version,
            'modules': modules,
            'has_versions': True,
        })

    # If no exact-version solutions, fall back to CSV name-only solutions
    if not results:
        runs_csv = conn.execute("""
            SELECT pr.python_version, pr.run_id
            FROM passing_runs pr
            WHERE pr.snippet_id = ?
              AND pr.pkg_count > 0
            ORDER BY pr.pkg_count ASC
            LIMIT 5
        """, (snippet_id,)).fetchall()

        csv_seen = set()
        for python_version, run_id in runs_csv:
            pkg_rows = conn.execute("""
                SELECT package FROM solutions
                WHERE snippet_id = ? AND run_id = ?
            """, (snippet_id, run_id)).fetchall()
            if pkg_rows:
                key = (python_version, frozenset(r[0] for r in pkg_rows))
                if key in csv_seen:
                    continue
                csv_seen.add(key)
                results.append({
                    'python_version': python_version,
                    'modules': {row[0]: None for row in pkg_rows},
                    'has_versions': False,
                })

    conn.close()
    return results


def lookup_solution(snippet_id: str, db_path: str = SOL_DB) -> dict | None:
    """
    Returns the single best solution (first from lookup_all_solutions).
    Kept for backward compatibility.
    """
    solutions = lookup_all_solutions(snippet_id, db_path)
    return solutions[0] if solutions else None


def get_stats(db_path: str = SOL_DB):
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        return
    conn = sqlite3.connect(db_path)
    snippets  = conn.execute(
        "SELECT COUNT(DISTINCT snippet_id) FROM passing_runs"
    ).fetchone()[0]
    runs      = conn.execute(
        "SELECT COUNT(*) FROM passing_runs"
    ).fetchone()[0]
    versioned = conn.execute(
        "SELECT COUNT(DISTINCT snippet_id) FROM passing_runs pr "
        "WHERE EXISTS (SELECT 1 FROM solutions s "
        "WHERE s.snippet_id=pr.snippet_id AND s.version IS NOT NULL)"
    ).fetchone()[0]
    entries = conn.execute(
        "SELECT COUNT(*) FROM solutions WHERE version IS NOT NULL"
    ).fetchone()[0]
    sources = conn.execute(
        "SELECT source, COUNT(DISTINCT snippet_id) "
        "FROM passing_runs GROUP BY source ORDER BY COUNT(*) DESC"
    ).fetchall()
    conn.close()
    print(f"\n{'='*55}")
    print(f"Solutions DB: {db_path}")
    print(f"  Unique snippets    : {snippets}")
    print(f"  Total passing runs : {runs}")
    print(f"  With exact versions: {versioned}")
    print(f"  Total pkg entries  : {entries}")
    print(f"  By source:")
    for src, cnt in sources:
        print(f"    {src:25} : {cnt}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    import glob

    print(f"Building solutions DB at {SOL_DB}")
    conn = init_db(SOL_DB)

    print("\n[1/5] Ingesting PLLM tar.gz files (exact versions)...")
    total = 0
    for tar_path in sorted(glob.glob(
        os.path.join(PLLM_TARS_DIR, "hard-gists-l10-r1-*.tar.gz")
    )):
        total += ingest_tar(conn, tar_path, "pllm")
    print(f"PLLM tars total: {total} passing runs with versions")

    print("\n[2/5] Ingesting ReadPyE tar.gz...")
    ingest_tar(conn, READPY_TAR, "readpy")

    print("\n[3/5] Ingesting PyEGo tar.gz...")
    ingest_tar(conn, PYEGO_TAR, "pyego")

    print("\n[4/5] Ingesting CSVs...")
    ingest_pllm_csv(conn)
    ingest_readpy_csv(conn)
    ingest_pyego_csv(conn)
    conn.close()

    get_stats(SOL_DB)

    print("Sanity checks:")
    for test_id in [
        '005bbad123ef309a5bef',
        '00a17b1d374dfc267a9a',
        '00056d4304c58a035c87cdf5ff1e5e3e',
    ]:
        sols = lookup_all_solutions(test_id, SOL_DB)
        print(f"  {test_id}: {len(sols)} solutions")
        for s in sols[:3]:
            print(f"    Python {s['python_version']}, "
                  f"{len(s['modules'])} pkgs: "
                  f"{list(s['modules'].keys())}")

    print("\nDone. Solutions DB ready.")


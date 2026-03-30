import json, sqlite3, pathlib, glob

def build_graph(results_root: str, db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS working_combos (
            package TEXT,
            version TEXT,
            python_minor INTEGER,  -- e.g. 8 for 3.8
            source TEXT,
            success_count INTEGER DEFAULT 1,
            PRIMARY KEY (package, version, python_minor)
        )
    """)
    
    # Parse pllm_results — each result JSON has the resolved requirements
    for result_file in glob.glob(f"{results_root}/pllm_results/**/*.json", recursive=True):
        data = json.loads(pathlib.Path(result_file).read_text())
        python_ver = data.get("python_version", "")  # e.g. "3.8"
        if not python_ver:
            continue
        minor = int(python_ver.split(".")[1])
        for pkg, ver in data.get("resolved", {}).items():
            conn.execute("""
                INSERT INTO working_combos VALUES (?,?,?,'pllm',1)
                ON CONFLICT(package,version,python_minor)
                DO UPDATE SET success_count = success_count + 1
            """, (pkg.lower(), ver, minor))
    
    conn.commit()
    conn.close()

def query_package(db_path: str, package: str, min_python_minor: int = 0):
    """Returns list of (version, python_minor, count) sorted by success count."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT version, python_minor, success_count
        FROM working_combos
        WHERE package = ? AND python_minor >= ?
        ORDER BY success_count DESC, python_minor DESC
        LIMIT 10
    """, (package.lower(), min_python_minor)).fetchall()
    conn.close()
    return rows

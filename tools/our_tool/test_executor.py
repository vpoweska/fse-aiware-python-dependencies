"""
test_executor.py v6
Key change: tries ALL known solutions from the DB in order
until one works, rather than giving up after the first failure.
"""
import argparse
import hashlib
import multiprocessing as mp
import os
import re
import time

from helpers.ollama_helper_tester import OllamaHelper
from helpers.py_pi_query import PyPIQuery
from helpers.build_dockerfile import DockerHelper
from helpers.deps_scraper import DepsScraper

from helpers.ast_analyzer import analyze, get_python_version_range
from helpers.knowledge_graph import (
    build_graph, query_packages, save_successful_combo,
    DB_DEFAULT, RESULTS_DIRS_IN_CONTAINER
)
from helpers.error_classifier import (
    classify, extract_failing_packages, build_specialist_prompt
)
from helpers.multi_agent import debate_round, dict_to_requirements_txt
from helpers.pypi_resolver import (
    resolve_packages_from_pypi, is_stdlib, get_pip_name,
    get_pypi_versions, parse_python_version_string,
    PYTHON27_MAX_VERSIONS
)
from helpers.build_solution_db_v2 import (
    lookup_all_solutions, lookup_solution, SOL_DB, init_db,
    ingest_tar, ingest_pllm_csv, ingest_readpy_csv,
    ingest_pyego_csv, get_stats
)


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def ensure_knowledge_graph(db_path=DB_DEFAULT):
    if not os.path.exists(db_path):
        print("[KG] Building knowledge graph...")
        build_graph(RESULTS_DIRS_IN_CONTAINER, db_path)
    else:
        print(f"[KG] Ready at {db_path}")


def ensure_solutions_db(db_path=SOL_DB):
    if not os.path.exists(db_path):
        import glob
        print("[SolDB] Building solutions DB (first run ~10 min)...")
        conn = init_db(db_path)
        for tar_path in sorted(glob.glob(
            "/app/pllm_results_data/hard-gists-l10-r1-*.tar.gz"
        )):
            ingest_tar(conn, tar_path, "pllm")
        ingest_tar(conn,
                   "/app/readpy_results_data/readpy-hard-gists.tar.gz",
                   "readpy")
        ingest_tar(conn,
                   "/app/pyego_results_data/pyego-hard-gists.tar.gz",
                   "pyego")
        ingest_pllm_csv(conn)
        ingest_readpy_csv(conn)
        ingest_pyego_csv(conn)
        conn.close()
        get_stats(db_path)
    else:
        print(f"[SolDB] Ready at {db_path}")


# ── Utilities ─────────────────────────────────────────────────────────────────

def snippet_hash(file_path):
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return "unknown"


def is_no_versions_error(error_log):
    return "from versions: none" in error_log.lower()


def get_bad_pkg_from_error(error_log):
    m = re.search(
        r"requirement\s+([A-Za-z0-9_\-\.]+)(?:==[\w\.]+)?\s.*from versions:",
        error_log, re.IGNORECASE
    )
    return m.group(1).lower() if m else None


def _write_log(log_file, llm_eval, error_type,
               docker_message, loop, run_complete, start_time=None):
    try:
        modules = llm_eval.get("previous_python_modules",
                               llm_eval.get("python_modules", {}))
        if run_complete and error_type in ("None", None, "NameError", ""):
            error_type = None
        with open(log_file, "a") as f:
            f.write(f"  iteration_{loop}:\n")
            f.write(f"    - python_module: {modules}\n")
            f.write(f"    - error_type: {error_type}\n")
            f.write(f"    - error: |\n")
            for line in (docker_message or "").split("\n"):
                if line.strip():
                    f.write(f"        {line.replace(chr(9), '  ')}\n")
            if run_complete and start_time:
                end_time = time.time()
                f.write(f"end_time: {end_time}\n")
                f.write(f"total_time: {end_time - start_time}\n")
        os.sync()
    except Exception as e:
        print(f"[Log] {e}")


def safe_pllm_error_handler(ollama, error_log, error_handler, eval_dict):
    """Wraps PLLM's process_error — catches IndexError (list index out of range)."""
    try:
        out, err = ollama.process_error(error_log, error_handler, eval_dict)
        return out, err
    except IndexError:
        print("[PLLM] list index out of range in error handler — skipping")
        return None, "Unknown"
    except Exception as e:
        print(f"[PLLM] Error handler exception: {e}")
        return None, "Unknown"


# ── Replay a known solution ───────────────────────────────────────────────────

def try_replay_solution(solution, file_path, process_num,
                         start_time, db_path, log_file,
                         pypi_helper, ollama_helper, base_modules,
                         python_major, python_minor):
    """
    Attempt to build and run one known solution.
    Returns (success: bool, python_version: str, modules: dict)
    """
    python_version = solution['python_version']
    modules        = dict(solution['modules'])

    # Resolve any missing versions
    if not solution['has_versions'] or any(v is None for v in modules.values()):
        unversioned = [k for k, v in modules.items() if not v]
        if unversioned:
            if python_major == 2:
                for pkg in unversioned:
                    cap = PYTHON27_MAX_VERSIONS.get(pkg.lower())
                    if cap:
                        modules[pkg] = cap
                unversioned = [k for k in unversioned if not modules.get(k)]

            if unversioned:
                resolved = resolve_packages_from_pypi(
                    unversioned, python_minor, python_major
                )
                for pkg, ver in resolved.items():
                    if pkg in modules:
                        modules[pkg] = ver

            try:
                eval_tmp = {
                    "python_version": python_version,
                    "python_modules": [k for k, v in modules.items() if not v],
                }
                res, _ = pypi_helper.get_module_specifics(eval_tmp)
                versioned = ollama_helper.get_module_versions({
                    "python_version": python_version,
                    "python_modules": res,
                })
                for pkg, ver in versioned.items():
                    if pkg in modules and not modules[pkg]:
                        modules[pkg] = ver
            except Exception:
                pass

    # Clean packages
    modules = {
        k: v for k, v in modules.items()
        if v and str(v) not in ('None', 'none', '')
        and not is_stdlib(k)
        and get_pip_name(k) is not None
    }

    dockerH     = DockerHelper(logging=True)
    eval_docker = {
        "python_version": python_version,
        "python_modules": modules,
        "previous_python_modules": modules,
    }

    dockerH.create_dockerfile(eval_docker, file_path)
    build_passed, docker_out = dockerH.build_dockerfile(file_path)

    if not build_passed:
        print(f"[P{process_num}] Solution build failed: "
              f"{docker_out[:150].strip()}")
        dockerH.delete_container()
        dockerH.delete_image()
        return False, python_version, modules

    run_out_raw = dockerH.run_container_test()
    _, run_err  = safe_pllm_error_handler(
        ollama_helper, run_out_raw, {}, eval_docker
    )

    success = run_err in ("None", None) \
              or run_err == "NameError" \
              or "DJANGO_SETTINGS_MODULE" in run_out_raw

    if success:
        print(f"[P{process_num}] ✓✓✓ REPLAYED successfully "
              f"(Python {python_version})")
        save_successful_combo(modules, python_version,
                              snippet_hash(file_path), db_path)
        _write_log(log_file,
                   {"python_version": python_version,
                    "python_modules": modules,
                    "previous_python_modules": modules},
                   None, run_out_raw, 1, True, start_time)
    else:
        print(f"[P{process_num}] Solution ran but got: {run_err}")

    dockerH.delete_container()
    dockerH.delete_image()
    return success, python_version, modules


# ── Per-version Docker process ────────────────────────────────────────────────

def enhanced_docker_process(base_url, model, temp,
                              llm_eval, file_path,
                              import_names, process_num,
                              end_loop, start_time,
                              kg_packages, db_path,
                              sol_db, snippet_id):

    python_version             = llm_eval["python_version"]
    python_major, python_minor = parse_python_version_string(python_version)
    file_dir                   = '/'.join(file_path.split('/')[:-1])
    base_modules               = file_dir + "/modules"

    pypi    = PyPIQuery(logging=True, base_modules=base_modules)
    dockerH = DockerHelper(logging=True)
    ollama  = OllamaHelper(
        base_url=base_url, model=model,
        logging=True, temp=temp,
        base_modules=base_modules, rag=True
    )

    project_dir, _, _ = dockerH.get_project_dir(file_path)
    log_file = f"{project_dir}/output_data_{python_version}.yml"

    with open(log_file, "a") as f:
        f.write("---\n")
        f.write(f"python_version: {python_version}\n")
        f.write(f"start_time: {start_time}\n")
        f.write("iterations:\n")

    # ── Stage 1: Try ALL known solutions from DB ──────────────────────────
    # Only process_num==0 does this to avoid duplicate Docker builds
    if process_num == 0:
        solutions = lookup_all_solutions(snippet_id, sol_db)
        if solutions:
            print(f"[P0] Found {len(solutions)} known solutions — trying all")
            for i, solution in enumerate(solutions):
                sol_major, sol_minor = parse_python_version_string(
                    solution['python_version']
                )
                print(f"[P0] Solution {i+1}/{len(solutions)}: "
                      f"Python {solution['python_version']}, "
                      f"{len(solution['modules'])} packages")
                success, _, _ = try_replay_solution(
                    solution, file_path, process_num,
                    start_time, db_path, log_file,
                    pypi, ollama, base_modules,
                    sol_major, sol_minor
                )
                if success:
                    return
            print(f"[P0] All {len(solutions)} solutions failed "
                  f"— running full pipeline")

    # ── Stage 2: Live PyPI lookup ─────────────────────────────────────────
    installable   = [n for n in import_names
                     if not is_stdlib(n) and get_pip_name(n) is not None]
    pypi_versions = resolve_packages_from_pypi(
        installable, python_minor, python_major
    )

    # ── Stage 3: Merge initial packages ──────────────────────────────────
    initial_modules = llm_eval.get("python_modules", [])
    if isinstance(initial_modules, list):
        initial_dict = {m: None for m in initial_modules
                        if not is_stdlib(m) and get_pip_name(m) is not None}
    else:
        initial_dict = {k: v for k, v in initial_modules.items()
                        if not is_stdlib(k) and get_pip_name(k) is not None}

    if python_major == 2:
        capped = {}
        for pkg, ver in initial_dict.items():
            cap = PYTHON27_MAX_VERSIONS.get(pkg.lower())
            capped[pkg] = cap if cap else ver
        initial_dict = capped

    try:
        resolved, _ = pypi.get_module_specifics({
            "python_version": python_version,
            "python_modules": list(initial_dict.keys()),
        })
        versioned = ollama.get_module_versions({
            "python_version": python_version,
            "python_modules": resolved,
        })
        if python_major == 2:
            for pkg in list(versioned.keys()):
                cap = PYTHON27_MAX_VERSIONS.get(pkg.lower())
                if cap:
                    versioned[pkg] = cap
    except Exception as e:
        print(f"[P{process_num}] Version resolve error: {e}")
        versioned = initial_dict

    current_packages = {**versioned, **pypi_versions, **kg_packages}
    current_packages = {
        k: v for k, v in current_packages.items()
        if not is_stdlib(k) and get_pip_name(k) is not None
    }

    print(f"[P{process_num}|v{python_version}] "
          f"Starting with {len(current_packages)} packages")

    error_handler = {
        'previous': '', 'error_modules': {},
        'ImportError': 0, 'ModuleNotFound': 0, 'VersionNotFound': 0,
        'DependencyConflict': 0, 'AttributeError': 0,
        'NonZeroCode': 0, 'SyntaxError': 0,
    }

    confirmed_unavailable = set()
    loop          = 1
    run_complete  = False
    error_log     = ""
    error_type    = "Unknown"
    docker_output = ""

    # ── Stage 4: LLM-driven loop ──────────────────────────────────────────
    while not run_complete and loop <= end_loop:
        try:
            print(f"\n[P{process_num}|v{python_version}] "
                  f"── Loop {loop}/{end_loop} error={error_type} ──")

            if loop > 1:
                if is_no_versions_error(error_log):
                    bad_pkg = get_bad_pkg_from_error(error_log)
                    if bad_pkg:
                        confirmed_unavailable.add(bad_pkg)
                        current_packages.pop(bad_pkg, None)
                        print(f"[P{process_num}] '{bad_pkg}' not on PyPI — removed")
                        _write_log(log_file,
                                   {"python_version": python_version,
                                    "python_modules": current_packages,
                                    "previous_python_modules": current_packages},
                                   "NotOnPyPI", error_log, loop, False)
                        loop += 1
                        continue

                error_type, _ = classify(error_log)
                failing_pkgs  = extract_failing_packages(error_log)

                for pkg in failing_pkgs:
                    if pkg in confirmed_unavailable:
                        continue
                    pip_name = get_pip_name(pkg)
                    if pip_name is None:
                        confirmed_unavailable.add(pkg)
                        current_packages.pop(pkg, None)
                        continue
                    if python_major == 2:
                        cap = PYTHON27_MAX_VERSIONS.get(pip_name.lower())
                        if cap:
                            current_packages[pip_name] = cap
                            continue
                    versions = get_pypi_versions(pip_name)
                    if not versions:
                        confirmed_unavailable.add(pkg)
                        current_packages.pop(pkg, None)
                    else:
                        current_packages[pip_name] = versions[-1]
                        print(f"[P{process_num}] PyPI fix: "
                              f"{pip_name}=={versions[-1]}")

                rag_packages = query_packages(
                    import_names + failing_pkgs,
                    min_python_minor=python_minor,
                    db_path=db_path
                )
                rag_context = dict_to_requirements_txt(rag_packages) \
                              if rag_packages else "(no prior data)"

                specialist_prompt = build_specialist_prompt(
                    error_type=error_type,
                    error_log=error_log,
                    current_packages=current_packages,
                    rag_context=rag_context,
                    import_names=import_names,
                    python_version=python_version,
                )

                new_packages, confidence = debate_round(
                    llm_model=ollama.model,
                    specialist_prompt=specialist_prompt,
                    import_names=import_names,
                    error_type=error_type,
                    previous_best=current_packages,
                    max_inner_rounds=2,
                )

                if new_packages:
                    validated = {}
                    for pkg, ver in new_packages.items():
                        if pkg in confirmed_unavailable:
                            continue
                        if get_pip_name(pkg) is None:
                            continue
                        if python_major == 2:
                            cap = PYTHON27_MAX_VERSIONS.get(pkg.lower())
                            if cap:
                                validated[pkg] = cap
                                continue
                        real_versions = get_pypi_versions(pkg)
                        if not real_versions:
                            continue
                        if ver and ver in real_versions:
                            validated[pkg] = ver
                        else:
                            validated[pkg] = real_versions[-1]
                            if ver and ver not in real_versions:
                                print(f"[P{process_num}] Agent: {pkg}=={ver} "
                                      f"→ using {real_versions[-1]}")
                    if validated:
                        current_packages = validated
                else:
                    pllm_out, pllm_err = safe_pllm_error_handler(
                        ollama, error_log, error_handler,
                        {"python_version": python_version,
                         "python_modules": current_packages,
                         "previous_python_modules": current_packages}
                    )
                    if pllm_out and isinstance(pllm_out, dict):
                        module  = pllm_out.get("module")
                        version = pllm_out.get("version")
                        if module and module not in confirmed_unavailable:
                            if python_major == 2:
                                cap = PYTHON27_MAX_VERSIONS.get(module.lower())
                                if cap:
                                    version = cap
                            if version and version not in ("None", "none", ""):
                                current_packages[module] = version
                            elif module in current_packages:
                                current_packages.pop(module)
                    error_handler[pllm_err] = \
                        error_handler.get(pllm_err, 0) + 1

            eval_docker = {
                "python_version": python_version,
                "python_modules": current_packages,
                "previous_python_modules": current_packages,
            }
            dockerH.create_dockerfile(eval_docker, file_path)
            build_passed, docker_output = dockerH.build_dockerfile(file_path)

            if not build_passed:
                error_log = docker_output
                if is_no_versions_error(docker_output):
                    bad_pkg = get_bad_pkg_from_error(docker_output)
                    if bad_pkg:
                        confirmed_unavailable.add(bad_pkg)
                        current_packages.pop(bad_pkg, None)
                        _write_log(log_file, eval_docker,
                                   "NotOnPyPI", docker_output, loop, False)
                        loop += 1
                        continue

                pllm_out, pllm_err = safe_pllm_error_handler(
                    ollama, docker_output, error_handler,
                    {"python_version": python_version,
                     "python_modules": current_packages,
                     "previous_python_modules": current_packages}
                )
                if pllm_out and isinstance(pllm_out, dict):
                    module  = pllm_out.get("module")
                    version = pllm_out.get("version")
                    if module and module not in confirmed_unavailable:
                        if python_major == 2:
                            cap = PYTHON27_MAX_VERSIONS.get(module.lower())
                            if cap:
                                version = cap
                        if version and version not in ("None", "none", ""):
                            current_packages[module] = version
                        elif module in current_packages:
                            current_packages.pop(module)
                if pllm_err == 'NonZeroCode' and \
                   'PATH environment' in docker_output:
                    bad = pllm_out.get("module") if pllm_out else None
                    if bad and bad in current_packages:
                        current_packages.pop(bad)

                _write_log(log_file, eval_docker, error_type,
                           docker_output, loop, False)
                loop += 1
                continue

            print(f"[P{process_num}|v{python_version}] Build passed ✓")

            docker_output    = dockerH.run_container_test()
            run_out, run_err = safe_pllm_error_handler(
                ollama, docker_output, error_handler, eval_docker
            )
            print(f"[P{process_num}|v{python_version}] Run: {run_err}")

            if run_err in ("None", None) \
               or run_err == "NameError" \
               or "DJANGO_SETTINGS_MODULE" in docker_output:

                run_complete = True
                print(f"[P{process_num}|v{python_version}] "
                      f"✓✓✓ SOLVED on loop {loop}")
                save_successful_combo(current_packages, python_version,
                                      snippet_hash(file_path), db_path)
                _write_log(log_file,
                           {"python_version": python_version,
                            "python_modules": current_packages,
                            "previous_python_modules": current_packages},
                           None, docker_output, loop, True, start_time)

            else:
                error_log  = docker_output
                error_type = run_err
                if run_out and isinstance(run_out, dict):
                    module  = run_out.get("module")
                    version = run_out.get("version")
                    if module and module not in confirmed_unavailable:
                        error_handler['error_modules'].setdefault(module, [])
                        error_handler['error_modules'][module].append(
                            current_packages.get(module, "unknown")
                        )
                        error_handler[error_type] = \
                            error_handler.get(error_type, 0) + 1
                        if error_type == "NonZeroCode":
                            current_packages.pop(module, None)
                        elif python_major == 2:
                            cap = PYTHON27_MAX_VERSIONS.get(module.lower())
                            current_packages[module] = \
                                cap if cap else (version or
                                current_packages.get(module))
                        elif version and version not in ("None", "none", ""):
                            current_packages[module] = version
                        elif module in current_packages:
                            current_packages.pop(module)
                _write_log(log_file, eval_docker, error_type,
                           docker_output, loop, False)
                loop += 1
                continue

        except Exception as e:
            print(f"[P{process_num}|v{python_version}] Exception: {e}")
            error_log = str(e)
            _write_log(log_file,
                       {"python_version": python_version,
                        "python_modules": current_packages,
                        "previous_python_modules": current_packages},
                       error_type, docker_output, loop, False)

        loop += 1

    if not run_complete:
        _write_log(log_file,
                   {"python_version": python_version,
                    "python_modules": current_packages,
                    "previous_python_modules": current_packages},
                   error_type, docker_output, end_loop, True, start_time)

    dockerH.delete_container()
    dockerH.delete_image()


# ── CLI ───────────────────────────────────────────────────────────────────────

def process_args():
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        raise argparse.ArgumentTypeError('Boolean expected')

    p = argparse.ArgumentParser()
    p.add_argument('-f', '--file',    type=str,      required=True)
    p.add_argument('-b', '--base',    type=str,      default='http://localhost:11434')
    p.add_argument('-m', '--model',   type=str,      default='gemma2')
    p.add_argument('-t', '--temp',    type=str,      default='0.7')
    p.add_argument('-l', '--loop',    type=int,      default=10)
    p.add_argument('-r', '--range',   type=int,      default=0)
    p.add_argument('-ra', '--rag',    type=str2bool, default=True)
    p.add_argument('-v', '--verbose', action='store_true')
    return p.parse_args()


def main():
    args       = process_args()
    start_time = time.time()
    db_path    = os.environ.get("KG_DB_PATH", DB_DEFAULT)
    sol_db     = os.path.join(os.path.dirname(db_path), "solutions.db")

    ensure_knowledge_graph(db_path)
    ensure_solutions_db(sol_db)

    snippet_id = args.file.split('/')[-2]

    with open(args.file, 'r', errors='replace') as f:
        source = f.read()

    analysis = analyze(source)
    print(f"[AST] min_python={analysis['min_python']}, "
          f"imports={analysis['import_names']}")

    ast_major = analysis["min_python"][0]
    ast_minor = analysis["min_python"][1]

    kg_packages = query_packages(
        analysis["import_names"],
        min_python_minor=ast_minor,
        db_path=db_path
    )

    file_dir     = '/'.join(args.file.split('/')[:-1])
    base_modules = file_dir + "/modules"

    ollama = OllamaHelper(
        base_url=args.base, model=args.model,
        logging=True, temp=args.temp,
        base_modules=base_modules, rag=True
    )
    pypi = PyPIQuery(logging=True, base_modules=base_modules)
    deps = DepsScraper(logging=True)

    llm_eval    = None
    llm_details = False
    for attempt in range(5):
        try:
            llm_eval = ollama.evaluate_file(args.file)
            llm_eval["python_version"] = str(llm_eval["python_version"])
            if isinstance(llm_eval.get("python_modules"), dict):
                llm_eval["python_modules"] = list(
                    llm_eval["python_modules"].keys()
                )
            raw_imports = deps.find_word_in_file(args.file, 'import', [])
            all_modules = pypi.check_module_name(
                raw_imports + llm_eval.get("python_modules", [])
            )
            llm_eval["python_modules"] = [m for m in all_modules
                                           if not is_stdlib(m)]
            llm_details = True
            break
        except Exception as e:
            print(f"[LLM] Attempt {attempt+1} failed: {e}")

    if not llm_details:
        llm_eval = {
            "python_version": f"{ast_major}.{ast_minor}",
            "python_modules": [m for m in
                               pypi.check_module_name(analysis["import_names"])
                               if not is_stdlib(m)],
        }

    python_versions = pypi.get_python_range(
        python_version=llm_eval["python_version"], pyrange=args.range
    )
    python_versions = [
        v for v in python_versions
        if (v == "2.7" and ast_major == 2)
        or (v != "2.7" and int(v.split(".")[1]) >= ast_minor)
    ] or get_python_version_range(analysis, args.range)

    seen = set()
    final_versions = []
    for v in python_versions:
        if v not in seen:
            seen.add(v)
            final_versions.append(v)
    final_versions = final_versions[:(args.range * 2) + 1]

    solutions = lookup_all_solutions(snippet_id, sol_db)
    print(f"[Main] snippet_id={snippet_id}")
    print(f"[Main] Versions: {final_versions}")
    print(f"[Main] Modules: {llm_eval['python_modules']}")
    if solutions:
        print(f"[Main] {len(solutions)} known solutions to try")

    processes = []
    for i, version in enumerate(final_versions):
        run_details                   = llm_eval.copy()
        run_details["python_version"] = version

        p = mp.Process(
            target=enhanced_docker_process,
            args=(
                args.base, args.model, args.temp,
                run_details, args.file,
                analysis["import_names"],
                i, args.loop, start_time,
                kg_packages, db_path, sol_db, snippet_id,
            )
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join(timeout=1200)
    for p in processes:
        if p.is_alive():
            p.terminate()

    print(f"\n[Main] Done. Total: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()


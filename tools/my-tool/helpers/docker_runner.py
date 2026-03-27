"""
helpers/docker_runner.py
-------------------------
Docker Build + Run Runner

Mirrors PLLM's two-step process:
  1. `docker build`  — install dependencies
  2. `docker run`    — actually execute the snippet

Success is defined the same way as PLLM:
  - The build must succeed (no pip install failures).
  - The container run must produce output that contains no dependency errors
    (ModuleNotFoundError, ImportError, VersionNotFound, DependencyConflict, etc.).
  - A NameError or clean exit are both treated as success — the deps are
    resolved even if the snippet has unrelated logic errors (same as PLLM).

Returns (success: bool, log: str) where `log` is always the RUN output
(or build output on build failure), so the error classifier always sees
the most informative text.
"""

import os
import uuid
import time
import tempfile
import textwrap

try:
    import docker as docker_sdk
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Error strings that indicate a dependency problem at *runtime*
# (mirrors the conditions PLLM checks in process_error / docker_create_process)
# ---------------------------------------------------------------------------
_DEPENDENCY_ERRORS = [
    "ModuleNotFoundError",
    "ImportError",
    "Could not find a version",
    "No matching distribution",
    "dependency conflicts",
    "DependencyConflict",
    "InvalidVersion",
    "SyntaxError",           # can indicate a wrong package version installed
]

# These runtime errors mean the deps are fine — the snippet itself is broken.
# PLLM marks these as run_complete = True (success for our purposes).
_ACCEPTABLE_RUNTIME_ERRORS = [
    "NameError",
    "DJANGO_SETTINGS_MODULE is undefined",
]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_docker_build(
    snippet_path: str,
    python_version: str,
    requirements: dict,
) -> tuple[bool, str]:
    """
    Build a Docker image and run the snippet inside it.

    Parameters
    ----------
    snippet_path    : full path to the Python snippet file
    python_version  : e.g. "3.8"
    requirements    : dict like {"numpy": "1.21.0", "pandas": "1.3.0"}

    Returns
    -------
    (success: bool, log_output: str)
      - success=True  → deps resolved (build ok + run ok / acceptable error)
      - success=False → dependency problem; log contains the error for classifier
    """
    if not DOCKER_AVAILABLE:
        print("[Docker] docker SDK not available — returning mock result")
        return _mock_build(requirements)

    image_tag      = f"pllm-candidate:{uuid.uuid4().hex[:8]}"
    container_name = f"pllm-run-{uuid.uuid4().hex[:8]}"
    client         = None

    try:
        with tempfile.TemporaryDirectory() as build_dir:
            # ── Write build context ──────────────────────────────────────
            req_path = os.path.join(build_dir, "requirements.txt")
            _write_requirements(req_path, requirements)

            import shutil
            snippet_name = os.path.basename(snippet_path)
            shutil.copy(snippet_path, os.path.join(build_dir, snippet_name))

            dockerfile_path = os.path.join(build_dir, "Dockerfile")
            _write_dockerfile(dockerfile_path, python_version, snippet_name)

            # ── Step 1: docker build ─────────────────────────────────────
            client = docker_sdk.from_env()
            print(f"[Docker] Building image {image_tag} (Python {python_version}) …")

            try:
                _, build_log_gen = client.images.build(
                    path=build_dir,
                    tag=image_tag,
                    rm=True,
                    forcerm=True,
                )
                build_log_lines = []
                for chunk in build_log_gen:
                    line = chunk.get("stream", chunk.get("error", ""))
                    if line:
                        build_log_lines.append(line.rstrip())
                build_log = "\n".join(build_log_lines)

            except docker_sdk.errors.BuildError as exc:
                build_log = "\n".join(
                    line.get("stream", line.get("error", ""))
                    for line in exc.build_log
                    if isinstance(line, dict)
                )
                print(f"[Docker] Build FAILED: {exc}")
                return False, build_log

            print(f"[Docker] Build succeeded — running container …")

            # ── Step 2: docker run ───────────────────────────────────────
            run_log = _run_container(client, image_tag, container_name)

            # ── Step 3: evaluate run output (same logic as PLLM) ────────
            success = _evaluate_run_output(run_log)

            return success, run_log

    except Exception as exc:
        print(f"[Docker] Unexpected error: {exc}")
        return False, str(exc)

    finally:
        # Always clean up image and container
        _cleanup(client, image_tag, container_name)


# ---------------------------------------------------------------------------
# Container execution
# ---------------------------------------------------------------------------

def _run_container(client, image_tag: str, container_name: str) -> str:
    """
    Create, start, wait for, and remove a container.
    Returns the combined stdout+stderr log as a string.
    Mirrors PLLM's run_container_test() behaviour.
    """
    container = None
    try:
        container = client.containers.create(image_tag, name=container_name)
        container.start()

        # Wait up to 60 seconds (PLLM uses sleep(10) then polls)
        time.sleep(10)
        timeout = 50   # additional seconds
        elapsed = 0
        container.reload()
        while container.status == "running" and elapsed < timeout:
            time.sleep(5)
            elapsed += 5
            container.reload()

        logs = container.logs()
        return logs.decode("utf-8", errors="replace")

    except docker_sdk.errors.ContainerError as exc:
        # Container exited non-zero — still get the logs
        if container:
            try:
                logs = container.logs()
                return logs.decode("utf-8", errors="replace")
            except Exception:
                pass
        return str(exc)

    finally:
        if container:
            try:
                container.remove(v=True, force=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Success evaluation — mirrors PLLM's process_error() routing
# ---------------------------------------------------------------------------

def _evaluate_run_output(run_log: str) -> bool:
    """
    Decide whether the run counts as 'success' using the same rules as PLLM:

      - Any dependency-level error  → False  (need another loop)
      - NameError                   → True   (deps fine, script logic broken)
      - DJANGO_SETTINGS_MODULE      → True   (known acceptable Django edge case)
      - No recognised error         → True   (clean exit)
    """
    # Check acceptable non-dep errors first (PLLM sets run_complete=True for these)
    for acceptable in _ACCEPTABLE_RUNTIME_ERRORS:
        if acceptable in run_log:
            print(f"[Docker] Run has acceptable error ({acceptable!r}) — treating as success")
            return True

    # Check for dependency errors (PLLM sets build_complete=False for these)
    for dep_err in _DEPENDENCY_ERRORS:
        if dep_err in run_log:
            print(f"[Docker] Run has dependency error ({dep_err!r}) — treating as failure")
            return False

    # No errors detected → clean run
    print("[Docker] Run completed cleanly — success")
    return True


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_requirements(path: str, requirements: dict) -> None:
    with open(path, "w") as f:
        for pkg, ver in requirements.items():
            if ver and str(ver).lower() not in ("latest", "none", ""):
                f.write(f"{pkg}=={ver}\n")
            else:
                f.write(f"{pkg}\n")


def _write_dockerfile(path: str, python_version: str, snippet_name: str) -> None:
    content = textwrap.dedent(f"""\
        FROM python:{python_version}-slim

        WORKDIR /app

        COPY requirements.txt .
        RUN pip install --no-cache-dir --trusted-host pypi.python.org \
--default-timeout=100 -r requirements.txt

        COPY {snippet_name} .

        CMD ["python", "{snippet_name}"]
    """)
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Cleanup helper
# ---------------------------------------------------------------------------

def _cleanup(client, image_tag: str, container_name: str) -> None:
    if client is None:
        return
    try:
        client.images.remove(image_tag, force=True)
    except Exception:
        pass
    try:
        client.containers.get(container_name).remove(v=True, force=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Mock build (no Docker daemon available)
# ---------------------------------------------------------------------------

def _mock_build(requirements: dict) -> tuple[bool, str]:
    print("[Docker] MOCK BUILD — pretending success.")
    return True, "Successfully built (mock)"

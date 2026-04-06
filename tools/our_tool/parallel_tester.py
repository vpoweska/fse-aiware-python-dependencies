import subprocess, concurrent.futures, threading

def test_spec_in_docker(python_version: str, requirements: str, snippet_path: str,
                        timeout: int = 120) -> dict:
    """Spins up an isolated Docker container and returns success/failure + logs."""
    dockerfile_content = f"""
FROM python:{python_version}-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 2>&1
COPY snippet.py .
CMD ["python", "snippet.py"]
"""
    # Write temp files, run docker build, capture output
    # ... (same Docker logic as PLLM but parameterized)
    result = subprocess.run(
        ["docker", "build", "--no-cache", "-t", f"test-{python_version}", "."],
        capture_output=True, text=True, timeout=timeout
    )
    return {
        "python_version": python_version,
        "success": result.returncode == 0,
        "log": result.stdout + result.stderr
    }

def test_parallel(requirements: str, snippet_path: str, 
                  python_versions: list[str]) -> list[dict]:
    """Test multiple Python versions concurrently."""
    # Limit to 3 concurrent to stay within VRAM/CPU budget
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(test_spec_in_docker, v, requirements, snippet_path): v
            for v in python_versions
        }
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=300):
            results.append(future.result())
    return results

def smart_version_range(min_python: tuple, signals: list) -> list[str]:
    """
    Given AST analysis results, return only the Python versions worth testing.
    e.g. if min_python=(3,8) and 'walrus_operator' detected, skip 3.6, 3.7.
    """
    all_versions = ["3.8", "3.9", "3.10", "3.11", "3.12"]
    min_minor = min_python[1]
    return [v for v in all_versions if int(v.split(".")[1]) >= min_minor]

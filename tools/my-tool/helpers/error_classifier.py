"""
helpers/error_classifier.py

Structured Error Classifier: Improvement #2

Instead of dumping raw Docker build logs into the LLM, this module:
  1. Reads the error text and decides WHAT TYPE of problem occurred.
  2. Pulls out the key facts (which package, which version, which constraint).
  3. Returns a tidy dict so the LLM gets a focused, structured prompt later.

Error categories we detect:
  - version_conflict          e.x. "package A requires X>=2 but B requires X<1"
  - no_matching_distribution  e.x. "No matching distribution found for numpy==99"
  - missing_system_dep        e.x. "fatal error: libssl not found"
  - python_version_mismatch   e.x. "Requires-Python >=3.9, we have 3.7"
  - module_not_found          e.x. "ModuleNotFoundError: No module named 'foo'"
  - unknown                   fallback when nothing matches
"""

import re
from typing import Optional

def classify_error(error_log: str) -> dict:
    """
    Main entry point.

    Takes raw Docker / pip error text and returns a structured dict:
    {
        "error_type":  str,              # one of the categories above
        "package":     str | None,       # the offending package name, if found
        "version":     str | None,       # the version that caused the problem
        "constraint":  str | None,       # the constraint string, e.g. ">=2.0"
        "raw_snippet": str,              # a short excerpt of the original log
    }
    """
    # Try each detector in order; return the first match.
    for detector in [
        _detect_version_conflict,
        _detect_no_matching_distribution,
        _detect_missing_system_dep,
        _detect_python_version_mismatch,
        _detect_module_not_found,
    ]:
        result = detector(error_log)
        if result is not None:
            result["raw_snippet"] = _trim_log(error_log)
            return result

    # Nothing matched — return an 'unknown' classification
    return {
        "error_type":  "unknown",
        "package":     None,
        "version":     None,
        "constraint":  None,
        "raw_snippet": _trim_log(error_log),
    }


def _detect_version_conflict(log: str) -> Optional[dict]:
    """
    Catches pip's dependency resolver conflicts.
    Example line: "ERROR: pip's dependency resolver does not currently take
    into account all packages ... numpy 1.21 requires scipy>=1.0"
    """
    # Look for the typical pip conflict phrase
    if "conflict" in log.lower() or "incompatible" in log.lower():
        # Try to find a package name near the conflict mention
        match = re.search(r"(\w[\w\-\.]+)\s+(\d[\d\.]+)\s+(?:requires|needs)\s+([\w\-\.]+(?:[><=!]+[\d\.]+)?)", log)
        if match:
            return {
                "error_type": "version_conflict",
                "package":    match.group(1),
                "version":    match.group(2),
                "constraint": match.group(3),
            }
        # Conflict found but we couldn't parse details
        return {"error_type": "version_conflict", "package": None, "version": None, "constraint": None}
    return None


def _detect_no_matching_distribution(log: str) -> Optional[dict]:
    """
    Catches pip's "no matching distribution" errors.
    Example: "ERROR: Could not find a version that satisfies the requirement numpy==99.0"
    """
    match = re.search(
        r"No matching distribution found for ([\w\-\.]+)==?([\d\.]*)",
        log, re.IGNORECASE
    )
    if match:
        return {
            "error_type": "no_matching_distribution",
            "package":    match.group(1),
            "version":    match.group(2) or None,
            "constraint": None,
        }

    match = re.search(
        r"Could not find a version that satisfies the requirement ([\w\-\.]+)((?:[><=!]+[\d\.]+)*)",
        log, re.IGNORECASE
    )
    if match:
        return {
            "error_type": "no_matching_distribution",
            "package":    match.group(1),
            "version":    None,
            "constraint": match.group(2) or None,
        }
    return None


def _detect_missing_system_dep(log: str) -> Optional[dict]:
    """
    Catches OS-level build failures (missing headers, libraries, compilers).
    Example: "fatal error: Python.h: No such file or directory"
    """
    system_keywords = ["fatal error:", "no such file", "libssl", "libffi", "gcc", "g++", "cannot find"]
    if any(kw in log.lower() for kw in system_keywords):
        # Try to find the missing lib name
        match = re.search(r"fatal error:\s+([\w\.\/]+)", log, re.IGNORECASE)
        package = match.group(1) if match else None
        return {
            "error_type": "missing_system_dep",
            "package":    package,
            "version":    None,
            "constraint": None,
        }
    return None


def _detect_python_version_mismatch(log: str) -> Optional[dict]:
    """
    Catches when a package requires a Python version different from the one
    used in the Docker image.
    Example: "Package requires Python >=3.9 but we have 3.7"
    """
    match = re.search(
        r"Requires-Python\s*((?:[><=!]+[\d\.]+)+).*?(\d\.\d+)",
        log, re.IGNORECASE | re.DOTALL
    )
    if match:
        return {
            "error_type": "python_version_mismatch",
            "package":    None,
            "version":    match.group(2),   # the Python version we're using
            "constraint": match.group(1),   # e.g. ">=3.9"
        }

    if "requires python" in log.lower() or "python version" in log.lower():
        return {
            "error_type": "python_version_mismatch",
            "package":    None,
            "version":    None,
            "constraint": None,
        }
    return None


def _detect_module_not_found(log: str) -> Optional[dict]:
    """
    Catches runtime import errors that surface during the test run step
    (after pip install succeeds but the script itself crashes).
    Example: "ModuleNotFoundError: No module named 'sklearn'"
    """
    match = re.search(r"ModuleNotFoundError: No module named '([\w\.]+)'", log)
    if match:
        return {
            "error_type": "module_not_found",
            "package":    match.group(1),
            "version":    None,
            "constraint": None,
        }
    return None

def _trim_log(log: str, max_chars: int = 800) -> str:
    """Return the last `max_chars` characters of the log (most informative part)."""
    return log[-max_chars:].strip()

if __name__ == "__main__":
    sample_logs = [
        "ERROR: No matching distribution found for numpy==99.0",
        "fatal error: libssl.h: No such file or directory",
        "ModuleNotFoundError: No module named 'sklearn'",
        "conflict: numpy 1.21 requires scipy>=1.0 but you have scipy 0.9",
        "Requires-Python >=3.9 but current Python is 3.7",
        "Something totally unexpected happened.",
    ]
    for log in sample_logs:
        print(classify_error(log))

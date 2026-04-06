import re

ERROR_PATTERNS = {
    "version_conflict": [
        r"ResolutionImpossible",
        r"cannot install .* and .* because these package versions have conflicting dependencies",
        r"incompatible with.*requires",
    ],
    "no_matching_distribution": [
        r"No matching distribution found for",
        r"Could not find a version that satisfies",
    ],
    "missing_system_dep": [
        r"Microsoft Visual C\+\+",
        r"error: command.*gcc.*failed",
        r"fatal error:.*\.h: No such file",
    ],
    "python_version_mismatch": [
        r"requires Python '([><=!]+[\d.]+)'",
        r"Requires-Python",
    ],
    "module_not_found": [
        r"ModuleNotFoundError: No module named",
        r"ImportError: cannot import name",
    ],
}

SPECIALIST_PROMPTS = {
    "version_conflict": """
You are resolving a VERSION CONFLICT between Python packages.
The conflict is: {error_summary}
Known working packages from similar projects: {rag_context}

Strategy: Find the 'pivot version' — the oldest version of the conflicting package 
that satisfies all other constraints. Start with the lower bound, not the latest.
Output ONLY a requirements.txt with pinned versions. No explanation.
""",
    "no_matching_distribution": """
You are resolving a MISSING DISTRIBUTION error.
Package: {package_name}
This package may have been renamed, deprecated, or requires a specific Python version.
Known alternatives from similar projects: {rag_context}

Check: 1) Was this package renamed? 2) Is there a backport version? 3) Is it bundled in stdlib?
Output ONLY a requirements.txt. No explanation.
""",
    # ... etc
}

def classify(error_log: str) -> tuple[str, dict]:
    """Returns (error_type, extracted_info)"""
    for error_type, patterns in ERROR_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, error_log, re.IGNORECASE)
            if m:
                return error_type, {"match": m.group(0), "groups": m.groups()}
    return "unknown", {}

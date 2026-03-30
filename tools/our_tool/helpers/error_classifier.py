import re

ERROR_PATTERNS = [
    ("VersionConflict", [
        r"ResolutionImpossible",
        r"Cannot install .+ and .+ because these package versions have conflicting",
        r"incompatible with .+requires",
        r"conflict",
    ]),
    ("NoMatchingDistribution", [
        r"No matching distribution found for",
        r"Could not find a version that satisfies the requirement",
        r"ERROR: Could not find a version",
    ]),
    ("PythonVersionMismatch", [
        r"requires Python ['\"]([><=!]+[\d.]+)['\"]",
        r"Requires-Python",
        r"python_requires",
    ]),
    ("ImportError", [
        r"ModuleNotFoundError: No module named",
        r"ImportError: cannot import name",
        r"ImportError: No module named",
    ]),
    ("AttributeError", [
        r"AttributeError:",
    ]),
    ("DependencyConflict", [
        r"ContextualVersionConflict",
        r"pkg_resources.ContextualVersionConflict",
        r"distutils.errors",
    ]),
    ("SyntaxError", [
        r"SyntaxError:",
        r"invalid syntax",
    ]),
    ("NonZeroCode", [
        r"returned a non-zero exit code",
        r"exit code: [1-9]",
    ]),
    ("SystemDependency", [
        r"fatal error: .+\.h: No such file",
        r"error: command.*gcc.*failed",
        r"Microsoft Visual C\+\+",
    ]),
]

SPECIALIST_PROMPTS = {
    "VersionConflict": """\
You are resolving a Python dependency VERSION CONFLICT.

Error:
{error_summary}

Current requirements being attempted:
{current_requirements}

Known working package versions from similar projects:
{rag_context}

Strategy:
- Find the pivot package causing the conflict.
- Reduce its version constraint until all other packages are satisfied.
- Pin every version exactly with ==. Do NOT use >= or ~=.

Respond with ONLY a valid requirements.txt. No explanation, no markdown, no comments.
""",

    "NoMatchingDistribution": """\
You are resolving a Python MISSING DISTRIBUTION error.

The package {package_name} does not have version {bad_version}.
The ONLY versions that actually exist on PyPI are:
{available_versions}

You MUST pick a version from that list above. Do not invent versions.
Pick the most recent version from the list that is compatible with Python {python_version}.

All packages needed:
{current_requirements}

Known working versions from similar projects:
{rag_context}

Respond with ONLY a valid requirements.txt using versions from the list above.
No explanation, no markdown, no comments.
""",

    "PythonVersionMismatch": """\
You are resolving a Python VERSION MISMATCH.

Error:
{error_summary}

The current Python version being tested is {python_version}.

Known working package versions from similar projects:
{rag_context}

Find the exact package version compatible with Python {python_version}.
If no version is compatible, choose an older version that predates the restriction.

Respond with ONLY a valid requirements.txt. No explanation, no markdown, no comments.
""",

    "ImportError": """\
You are resolving a Python IMPORT ERROR after a successful pip install.

Error:
{error_summary}

Current requirements:
{current_requirements}

This usually means:
1. The package was installed but the import name differs from the pip name.
2. A transitive dependency is missing or has the wrong version.
3. Installation order matters - put dependencies before dependents.

Known working versions from similar projects:
{rag_context}

Respond with ONLY a valid requirements.txt. No explanation, no markdown, no comments.
""",

    "SystemDependency": """\
You are resolving a package that requires a SYSTEM-LEVEL DEPENDENCY.

Error:
{error_summary}

Strategy:
1. Use a pre-built wheel version instead of compiling from source.
2. Check if there is a pure-Python fallback package.
3. Downgrade to a version that ships pre-built wheels for linux/amd64.

Known working versions:
{rag_context}

Respond with ONLY a valid requirements.txt. No explanation, no markdown, no comments.
""",

    "Unknown": """\
You are resolving a Python dependency conflict.

Error from the last Docker build attempt:
{error_summary}

Current requirements being attempted:
{current_requirements}

Known working package versions from similar projects:
{rag_context}

Provide corrected package versions that resolve the error.
Pin every version exactly with ==. Do NOT use >=, ~=, or leave any version unpinned.

Respond with ONLY a valid requirements.txt. No explanation, no markdown, no comments.
""",
}


def classify(error_log: str) -> tuple:
    for error_type, patterns in ERROR_PATTERNS:
        for pat in patterns:
            m = re.search(pat, error_log, re.IGNORECASE)
            if m:
                return error_type, {"match": m.group(0), "groups": list(m.groups())}
    return "Unknown", {}


def extract_failing_packages(error_log: str) -> list:
    packages = []
    for pat in [
        r"No matching distribution found for ([^\s>=<!\n]+)",
        r"Cannot install ([^\s>=<!\n]+)",
        r"No module named ['\"]?([^'\">\s]+)['\"]?",
        r"ImportError.*['\"]([^'\"]+)['\"]",
    ]:
        for m in re.finditer(pat, error_log, re.IGNORECASE):
            pkg = m.group(1).split("[")[0].strip()
            if pkg:
                packages.append(pkg)
    return list(set(packages))


def extract_available_versions(error_log: str) -> dict:
    """
    Pull the available versions list directly out of the pip error message.
    e.g. 'from versions: 1.0, 1.1, 2.0' -> {'scrapy': {'available': [...], 'bad_version': '3.0.2'}}
    """
    result = {}
    pattern = re.compile(
        r"requirement\s+([A-Za-z0-9_\-\.]+)==([\w\.]+).*?from versions:\s*([\d\w\s,\.\-rc]+?)\)",
        re.IGNORECASE | re.DOTALL
    )
    for m in pattern.finditer(error_log):
        pkg     = m.group(1).lower().strip()
        bad_ver = m.group(2).strip()
        versions_raw = m.group(3)
        versions = [v.strip() for v in versions_raw.split(",") if v.strip()]
        result[pkg] = {"available": versions, "bad_version": bad_ver}
    return result


def build_specialist_prompt(error_type: str, error_log: str,
                             current_packages: dict, rag_context: str,
                             import_names: list, python_version: str) -> str:
    template = SPECIALIST_PROMPTS.get(error_type, SPECIALIST_PROMPTS["Unknown"])

    if current_packages:
        current_req = "\n".join(
            f"{k}=={v}" for k, v in current_packages.items()
            if v and str(v) not in ("None", "none", "")
        )
    else:
        current_req = "(none yet)"

    pkg_list = ", ".join(extract_failing_packages(error_log)) or "(see error)"

    # For NoMatchingDistribution — extract real available versions from error
    # and inject them so the LLM cannot hallucinate a non-existent version
    if error_type == "NoMatchingDistribution":
        version_info = extract_available_versions(error_log)
        if version_info:
            pkg_name = list(version_info.keys())[0]
            info     = version_info[pkg_name]
            available  = info["available"]
            bad_version = info["bad_version"]
            # Give the LLM the last 20 versions (most recent) to keep prompt short
            recent_versions = available[-20:] if len(available) > 20 else available
            return template.format(
                error_summary=error_log[:600],
                package_name=pkg_name,
                bad_version=bad_version,
                available_versions=", ".join(recent_versions),
                current_requirements=current_req,
                rag_context=rag_context,
                python_version=python_version,
                package_list=pkg_list,
            )

    return template.format(
        error_summary=error_log[:800],
        current_requirements=current_req,
        rag_context=rag_context,
        package_list=pkg_list,
        python_version=python_version,
    )


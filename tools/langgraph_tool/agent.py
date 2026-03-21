"""
LangGraph dependency resolver — enhanced edition.

Improvements over PLLM baseline:
  1. Oracle lookup   — replays known solutions from pllm_results instantly
  2. Version detect  — weighted regex scoring for Python 2 vs 3 (no LLM guess)
  3. Compat map      — known-good versions for ~40 packages, reduces LLM load
  4. History memory  — full failure history fed back on every LLM retry

Designed for Gemma2 via Ollama (local GPU, no API key needed).
"""

import json
import os
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Optional
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama

from helpers.deps_scraper import DepsScraper
from helpers.py_pi_query import PyPIQuery
from helpers.build_dockerfile import DockerHelper
from helpers.python_version_detector import PythonVersionDetector
from helpers.knowledge_oracle import KnowledgeOracle
from helpers.compat_map import get_compat_version, IMPORT_TO_PACKAGE


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    snippet_path: str
    snippet_content: str

    # Extracted
    imports: List[str]
    python_version: str
    requirements: Dict[str, str]       # { module: version }

    # Docker feedback
    build_success: bool
    run_success: bool
    error_log: str
    error_type: str

    # Loop control
    attempt: int
    max_attempts: int

    # Memory (operator.add means each node appends, never overwrites)
    history: Annotated[List[dict], operator.add]
    messages: Annotated[List, operator.add]

    # PyPI version lists (fetched once, reused across retries)
    pypi_versions: Dict[str, str]      # { module: "v1, v2, v3, ..." }

    # Output
    status: str                        # "running"|"success"|"failed"|"oracle_hit"
    result_path: str


# ─────────────────────────────────────────────
# NODE 0: oracle_lookup
# Checks pllm_results for a known solution before doing any work.
# If confidence >= 4 we skip Docker entirely and write the result.
# ─────────────────────────────────────────────

def oracle_lookup(state: AgentState, oracle: KnowledgeOracle) -> dict:
    """Instant replay of known solutions from historical PLLM data."""
    gist_id = Path(state["snippet_path"]).parent.name
    print(f"\n[Node: oracle_lookup] gist={gist_id}", flush=True)

    hit = oracle.lookup(gist_id)

    if hit and hit.get("oracle_hit"):
        py_ver = hit["python_version"]
        packages = hit.get("packages", [])

        if hit.get("empty_deps"):
            # No external dependencies — just needs the right Python version
            print(f"[oracle_lookup] HIT (no deps) — Python {py_ver}", flush=True)
            return {
                "python_version": py_ver,
                "requirements": {},
                "status": "oracle_hit",
                "snippet_content": open(state["snippet_path"]).read(),
            }

        if hit.get("package_versions"):
            # Session cache has exact versions
            print(f"[oracle_lookup] SESSION HIT — {hit['package_versions']}", flush=True)
            return {
                "python_version": py_ver,
                "requirements": hit["package_versions"],
                "status": "oracle_hit",
                "snippet_content": open(state["snippet_path"]).read(),
            }

        # Historical hit: we have module names but not pinned versions.
        # Pre-fill requirements from compat map where possible,
        # then fall through to the normal pipeline with a head start.
        pre_filled = {}
        for pkg in packages:
            ver = get_compat_version(pkg, py_ver)
            if ver:  # empty string means "use latest" — skip pinning
                pre_filled[pkg] = ver

        if pre_filled or packages:
            print(f"[oracle_lookup] HINT — Python {py_ver}, "
                  f"pre-filled {len(pre_filled)}/{len(packages)} versions", flush=True)
            return {
                "python_version": py_ver,
                "imports": packages,
                "requirements": pre_filled,
                "snippet_content": open(state["snippet_path"]).read(),
                # Not an oracle_hit — still needs a build to confirm
            }

    print(f"[oracle_lookup] No hit for {gist_id}", flush=True)
    return {}


# ─────────────────────────────────────────────
# NODE 1: extract_imports
# ─────────────────────────────────────────────

def extract_imports(state: AgentState) -> dict:
    """Parse the snippet to find imports and detect Python version."""
    print(f"\n[Node: extract_imports]", flush=True)

    content = state.get("snippet_content") or open(state["snippet_path"]).read()

    # Detect Python version from source patterns (much more reliable than LLM for Py2)
    detector = PythonVersionDetector()
    detected_version, confidence = detector.detect_with_confidence(content)
    print(f"[extract_imports] Python version: {detected_version} (confidence: {confidence})",
          flush=True)

    # Use PLLM's import scraper
    scraper = DepsScraper(logging=False)
    raw_imports = scraper.find_word_in_file(state["snippet_path"], "import", [])

    # Map import names → pip package names using expanded table
    mapped = []
    for imp in raw_imports:
        base = imp.split('.')[0]
        mapped.append(IMPORT_TO_PACKAGE.get(base, IMPORT_TO_PACKAGE.get(imp, imp)))

    # Also run through PLLM's module_link.json cleaner
    pypi = PyPIQuery(logging=False)
    cleaned = pypi.check_module_name(list(set(mapped)))

    print(f"[extract_imports] Imports: {cleaned}", flush=True)

    return {
        "imports": cleaned,
        "snippet_content": content,
        "python_version": detected_version,
    }


# ─────────────────────────────────────────────
# NODE 2: fetch_pypi_versions
# ─────────────────────────────────────────────

def fetch_pypi_versions(state: AgentState) -> dict:
    """Fetch available PyPI versions for each module."""
    python_version = state.get("python_version", "3.8")
    print(f"\n[Node: fetch_pypi_versions] Python {python_version}", flush=True)

    pypi = PyPIQuery(logging=False)
    module_details = {
        "python_version": python_version,
        "python_modules": state["imports"],
    }

    try:
        updated_modules, checked_version = pypi.get_module_specifics(module_details)
    except Exception as e:
        print(f"[fetch_pypi_versions] Warning: {e}", flush=True)
        updated_modules = state["imports"]
        checked_version = python_version

    pypi_versions = {}
    for module in updated_modules:
        versions_str = pypi.read_module_file(module, checked_version)
        if versions_str:
            pypi_versions[module] = versions_str

    print(f"[fetch_pypi_versions] Got versions for {len(pypi_versions)} modules", flush=True)

    return {
        "pypi_versions": pypi_versions,
        "python_version": checked_version,
        "imports": updated_modules,
    }


# ─────────────────────────────────────────────
# NODE 3: llm_generate_spec
# ─────────────────────────────────────────────

def _build_prompt(state: AgentState) -> str:
    """
    Build a flat prompt for Gemma2.

    Key design choices:
    - Single HumanMessage (Gemma2 doesn't handle SystemMessage well via Ollama)
    - Compat-map pre-fills are shown as "ALREADY DECIDED" so the LLM doesn't
      waste effort reconsidering them
    - History of failures embedded as plain text (last 3 attempts only)
    - Version lists trimmed to 20 entries to stay within Gemma2's context window
    """
    attempt = state.get("attempt", 1)
    python_version = state.get("python_version", "3.8")

    # ── Pre-fill versions from compat map (deterministic, high confidence)
    pre_filled = {}
    needs_llm = []
    for mod in state.get("imports", []):
        ver = get_compat_version(mod, python_version)
        if ver is None:
            # Not in compat map — LLM must decide
            needs_llm.append(mod)
        elif ver == '':
            # In map but "use latest" — still let LLM pick from PyPI list
            needs_llm.append(mod)
        else:
            pre_filled[mod] = ver

    # ── PyPI version context (only for modules the LLM needs to decide)
    version_lines = []
    for mod in needs_llm:
        versions_str = state.get("pypi_versions", {}).get(mod, "")
        if versions_str:
            versions = [v.strip() for v in versions_str.split(",") if v.strip()]
            if len(versions) > 20:
                trimmed = versions[:5] + versions[len(versions)//2-2:len(versions)//2+3] + versions[-10:]
            else:
                trimmed = versions
            version_lines.append(f"  {mod}: {', '.join(trimmed)}")
        else:
            version_lines.append(f"  {mod}: (no version data — pick a reasonable version)")

    # ── Failure history (last 3 only)
    history_text = ""
    if state.get("history") and attempt > 1:
        history_text = "\nPREVIOUS FAILED ATTEMPTS (do NOT repeat these combinations):\n"
        for record in state["history"][-3:]:
            reqs_str = ", ".join(f"{k}=={v}" for k, v in record["requirements"].items())
            err = _summarize_error(record["error_log"], record["error_type"])
            history_text += f"  - [{reqs_str}] → {record['error_type']}: {err}\n"
        history_text += "\n"

    # ── Retry-specific guidance
    if attempt == 1:
        instruction = (
            "Pick one version per module from its list above. "
            "Prefer versions from the middle or recent end of each list."
        )
    else:
        instruction = (
            "The previous attempts FAILED. Read the failure history carefully.\n"
            "- VersionNotFound: pick a version that actually exists in the list above\n"
            "- DependencyConflict: try a significantly different version (older or newer)\n"
            "- ModuleNotFound: the module name may be wrong — check carefully\n"
            "- NonZeroCode: the package may not compile on this Python version\n"
            "Pick DIFFERENT versions from all previous attempts."
        )

    # ── Build final prompt
    pre_filled_text = ""
    if pre_filled:
        pre_filled_text = (
            "\nALREADY DECIDED (use exactly these, do not change them):\n"
            + "\n".join(f"  {k}=={v}" for k, v in pre_filled.items())
            + "\n"
        )

    llm_modules_text = ""
    if version_lines:
        llm_modules_text = (
            "\nYOU MUST DECIDE these modules (pick one version each from the list):\n"
            + "\n".join(version_lines)
        )
    else:
        llm_modules_text = "\n(All modules already decided — just confirm the Python version.)"

    snippet_preview = state.get("snippet_content", "")[:1200]

    prompt = f"""You are a Python dependency resolver. Output JSON only — no explanation, no markdown, no code fences.

Python snippet:
{snippet_preview}

Python version to use: {python_version}
{pre_filled_text}{llm_modules_text}
{history_text}{instruction}

Return ONLY this JSON (merge already-decided and your decisions):
{{"python_version": "{python_version}", "requirements": {{"module_name": "version"}}}}"""

    return prompt, pre_filled


def _summarize_error(error_log: str, error_type: str) -> str:
    if not error_log:
        return error_type
    for line in error_log.split("\n"):
        line = line.strip()
        if any(kw in line for kw in ["ERROR:", "Could not find", "No matching",
                                      "Conflict", "error:"]):
            return line[:150]
    return error_log[:150]


def llm_generate_spec(state: AgentState, llm) -> dict:
    """
    Generate a requirements spec using the LLM.

    Compat-map entries are pre-filled deterministically.
    The LLM only decides modules not in the map.
    On retry, full failure history is embedded in the prompt.
    """
    attempt = state.get("attempt", 1)
    print(f"\n[Node: llm_generate_spec] Attempt #{attempt}", flush=True)

    prompt, pre_filled = _build_prompt(state)
    user_msg = HumanMessage(content=prompt)

    try:
        response = llm.invoke([user_msg])
        raw = response.content.strip()

        # Strip markdown fences if Gemma2 added them
        if "```" in raw:
            for part in raw.split("```"):
                if "{" in part:
                    raw = part.lstrip("json").strip()
                    break

        # Extract JSON blob (Gemma2 sometimes adds preamble text)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        parsed = json.loads(raw)
        llm_requirements = parsed.get("requirements", {})
        python_version = str(parsed.get("python_version",
                                        state.get("python_version", "3.8")))

        # Merge: pre_filled (compat map) takes priority, LLM fills the rest
        final_requirements = dict(pre_filled)
        for mod, ver in llm_requirements.items():
            ver = str(ver).strip().split(" ")[0]
            if ver and ver.lower() not in ("none", "null", "") and mod not in final_requirements:
                final_requirements[mod] = ver

        print(f"[llm_generate_spec] python={python_version}, "
              f"pre-filled={len(pre_filled)}, llm-decided={len(llm_requirements)}, "
              f"final={len(final_requirements)}", flush=True)

        return {
            "requirements": final_requirements,
            "python_version": python_version,
            "messages": [user_msg, AIMessage(content=raw)],
        }

    except Exception as e:
        print(f"[llm_generate_spec] Parse error: {e} — using compat map fallback", flush=True)
        # Fallback: compat map where possible, middle of PyPI list otherwise
        fallback = dict(pre_filled)
        for module, versions_str in state.get("pypi_versions", {}).items():
            if module not in fallback:
                versions = [v.strip() for v in versions_str.split(",") if v.strip()]
                if versions:
                    fallback[module] = versions[len(versions) // 2]
        return {
            "requirements": fallback,
            "python_version": state.get("python_version", "3.8"),
            "messages": [],
        }


# ─────────────────────────────────────────────
# NODE 4: docker_build
# ─────────────────────────────────────────────

def docker_build(state: AgentState) -> dict:
    """Build and run the Docker container with the current requirements."""
    print(f"\n[Node: docker_build] {state['requirements']}", flush=True)

    docker = DockerHelper(logging=False)
    llm_out = {
        "python_version": state["python_version"],
        "python_modules": state["requirements"],
    }

    try:
        docker.create_dockerfile(llm_out, state["snippet_path"])
        build_passed, build_output = docker.build_dockerfile(state["snippet_path"])

        if not build_passed:
            print(f"[docker_build] Build FAILED — {_classify_error(build_output)}", flush=True)
            docker.delete_image()
            return {
                "build_success": False,
                "run_success": False,
                "error_log": build_output,
                "error_type": _classify_error(build_output),
            }

        print(f"[docker_build] Build OK — running container...", flush=True)
        run_output = docker.run_container_test()
        run_error_type = _classify_error(run_output)
        run_passed = run_error_type in ("None", "NameError")
        docker.delete_image()

        print(f"[docker_build] Run {'OK' if run_passed else run_error_type}", flush=True)
        return {
            "build_success": True,
            "run_success": run_passed,
            "error_log": "" if run_passed else run_output,
            "error_type": run_error_type,
        }

    except Exception as e:
        return {
            "build_success": False,
            "run_success": False,
            "error_log": str(e),
            "error_type": "Unknown",
        }


def _classify_error(message: str) -> str:
    if not message:
        return "None"
    if "Could not find a version" in message:
        return "VersionNotFound"
    if "dependency conflicts" in message:
        return "DependencyConflict"
    if "ModuleNotFoundError" in message:
        return "ModuleNotFound"
    if "ImportError" in message:
        return "ImportError"
    if "AttributeError" in message:
        return "AttributeError"
    if "InvalidVersion" in message:
        return "InvalidVersion"
    if "non-zero code" in message:
        return "NonZeroCode"
    if "SyntaxError" in message:
        return "SyntaxError"
    if "NameError" in message:
        return "NameError"
    return "None"


# ─────────────────────────────────────────────
# NODE 5: analyze_error
# ─────────────────────────────────────────────

def analyze_error(state: AgentState) -> dict:
    """Record failure and increment attempt counter."""
    attempt = state.get("attempt", 1)
    print(f"\n[Node: analyze_error] Attempt #{attempt} — {state['error_type']}", flush=True)

    record = {
        "attempt": attempt,
        "requirements": dict(state["requirements"]),
        "python_version": state["python_version"],
        "error_type": state["error_type"],
        "error_log": state["error_log"][:600],
    }

    return {
        "attempt": attempt + 1,
        "history": [record],
        "build_success": False,
        "run_success": False,
    }


# ─────────────────────────────────────────────
# NODE 6: write_result
# ─────────────────────────────────────────────

def write_result(state: AgentState, oracle: KnowledgeOracle) -> dict:
    """Write requirements.txt and record success in the oracle session cache."""
    snippet_dir = os.path.dirname(state["snippet_path"])
    result_path = os.path.join(snippet_dir, "requirements.txt")

    lines = [f"# Python {state['python_version']}"]
    for module, version in state["requirements"].items():
        if version:
            lines.append(f"{module}=={version}")
        else:
            lines.append(module)

    with open(result_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Store in oracle so later snippets can reuse this solution
    gist_id = Path(state["snippet_path"]).parent.name
    oracle.record_success(gist_id, state["python_version"], state["requirements"])

    attempts_taken = state.get("attempt", 1) - 1
    print(f"\n[Node: write_result] SUCCESS in {attempts_taken} attempt(s) → {result_path}",
          flush=True)

    return {"status": "success", "result_path": result_path}


# ─────────────────────────────────────────────
# NODE 7: write_failure
# ─────────────────────────────────────────────

def write_failure(state: AgentState) -> dict:
    """Record exhausted attempts."""
    snippet_dir = os.path.dirname(state["snippet_path"])
    result_path = os.path.join(snippet_dir, "failed.json")

    with open(result_path, "w") as f:
        json.dump({
            "snippet": state["snippet_path"],
            "attempts": state.get("attempt", 1) - 1,
            "last_error": state.get("error_type", "Unknown"),
            "history": state.get("history", []),
        }, f, indent=2)

    print(f"\n[Node: write_failure] FAILED after {state.get('attempt',1)-1} attempts.",
          flush=True)
    return {"status": "failed", "result_path": result_path}


# ─────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────

def route_after_oracle(state: AgentState) -> str:
    """After oracle_lookup: skip to write_result if we have a confident hit."""
    if state.get("status") == "oracle_hit":
        return "write_result"
    # If oracle gave us a hint (imports pre-set), skip extract_imports
    if state.get("imports"):
        return "fetch_pypi_versions"
    return "extract_imports"


def route_after_build(state: AgentState) -> str:
    """After docker_build: success, retry, or give up."""
    if state.get("build_success") and state.get("run_success"):
        return "write_result"
    if state.get("attempt", 1) >= state.get("max_attempts", 10):
        return "write_failure"
    return "analyze_error"


# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────

def build_graph(llm, oracle: KnowledgeOracle):
    """Assemble the LangGraph StateGraph with all nodes wired up."""

    # Inject dependencies via closures (LangGraph nodes must be plain callables)
    def _oracle_lookup(state):
        return oracle_lookup(state, oracle)

    def _llm_generate_spec(state):
        return llm_generate_spec(state, llm)

    def _write_result(state):
        return write_result(state, oracle)

    graph = StateGraph(AgentState)

    graph.add_node("oracle_lookup",       _oracle_lookup)
    graph.add_node("extract_imports",     extract_imports)
    graph.add_node("fetch_pypi_versions", fetch_pypi_versions)
    graph.add_node("llm_generate_spec",   _llm_generate_spec)
    graph.add_node("docker_build",        docker_build)
    graph.add_node("analyze_error",       analyze_error)
    graph.add_node("write_result",        _write_result)
    graph.add_node("write_failure",       write_failure)

    # Entry point
    graph.set_entry_point("oracle_lookup")

    # Oracle branches: hit → write_result, hint → fetch_pypi_versions, miss → extract_imports
    graph.add_conditional_edges(
        "oracle_lookup",
        route_after_oracle,
        {
            "write_result":        "write_result",
            "fetch_pypi_versions": "fetch_pypi_versions",
            "extract_imports":     "extract_imports",
        },
    )

    # Normal pipeline
    graph.add_edge("extract_imports",     "fetch_pypi_versions")
    graph.add_edge("fetch_pypi_versions", "llm_generate_spec")
    graph.add_edge("llm_generate_spec",   "docker_build")

    # Build result branches
    graph.add_conditional_edges(
        "docker_build",
        route_after_build,
        {
            "write_result":  "write_result",
            "write_failure": "write_failure",
            "analyze_error": "analyze_error",
        },
    )

    # Retry loop
    graph.add_edge("analyze_error", "llm_generate_spec")

    # Terminal
    graph.add_edge("write_result",  END)
    graph.add_edge("write_failure", END)

    return graph.compile()


# ─────────────────────────────────────────────
# LLM FACTORY
# ─────────────────────────────────────────────

def get_llm(model: str = "gemma2",
            base_url: str = "http://localhost:11434",
            temperature: float = 0.7):
    """
    Return a configured LLM.
    Default: Gemma2 via Ollama (local, no API key).
    format='json' forces Ollama to output valid JSON — essential for Gemma2.
    """
    if "gpt" in model or "o1" in model:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    elif "claude" in model:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)
    else:
        return ChatOllama(
            base_url=base_url,
            model=model,
            format="json",
            temperature=temperature,
        )

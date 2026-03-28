"""
agent.py

LangGraph Agent: PLLM Improved Baseline

This file defines the full agentic pipeline using LangGraph's StateGraph, wiring together our three improvements:

  Improvement #1: Knowledge Graph lookup  (helpers/knowledge_graph.py)
  Improvement #2: Structured Error Classifier (helpers/error_classifier.py)
  Improvement #3: Multi-Agent Debate Loop  (helpers/multi_agent.py)

Flow:
[Start] -> Extract Dependencies -> Knowledge Graph Lookup -> (if good coverage) Docker Build -> Success
                                │
                                │
                                └─ (if poor coverage) -> Debate Loop -> Docker Build -> Success / Loop
If Docker Build fails, we classify the error and feed that back into the debate loop for a new proposal. We also track previous attempts to avoid repeating the same failed requirements.
"""

import ast
import os
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END

from helpers.knowledge_graph import (
    query_working_versions,
    coverage as kg_coverage,
    record_success,
    build_db,
)
from helpers.error_classifier import classify_error
from helpers.multi_agent     import run_debate
from helpers.docker_runner   import run_docker_build

class AgentState(TypedDict):
    """
    Everything the agent needs to remember across nodes and iterations.

    LangGraph passes this dict from node to node, and each node returns
    a partial update (only the keys it changed).
    """
    snippet_path:     str        # full path to the Python file, e.g. /gists/.../snippet.py
    python_version:   str        # initial Python version guess, e.g. "3.8"
    model:            str        # Ollama model name, e.x. "gemma2"
    ollama_base_url:  str        # Ollama endpoint, e.x. "http://ollama:11434"
    max_iterations:   int        # how many LLM retry loops are allowed
    packages:         list       # import names extracted from the snippet
    requirements:     dict       # current best {package: version} proposal
    iteration:        int        # how many LLM loops have run
    previous_attempts: list      # list of requirements dicts tried so far
    error_log:        str        # raw Docker build output (on failure)
    error_info:       dict       # structured output from classify_error()
    status:           str        # "running" | "success" | "failed"
    used_kg:          bool       # did the KG provide the requirements?

def extract_dependencies(state: AgentState) -> dict:
    """
    Parse the snippet file and pull out all import names.

    We use Python's built-in `ast` module for accuracy.
    """
    print(f"\n[Node] extract_dependencies — file: {state['snippet_path']}")

    try:
        with open(state["snippet_path"], "r") as f:
            source = f.read()

        tree = ast.parse(source)
        packages = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    packages.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    packages.add(node.module.split(".")[0])

        stdlib_names = _get_stdlib_names()
        third_party = sorted(packages - stdlib_names)

        print(f"[Node] Found packages: {third_party}")
        return {"packages": third_party}

    except Exception as exc:
        print(f"[Node] extract_dependencies error: {exc} — continuing with empty list")
        return {"packages": []}

def knowledge_graph_lookup(state: AgentState) -> dict:
    """
    Query the SQLite knowledge graph for known-good package versions.

    If we get >= 70% coverage of the required packages, we use those
    versions directly, no LLM call needed.

    Sets `used_kg = True` when we trust the KG result enough to skip the LLM.
    """
    print(f"\n[Node] knowledge_graph_lookup — python {state['python_version']}")

    packages = state.get("packages", [])
    python_version = state["python_version"]

    kg_versions = query_working_versions(packages, python_version)
    cov = kg_coverage(packages, python_version) if packages else 0.0
    print(f"[Node] KG coverage: {cov:.0%}  ({len(kg_versions)}/{len(packages)} packages)")

    use_kg = cov >= 0.70
    requirements = {pkg: kg_versions.get(pkg, "latest") for pkg in packages}

    return {
        "requirements": requirements,
        "used_kg":      use_kg,
    }

def docker_build(state: AgentState) -> dict:
    """
    Attempt to build a Docker environment with the current requirements.

    Calls `run_docker_build()` from docker_runner.py which:
      1. Writes a temporary Dockerfile + requirements.txt.
      2. Runs `docker build`.
      3. Returns (success: bool, log: str).

    On success: mark status = "success"
    On failure: store the error log for the classifier to read next.
    """
    iteration = state.get("iteration", 0) + 1
    print(f"\n[Node] docker_build — iteration {iteration}")
    print(f"[Node] Requirements: {state['requirements']}")

    success, log = run_docker_build(
        snippet_path=state["snippet_path"],
        python_version=state["python_version"],
        requirements=state["requirements"],
    )

    if success:
        print("[Node] Docker build SUCCEEDED!")
        return {
            "status":    "success",
            "error_log": "",
            "iteration": iteration,
        }
    else:
        print(f"[Node] Docker build FAILED (iteration {iteration})")
        return {
            "status":    "running",
            "error_log": log,
            "iteration": iteration,
            "previous_attempts": state.get("previous_attempts", []) + [state["requirements"]],
        }

def classify_error_node(state: AgentState) -> dict:
    """
    Turn the raw Docker error log into a structured dict.

    The structured dict is what our specialised LLM prompts are built from.
    """
    print(f"\n[Node] classify_error")
    error_info = classify_error(state.get("error_log", ""))
    print(f"[Node] Error classified as: {error_info['error_type']} | package: {error_info.get('package')}")
    return {"error_info": error_info}

def debate_loop(state: AgentState) -> dict:
    """
    Run the Proposer -> Critic -> Decider debate to get a new requirements dict.

    All three roles share the same context:
      - Which packages are needed.
      - What went wrong (structured error_info).
      - What was tried before (previous_attempts).

    The Decider's output becomes the new `requirements` for the next
    docker_build attempt.
    """
    print(f"\n[Node] debate_loop — iteration {state.get('iteration', 0)}")

    final_proposal = run_debate(
        packages=state.get("packages", []),
        python_version=state["python_version"],
        error_info=state.get("error_info", {}),
        previous_attempts=state.get("previous_attempts", []),
        model=state.get("model", "gemma2"),
        base_url=state.get("ollama_base_url", "http://localhost:11434"),
    )

    new_requirements = final_proposal.get("requirements", state.get("requirements", {}))
    print(f"[Node] New requirements from debate: {new_requirements}")

    return {"requirements": new_requirements}


def record_success_node(state: AgentState) -> dict:
    """
    When a build succeeds, persist the winning (pkg, ver, python_ver) triples
    to the knowledge graph so future runs can benefit without calling the LLM.
    """
    print(f"\n[Node] record_success — storing results in knowledge graph")
    record_success(
        packages=state.get("requirements", {}),
        python_version=state["python_version"],
    )
    return {"status": "success"}

def mark_failed(state: AgentState) -> dict:
    """Terminal node — called when max iterations are exhausted."""
    print(f"\n[Node] mark_failed — giving up after {state.get('iteration')} iterations")
    return {"status": "failed"}

def route_after_kg_lookup(state: AgentState) -> str:
    """
    After the KG lookup, decide the next step:
      - If KG coverage is good, try the docker build right away.
      - Otherwise go straight to the debate loop (need LLM help).
    """
    if state.get("used_kg", False):
        print("[Router] KG coverage sufficient — attempting direct Docker build")
        return "docker_build"
    else:
        print("[Router] KG coverage insufficient — going to debate loop")
        return "debate_loop"


def route_after_docker_build(state: AgentState) -> str:
    """
    After a Docker build attempt, decide next step:
      - success: record it and finish.
      - failure + room: classify the error and loop.
      - failure + limit: give up.
    """
    if state.get("status") == "success":
        return "record_success"

    if state.get("iteration", 0) >= state.get("max_iterations", 5):
        print(f"[Router] Reached max iterations ({state['max_iterations']}) — failing")
        return "mark_failed"

    return "classify_error"

def build_graph() -> StateGraph:
    """
    Wire all nodes and edges together into a LangGraph StateGraph.
    """
    graph = StateGraph(AgentState)
    graph.add_node("extract_dependencies",  extract_dependencies)
    graph.add_node("knowledge_graph_lookup", knowledge_graph_lookup)
    graph.add_node("docker_build",          docker_build)
    graph.add_node("classify_error",        classify_error_node)
    graph.add_node("debate_loop",           debate_loop)
    graph.add_node("record_success",        record_success_node)
    graph.add_node("mark_failed",           mark_failed)

    graph.set_entry_point("extract_dependencies")

    graph.add_edge("extract_dependencies",  "knowledge_graph_lookup")
    graph.add_edge("classify_error",        "debate_loop")
    graph.add_edge("debate_loop",           "docker_build")
    graph.add_edge("record_success",        END)
    graph.add_edge("mark_failed",           END)

    graph.add_conditional_edges(
        "knowledge_graph_lookup",
        route_after_kg_lookup,
        {
            "docker_build": "docker_build",
            "debate_loop":  "debate_loop",
        }
    )

    graph.add_conditional_edges(
        "docker_build",
        route_after_docker_build,
        {
            "record_success": "record_success",
            "classify_error": "classify_error",
            "mark_failed":    "mark_failed",
        }
    )

    return graph

def run_agent(
    snippet_path: str,
    python_version: str  = "3.8",
    model: str           = "gemma2",
    ollama_base_url: str = "http://localhost:11434",
    max_iterations: int  = 5,
) -> dict:
    """
    Entry-point called by run.py (and the competition harness).

    Returns the final AgentState dict so the caller can inspect
    `status`, `requirements`, `iteration`, etc.
    """

    # Build/refresh the KG on first run (skips if DB already exists)
    _maybe_build_kg()

    # Compile the graph once
    app = build_graph().compile()

    # Seed the initial state
    initial_state: AgentState = {
        "snippet_path":     snippet_path,
        "python_version":   python_version,
        "model":            model,
        "ollama_base_url":  ollama_base_url,
        "max_iterations":   max_iterations,
        "packages":         [],
        "requirements":     {},
        "iteration":        0,
        "previous_attempts": [],
        "error_log":        "",
        "error_info":       {},
        "status":           "running",
        "used_kg":          False,
    }

    print(f"\n{'='*60}")
    print(f"  PLLM++ Agent starting")
    print(f"  Snippet : {snippet_path}")
    print(f"  Python  : {python_version}")
    print(f"  Model   : {model}")
    print(f"  Max iter: {max_iterations}")
    print(f"{'='*60}\n")

    final_state = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"  Agent finished — status: {final_state['status']}")
    print(f"  Iterations used: {final_state['iteration']}")
    print(f"  Final requirements: {final_state.get('requirements')}")
    print(f"{'='*60}\n")

    return final_state

def _maybe_build_kg() -> None:
    """
    Check that the knowledge graph DB exists. Never build it at test time.

    The DB should be built once before testing using build_kg.py:
        python build_kg.py

    If the DB is missing the agent still works; KG lookups return empty
    dicts and the LLM debate loop handles everything as a fallback.
    """
    from helpers.knowledge_graph import DEFAULT_DB_PATH
    if not os.path.exists(DEFAULT_DB_PATH):
        print(
            f"[KG] WARNING: No DB found at {DEFAULT_DB_PATH}. "
            "Run `python build_kg.py` once before testing to enable "
            "KG-assisted resolution. Continuing without KG."
        )
    else:
        print(f"[KG] Knowledge graph found at {DEFAULT_DB_PATH} — ready.")


def _get_stdlib_names() -> set:
    """
    Return a set of Python standard-library top-level module names.
    We use sys.stdlib_module_names (Python 3.10+) and fall back to a
    hand-curated list for older versions.
    """
    import sys
    if hasattr(sys, "stdlib_module_names"):
        return set(sys.stdlib_module_names)
    # Fallback list (not exhaustive but covers the most common ones)
    return {
        "os", "sys", "re", "io", "math", "time", "json", "csv",
        "ast", "abc", "copy", "enum", "glob", "gzip", "hash",
        "http", "logging", "multiprocessing", "pathlib", "pickle",
        "platform", "pprint", "queue", "random", "shutil", "signal",
        "socket", "sqlite3", "string", "struct", "subprocess",
        "tempfile", "threading", "typing", "unittest", "urllib",
        "uuid", "warnings", "xml", "zipfile", "collections",
        "contextlib", "dataclasses", "datetime", "decimal",
        "functools", "hashlib", "heapq", "inspect", "itertools",
        "operator", "optparse", "argparse", "traceback",
        "__future__", "builtins",
    }

"""
helpers/multi_agent.py
-----------------------
Multi-Agent Debate Loop — Improvement #3

Instead of asking the LLM once "fix my requirements", we run a small
debate between three specialised roles:

  1. Proposer  — suggests a candidate requirements.txt
  2. Critic    — reviews the proposal and lists problems (JSON output)
  3. Decider   — picks the best proposal (or refines further)

Why does this help?
  - The Critic often catches obvious mistakes BEFORE a Docker build,
    saving several minutes per iteration.
  - The Decider can merge insights from multiple proposals.
  - Each role gets a *focused* prompt → better output.

All three roles call the same Ollama endpoint but with different prompts.
The `call_llm` function is kept as a thin wrapper so it's easy to swap
Ollama for any other backend.
"""

import json
import re
import requests
from typing import Optional


# ---------------------------------------------------------------------------
# Low-level LLM call (thin wrapper around the Ollama /api/generate endpoint)
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    model: str = "gemma2",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
) -> str:
    """
    Send a prompt to Ollama and return the text response.

    If Ollama is unreachable (e.g. during unit tests) this returns a
    hard-coded placeholder so the rest of the pipeline keeps running.
    """
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,  # Docker builds can be slow; give LLM time to think
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.RequestException as exc:
        # Graceful fallback so tests / CI don't crash without Ollama
        print(f"[LLM] Ollama call failed ({exc}) — returning placeholder")
        return '{"requirements": [], "reasoning": "LLM unavailable"}'


# ---------------------------------------------------------------------------
# Role 1 — Proposer
# ---------------------------------------------------------------------------

def propose_requirements(
    packages: list,
    python_version: str,
    error_info: dict,
    previous_attempts: list,
    model: str = "gemma2",
    base_url: str = "http://localhost:11434",
) -> dict:
    """
    Ask the LLM to generate a candidate requirements.txt.

    Returns:
    {
        "python_version": "3.8",
        "requirements":   {"numpy": "1.21.0", "pandas": "1.3.0"},
        "reasoning":      "Chose numpy 1.21 because ..."
    }
    """
    # Build a short history of what we already tried (to avoid repetition)
    history_text = ""
    if previous_attempts:
        history_text = "Previous failed attempts (DO NOT repeat these):\n"
        for i, attempt in enumerate(previous_attempts[-3:], 1):   # last 3 only
            history_text += f"  Attempt {i}: {json.dumps(attempt)}\n"

    # Tailor the prompt to the specific error type (Improvement #2 synergy)
    error_hint = _error_hint(error_info)

    prompt = f"""You are a Python dependency resolver.

Task: Suggest a working requirements.txt for the following packages.
Python version: {python_version}
Packages needed: {', '.join(packages)}

Error that occurred with the previous attempt:
  Type   : {error_info.get('error_type', 'unknown')}
  Package: {error_info.get('package', 'unknown')}
  Hint   : {error_hint}

{history_text}

Respond with ONLY valid JSON in this exact format (no markdown, no preamble):
{{
  "python_version": "{python_version}",
  "requirements": {{"package_name": "version", ...}},
  "reasoning": "one sentence explaining your choices"
}}
"""
    raw = call_llm(prompt, model=model, base_url=base_url, temperature=0.7)
    return _safe_parse_json(raw, fallback={
        "python_version": python_version,
        "requirements":   {p: "latest" for p in packages},
        "reasoning":      "LLM parse failed — using latest",
    })


# ---------------------------------------------------------------------------
# Role 2 — Critic
# ---------------------------------------------------------------------------

def critique_proposal(
    proposal: dict,
    packages: list,
    python_version: str,
    error_info: dict,
    model: str = "gemma2",
    base_url: str = "http://localhost:11434",
) -> dict:
    """
    Ask the LLM to evaluate a proposal and flag any problems.

    Returns:
    {
        "issues":   ["numpy 99.0 does not exist on PyPI", ...],
        "score":    7,          # 0-10, higher = better
        "approved": true/false
    }
    """
    prompt = f"""You are a Python packaging expert reviewing a dependency proposal.

Proposed requirements:
{json.dumps(proposal.get('requirements', {}), indent=2)}

Python version: {python_version}
Original packages requested: {', '.join(packages)}
Error context: type={error_info.get('error_type','?')}, package={error_info.get('package','?')}

Review the proposal for:
  1. Versions that do not exist on PyPI.
  2. Known incompatibilities between packages.
  3. Packages missing from the proposal.
  4. Python version compatibility.

Respond with ONLY valid JSON (no markdown):
{{
  "issues":   ["list any problems here, or empty list if none"],
  "score":    <integer 0-10>,
  "approved": <true or false>
}}
"""
    raw = call_llm(prompt, model=model, base_url=base_url, temperature=0.3)
    return _safe_parse_json(raw, fallback={
        "issues":   ["Critic response could not be parsed"],
        "score":    5,
        "approved": True,   # allow pipeline to continue
    })


# ---------------------------------------------------------------------------
# Role 3 — Decider
# ---------------------------------------------------------------------------

def decide_best_proposal(
    proposal: dict,
    critique: dict,
    packages: list,
    python_version: str,
    model: str = "gemma2",
    base_url: str = "http://localhost:11434",
) -> dict:
    """
    Given the Proposer's output and the Critic's feedback, the Decider
    either endorses the proposal or produces a corrected version.

    Returns the same shape as `propose_requirements`:
    {
        "python_version": ...,
        "requirements":   {...},
        "reasoning":      ...
    }
    """
    # If the Critic approved and gave a good score, skip the extra LLM call
    if critique.get("approved") and critique.get("score", 0) >= 7:
        print("[Decider] Critic approved — accepting proposal directly.")
        return proposal

    issues_text = "\n".join(f"  - {i}" for i in critique.get("issues", []))

    prompt = f"""You are the final decision-maker for Python dependency resolution.

Original proposal:
{json.dumps(proposal.get('requirements', {}), indent=2)}

Critic's concerns (score {critique.get('score', '?')}/10):
{issues_text or "  (none)"}

Python version: {python_version}
Required packages: {', '.join(packages)}

Fix the proposal by addressing the critic's concerns.
Respond with ONLY valid JSON (no markdown):
{{
  "python_version": "{python_version}",
  "requirements": {{"package_name": "version", ...}},
  "reasoning": "explain what you changed and why"
}}
"""
    raw = call_llm(prompt, model=model, base_url=base_url, temperature=0.5)
    return _safe_parse_json(raw, fallback=proposal)   # fall back to original if parse fails


# ---------------------------------------------------------------------------
# Convenience wrapper — run the full debate in one call
# ---------------------------------------------------------------------------

def run_debate(
    packages: list,
    python_version: str,
    error_info: dict,
    previous_attempts: list,
    model: str = "gemma2",
    base_url: str = "http://localhost:11434",
) -> dict:
    """
    Orchestrates Proposer → Critic → Decider and returns a final proposal.

    This is the function the agent node calls.
    """
    print("[Debate] Step 1/3 — Proposer generating candidate …")
    proposal = propose_requirements(packages, python_version, error_info, previous_attempts, model, base_url)

    print("[Debate] Step 2/3 — Critic reviewing proposal …")
    critique = critique_proposal(proposal, packages, python_version, error_info, model, base_url)
    print(f"[Debate] Critic score: {critique.get('score')}/10, approved: {critique.get('approved')}")

    print("[Debate] Step 3/3 — Decider finalising …")
    final = decide_best_proposal(proposal, critique, packages, python_version, model, base_url)

    return final


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _error_hint(error_info: dict) -> str:
    """Return a human-readable hint based on the error category."""
    hints = {
        "version_conflict":         "Try downgrading the conflicting package or pinning its transitive deps.",
        "no_matching_distribution": "The version specified does not exist on PyPI — choose a real published version.",
        "missing_system_dep":       "This package needs a C library; consider a pre-built wheel or a different package.",
        "python_version_mismatch":  "Check the package's Requires-Python and align with the Python version.",
        "module_not_found":         "The package may be installed under a different PyPI name than its import name.",
        "unknown":                  "No specific guidance available — use your best judgement.",
    }
    return hints.get(error_info.get("error_type", "unknown"), hints["unknown"])


def _safe_parse_json(text: str, fallback: dict) -> dict:
    """
    Try to parse `text` as JSON.  If it fails, strip markdown fences and try
    again.  If that also fails, return `fallback`.
    """
    # Strip ```json ... ``` fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[LLM] Could not parse JSON from response:\n{text[:200]}")
        return fallback


# ---------------------------------------------------------------------------
# Smoke-test (requires no Ollama — uses the fallback path)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run_debate(
        packages=["numpy", "pandas"],
        python_version="3.8",
        error_info={"error_type": "no_matching_distribution", "package": "numpy", "version": "99.0"},
        previous_attempts=[],
        base_url="http://localhost:11434",   # will fail gracefully without Ollama
    )
    print("Final proposal:", json.dumps(result, indent=2))

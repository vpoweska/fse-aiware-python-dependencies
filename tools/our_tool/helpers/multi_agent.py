"""
multi_agent.py
Three-role debate loop: Proposer → Critic → Decider.
Uses LangChain (same as PLLM) so it works with the same Ollama connection.
Model is always gemma2 as required by the competition constraints.
"""
import json
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List


# ── Pydantic models for structured output ─────────────────────────────────────

class RequirementsSpec(BaseModel):
    packages: List[str] = Field(
        description="List of package==version strings, e.g. ['numpy==1.24.0', 'pandas==1.5.3']"
    )

class CritiqueResult(BaseModel):
    confidence: int = Field(description="Confidence 0-100 that this spec will work")
    issues: List[str] = Field(description="List of potential issues found")
    recommendation: str = Field(description="One of: accept, reject, modify")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _parse_requirements_from_list(packages_list: list) -> dict:
    """
    Convert ['numpy==1.24.0', 'pandas==1.5.3'] → {'numpy': '1.24.0', 'pandas': '1.5.3'}
    Tolerant of bare names, extra whitespace, markdown fences.
    """
    result = {}
    for item in packages_list:
        item = str(item).strip().strip('`').strip('"').strip("'")
        if not item or item.startswith('#'):
            continue
        m = re.match(r'^([A-Za-z0-9_\-\.]+)\s*[=><~!]+\s*([\w\.\-\+]+)', item)
        if m:
            result[m.group(1).lower()] = m.group(2)
        elif re.match(r'^[A-Za-z0-9_\-\.]+$', item):
            result[item.lower()] = None
    return result


def _parse_requirements_from_text(text: str) -> dict:
    """
    Fallback: parse a raw requirements.txt string.
    Handles markdown fences that Gemma 2 sometimes adds.
    """
    text = re.sub(r'```[a-z]*', '', text).strip('`').strip()
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = re.match(r'^([A-Za-z0-9_\-\.]+)\s*[=><~!]+\s*([\w\.\-\+]+)', line)
        if m:
            result[m.group(1).lower()] = m.group(2)
        elif re.match(r'^[A-Za-z0-9_\-\.]+$', line):
            result[line.lower()] = None
    return result


def dict_to_requirements_txt(packages: dict) -> str:
    """{'numpy': '1.24.0'} → 'numpy==1.24.0\npandas==...' """
    lines = []
    for pkg, ver in packages.items():
        if ver and str(ver) not in ("None", "none", ""):
            lines.append(f"{pkg}=={ver}")
        else:
            lines.append(str(pkg))
    return "\n".join(lines)


# ── Role 1: Proposer ───────────────────────────────────────────────────────────

def propose(llm_model, specialist_prompt: str) -> dict:
    """
    Calls Gemma 2 (via the passed LangChain model) with the specialist prompt.
    Returns {'packages': {'numpy': '1.24.0', ...}, 'raw': '...'}

    We use two strategies:
    1. Ask for JSON list of package==version strings (most reliable with Gemma 2)
    2. Fall back to parsing raw text if JSON parse fails
    """
    parser = JsonOutputParser(pydantic_object=RequirementsSpec)

    # Append JSON format instruction to the specialist prompt
    full_prompt_text = (
        specialist_prompt
        + "\n\nYou MUST respond with valid JSON in this exact format:\n"
        + parser.get_format_instructions()
    )

    prompt = PromptTemplate(
        template="{content}",
        input_variables=[],
        partial_variables={"content": full_prompt_text}
    )

    packages = {}
    raw = ""

    for attempt in range(3):
        try:
            chain = prompt | llm_model | parser
            out = chain.invoke({})
            raw = str(out)

            if isinstance(out, dict) and "packages" in out:
                packages = _parse_requirements_from_list(out["packages"])
            elif isinstance(out, list):
                packages = _parse_requirements_from_list(out)

            if packages:
                break

        except Exception as e:
            print(f"[Proposer] Attempt {attempt+1} failed: {e}")
            # Try to extract anything useful from whatever came back
            try:
                raw_text = str(e)
                packages = _parse_requirements_from_text(raw_text)
                if packages:
                    break
            except Exception:
                pass

    return {"packages": packages, "raw": raw}


# ── Role 2: Critic ─────────────────────────────────────────────────────────────

def critique(llm_model, proposed_packages: dict,
             import_names: list, error_type: str) -> dict:
    """
    Reviews the proposed packages for likely issues WITHOUT running Docker.
    Returns {'confidence': int, 'issues': [...], 'recommendation': str}
    """
    proposed_req_txt = dict_to_requirements_txt(proposed_packages)
    parser = JsonOutputParser(pydantic_object=CritiqueResult)

    prompt = PromptTemplate(
        template=(
            "You are a Python packaging expert reviewing a requirements.txt.\n\n"
            "Requirements to review:\n{requirements}\n\n"
            "The Python file uses these imports: {imports}\n"
            "Error type being fixed: {error_type}\n\n"
            "Check for:\n"
            "1. Version conflicts between packages\n"
            "2. Package names that don't match their pip install name\n"
            "3. Missing dependencies for the listed imports\n"
            "4. Versions that clearly don't exist\n\n"
            "Respond ONLY with JSON using this format:\n{format_instructions}"
        ),
        input_variables=[],
        partial_variables={
            "requirements": proposed_req_txt,
            "imports": ", ".join(import_names) if import_names else "unknown",
            "error_type": error_type,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    for attempt in range(3):
        try:
            chain = prompt | llm_model | parser
            out = chain.invoke({})
            if isinstance(out, dict):
                return {
                    "confidence": int(out.get("confidence", 50)),
                    "issues": out.get("issues", []),
                    "recommendation": out.get("recommendation", "accept"),
                }
        except Exception as e:
            print(f"[Critic] Attempt {attempt+1} failed: {e}")

    # If critic fails entirely, accept with low confidence — don't block progress
    return {"confidence": 40, "issues": ["Critic unavailable"], "recommendation": "accept"}


# ── Main debate loop ───────────────────────────────────────────────────────────

def debate_round(llm_model, specialist_prompt: str,
                 import_names: list, error_type: str,
                 previous_best: dict = None,
                 max_inner_rounds: int = 2) -> tuple:
    """
    Run the full propose → critique → decide loop.

    Args:
        llm_model       : the LangChain Ollama model instance (from OllamaHelperBase)
        specialist_prompt: built by error_classifier.build_specialist_prompt()
        import_names    : list of top-level import names from AST analysis
        error_type      : string from error_classifier.classify()
        previous_best   : last known packages dict (used as fallback)
        max_inner_rounds: how many propose→critique cycles (2 is enough)

    Returns:
        (best_packages: dict, confidence: int)
    """
    best_packages = previous_best.copy() if previous_best else {}
    best_confidence = 0

    for round_num in range(max_inner_rounds):
        print(f"[Agent] Debate round {round_num+1}/{max_inner_rounds}")

        # Role 1: Propose
        proposal = propose(llm_model, specialist_prompt)

        if not proposal["packages"]:
            print(f"[Agent] Proposer returned empty spec on round {round_num+1}, skipping")
            continue

        # Role 2: Critique
        critique_result = critique(
            llm_model,
            proposal["packages"],
            import_names,
            error_type
        )

        confidence = critique_result["confidence"]
        recommendation = critique_result["recommendation"]
        issues = critique_result["issues"]

        print(f"[Agent] Round {round_num+1}: confidence={confidence}, "
              f"recommendation={recommendation}")
        if issues:
            print(f"[Agent] Issues: {issues}")

        # Accept if better than current best and not explicitly rejected
        if recommendation != "reject" and confidence > best_confidence:
            best_packages = proposal["packages"]
            best_confidence = confidence
            print(f"[Agent] New best spec accepted: {best_packages}")

        # Early exit if confident enough
        if best_confidence >= 80:
            print(f"[Agent] Early exit at confidence={best_confidence}")
            break

    # Always return something — fall back to previous if agent produced nothing useful
    if not best_packages and previous_best:
        print("[Agent] Using previous best as fallback")
        best_packages = previous_best
        best_confidence = 30

    return best_packages, best_confidence


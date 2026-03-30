def propose(llm, snippet_code: str, error_log: str, rag_context: str, error_type: str) -> str:
    prompt = SPECIALIST_PROMPTS[error_type].format(
        error_summary=error_log[:500],
        rag_context=rag_context
    )
    return llm.generate(prompt)

def critique(llm, proposed_requirements: str, snippet_code: str) -> dict:
    prompt = f"""
You are a Python dependency expert. Review this requirements.txt for potential issues:
```
{proposed_requirements}
```

The code uses these imports: {extract_imports(snippet_code)}

Identify: version conflicts, packages that don't exist, wrong Python version assumptions.
Respond as JSON: {{"issues": [...], "confidence": 0-100, "recommendation": "accept|reject|modify"}}
"""
    response = llm.generate(prompt)
    try:
        return json.loads(response)
    except:
        return {"confidence": 50, "recommendation": "accept", "issues": []}

def debate_round(llm, snippet_code, error_log, rag_context, error_type, 
                 previous_best=None, max_rounds=3):
    best_spec = previous_best
    best_confidence = 0
    
    for round in range(max_rounds):
        candidate = propose(llm, snippet_code, error_log, rag_context, error_type)
        critique_result = critique(llm, candidate, snippet_code)
        
        if critique_result["recommendation"] == "accept" and \
           critique_result["confidence"] > best_confidence:
            best_spec = candidate
            best_confidence = critique_result["confidence"]
        
        if best_confidence >= 80:
            break  # good enough, don't waste calls
    
    return best_spec

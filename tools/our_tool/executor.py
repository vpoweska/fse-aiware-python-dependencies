def resolve(snippet_path: str, llm, db_path: str, max_loops: int = 10):
    source = open(snippet_path).read()
    
    # Step 1: Static analysis — free, instant
    analysis = analyze(source)
    python_versions = smart_version_range(analysis["min_python"], analysis["signals"])
    
    # Step 2: Knowledge graph lookup — try to skip LLM entirely
    kg_spec = build_spec_from_graph(db_path, analysis["import_names"], 
                                     analysis["min_python"][1])
    if kg_spec:
        results = test_parallel(kg_spec, snippet_path, python_versions)
        winner = next((r for r in results if r["success"]), None)
        if winner:
            save_to_memory(db_path, analysis["import_names"], kg_spec, 
                           winner["python_version"])
            return winner  # Done — no LLM needed!
    
    # Step 3: LLM-driven loop with multi-agent debate
    error_log = results[-1]["log"] if results else "No prior attempt"
    for loop in range(max_loops):
        error_type, error_info = classify(error_log)
        rag_context = query_rag(analysis["import_names"])
        
        spec = debate_round(llm, source, error_log, rag_context, error_type)
        results = test_parallel(spec, snippet_path, python_versions)
        
        winner = next((r for r in results if r["success"]), None)
        if winner:
            save_to_memory(db_path, analysis["import_names"], spec,
                           winner["python_version"])
            return winner
        
        # Use the worst error as input to next loop
        error_log = sorted(results, key=lambda r: len(r["log"]))[-1]["log"]
    
    return None  # Could not resolve

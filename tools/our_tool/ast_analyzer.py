import ast, re

VERSION_SIGNALS = {
    "walrus_operator":   (3, 8),   # := in expressions
    "match_statement":   (3, 10),  # match/case
    "fstring":           (3, 6),   # f"..."
    "async_def":         (3, 5),   # async def / await
    "type_hints":        (3, 5),   # def foo(x: int) -> str
    "positional_only":   (3, 8),   # def f(x, /, y)
    "exception_group":   (3, 11),  # ExceptionGroup
}

PRINT_STATEMENT = re.compile(r'^\s*print\s+[^\(]', re.MULTILINE)

def analyze(source: str) -> dict:
    """Returns {'min_python': (3,8), 'import_names': ['numpy','pandas'], ...}"""
    result = {"min_python": (3, 0), "signals": [], "import_names": []}
    
    # Python 2 check — can't even parse with ast
    if PRINT_STATEMENT.search(source):
        result["min_python"] = (2, 7)
        result["signals"].append("print_statement")
        return result  # no point parsing further
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        result["signals"].append("unparseable")
        return result
    
    for node in ast.walk(tree):
        # Collect all imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) \
                    else ([node.module] if node.module else [])
            result["import_names"].extend(names)
        
        # Walrus operator
        if isinstance(node, ast.NamedExpr):
            result["min_python"] = max(result["min_python"], (3, 8))
            result["signals"].append("walrus_operator")
        
        # Match statement (Python 3.10+)
        if hasattr(ast, "Match") and isinstance(node, ast.Match):
            result["min_python"] = max(result["min_python"], (3, 10))
            result["signals"].append("match_statement")
        
        # Async/await
        if isinstance(node, (ast.AsyncFunctionDef, ast.Await)):
            result["min_python"] = max(result["min_python"], (3, 5))
            result["signals"].append("async_def")
        
        # f-strings
        if isinstance(node, ast.JoinedStr):
            result["min_python"] = max(result["min_python"], (3, 6))
    
    result["import_names"] = list(set(result["import_names"]))
    return result

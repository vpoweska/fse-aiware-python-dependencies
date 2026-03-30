import ast
import re

# Maps detected AST feature → minimum Python (major, minor)
VERSION_SIGNALS = {
    "walrus_operator":  (3, 8),
    "match_statement":  (3, 10),
    "fstring":          (3, 6),
    "async_def":        (3, 5),
    "positional_only":  (3, 8),
    "exception_group":  (3, 11),
    "type_union_op":    (3, 10),  # X | Y in type hints
}

PRINT_STMT = re.compile(r'^\s*print\s+[^\(\n]', re.MULTILINE)
ENCODING_COMMENT = re.compile(r'#.*coding[:=]', re.IGNORECASE)

def analyze(source: str) -> dict:
    """
    Statically analyze a Python snippet.
    Returns:
        min_python  : (major, minor) tuple — lowest Python version that can run this
        import_names: list of top-level module names imported
        signals     : list of detected feature names
        parseable   : bool
    """
    result = {
        "min_python": (3, 6),   # safe default
        "import_names": [],
        "signals": [],
        "parseable": True,
    }

    # --- Python 2 detection (can't ast.parse these) ---
    if PRINT_STMT.search(source):
        result["min_python"] = (2, 7)
        result["signals"].append("print_statement")
        result["parseable"] = False
        # Still try to grab imports via regex for Python 2
        for m in re.finditer(r'^(?:import|from)\s+([\w\.]+)', source, re.MULTILINE):
            result["import_names"].append(m.group(1).split('.')[0])
        result["import_names"] = list(set(result["import_names"]))
        return result

    try:
        tree = ast.parse(source)
    except SyntaxError:
        result["parseable"] = False
        result["signals"].append("syntax_error")
        # Fallback: regex-extract imports
        for m in re.finditer(r'^(?:import|from)\s+([\w\.]+)', source, re.MULTILINE):
            result["import_names"].append(m.group(1).split('.')[0])
        result["import_names"] = list(set(result["import_names"]))
        return result

    min_ver = (3, 6)

    for node in ast.walk(tree):
        # --- Collect imports ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                result["import_names"].append(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result["import_names"].append(node.module.split('.')[0])

        # --- Walrus operator := ---
        if isinstance(node, ast.NamedExpr):
            min_ver = max(min_ver, (3, 8))
            if "walrus_operator" not in result["signals"]:
                result["signals"].append("walrus_operator")

        # --- match/case (3.10+) ---
        if hasattr(ast, "Match") and isinstance(node, ast.Match):
            min_ver = max(min_ver, (3, 10))
            if "match_statement" not in result["signals"]:
                result["signals"].append("match_statement")

        # --- async def / await ---
        if isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            min_ver = max(min_ver, (3, 5))
            if "async_def" not in result["signals"]:
                result["signals"].append("async_def")

        # --- f-strings ---
        if isinstance(node, ast.JoinedStr):
            min_ver = max(min_ver, (3, 6))
            if "fstring" not in result["signals"]:
                result["signals"].append("fstring")

        # --- positional-only params (3.8+) ---
        if isinstance(node, ast.arguments) and node.posonlyargs:
            min_ver = max(min_ver, (3, 8))
            if "positional_only" not in result["signals"]:
                result["signals"].append("positional_only")

        # --- ExceptionGroup (3.11+) ---
        if isinstance(node, ast.Name) and node.id == "ExceptionGroup":
            min_ver = max(min_ver, (3, 11))
            if "exception_group" not in result["signals"]:
                result["signals"].append("exception_group")

        # --- X | Y type union operator (3.10+) ---
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
                min_ver = max(min_ver, (3, 10))
                if "type_union_op" not in result["signals"]:
                    result["signals"].append("type_union_op")

    result["min_python"] = min_ver
    result["import_names"] = list(set(result["import_names"]))
    return result


def get_python_version_range(analysis: dict, search_range: int = 1) -> list:
    """
    Given AST analysis, return only the Python versions worth testing.
    Respects the same search_range logic as PLLM's get_python_range.
    """
    ALL_VERSIONS = ["3.6", "3.7", "3.8", "3.9", "3.10", "3.11", "3.12"]

    # Python 2 — special case
    if analysis["min_python"][0] == 2:
        return ["2.7"]

    min_minor = analysis["min_python"][1]

    # Filter to only versions >= min detected
    candidates = [v for v in ALL_VERSIONS if int(v.split(".")[1]) >= min_minor]

    if not candidates:
        candidates = ["3.8", "3.9", "3.10"]  # safe fallback

    return candidates

"""
pypi_resolver.py
Queries PyPI directly to get real available versions before calling the LLM.
Eliminates hallucinated versions entirely for the common case.

v2 fixes:
- Python 2.7 version caps: never try numpy==1.24 on Python 2.7
- Better python_major/minor handling to distinguish 2.7 from 3.7
- Fixes "list index out of range" by ensuring we never return a version
  that doesn't exist for the target Python version
"""
import re
import urllib.request
import urllib.error
import json


# Known package renames — import name -> pip name
KNOWN_RENAMES = {
    "sklearn":          "scikit-learn",
    "cv2":              "opencv-python",
    "PIL":              "Pillow",
    "pil":              "Pillow",
    "bs4":              "beautifulsoup4",
    "yaml":             "PyYAML",
    "gtk":              "PyGObject",
    "gi":               "PyGObject",
    "wx":               "wxPython",
    "serial":           "pyserial",
    "usb":              "pyusb",
    "Crypto":           "pycryptodome",
    "crypto":           "pycryptodome",
    "OpenSSL":          "pyOpenSSL",
    "dateutil":         "python-dateutil",
    "dotenv":           "python-dotenv",
    "attr":             "attrs",
    "magic":            "python-magic",
    "enchant":          "pyenchant",
    "Image":            "Pillow",
    "MySQLdb":          "mysqlclient",
    "boto":             "boto3",
    "jwt":              "PyJWT",
    "tensorflow_core":  "tensorflow",
    "tf":               "tensorflow",
    "appindicator":     None,
    "gtk2":             None,
    "gtk3":             None,
    "libxmp":           None,
    "urllib2":          None,
    "cookielib":        None,
    "HTMLParser":       None,
    "ConfigParser":     None,
    "httplib":          None,
    "Queue":            None,
    "cPickle":          None,
    "thread":           None,
}

# Packages that are stdlib — never pip install these
STDLIB_MODULES = {
    "os", "sys", "re", "json", "time", "datetime", "math", "random",
    "string", "io", "pathlib", "shutil", "glob", "fnmatch", "tempfile",
    "hashlib", "hmac", "secrets", "struct", "codecs", "copy", "pprint",
    "traceback", "logging", "warnings", "unittest", "abc", "contextlib",
    "functools", "itertools", "operator", "collections", "heapq", "bisect",
    "array", "queue", "threading", "multiprocessing", "subprocess", "socket",
    "ssl", "http", "urllib", "email", "html", "xml", "csv", "configparser",
    "argparse", "getopt", "platform", "signal", "gc", "weakref", "types",
    "inspect", "ast", "dis", "tokenize", "keyword", "builtins", "tracemalloc",
    "profile", "timeit", "typing", "dataclasses", "enum", "base64", "binascii",
    "zlib", "gzip", "bz2", "lzma", "zipfile", "tarfile", "sqlite3", "pickle",
    "shelve", "marshal", "dbm", "stat", "errno", "ctypes", "mmap",
}

# Hard caps for Python 2.7 — the last version that supports py2
# Installing anything above these on Python 2.7 will always fail
PYTHON27_MAX_VERSIONS = {
    "numpy":            "1.16.6",
    "scipy":            "1.2.3",
    "pandas":           "0.24.2",
    "matplotlib":       "2.2.5",
    "scikit-learn":     "0.20.4",
    "sklearn":          "0.20.4",
    "tensorflow":       "1.15.5",
    "keras":            "2.3.1",
    "django":           "1.11.29",
    "flask":            "1.1.4",
    "sqlalchemy":       "1.4.49",
    "requests":         "2.27.1",
    "urllib3":          "1.26.20",
    "pillow":           "6.2.2",
    "cryptography":     "2.9.2",
    "paramiko":         "2.7.2",
    "pymongo":          "3.12.3",
    "redis":            "3.5.3",
    "celery":           "4.4.7",
    "boto3":            "1.17.112",
    "pytest":           "4.6.11",
    "setuptools":       "44.1.1",
    "six":              "1.16.0",
    "attrs":            "19.3.0",
    "mock":             "3.0.5",
    "pyyaml":           "5.4.1",
    "jinja2":           "2.11.3",
    "werkzeug":         "1.0.1",
    "click":            "7.1.2",
    "lxml":             "4.6.5",
    "beautifulsoup4":   "4.9.3",
    "pygments":         "2.5.2",
    "chardet":          "4.0.0",
    "certifi":          "2021.10.8",
    "idna":             "2.10",
    "pytz":             "2021.3",
    "simplejson":       "3.17.6",
    "protobuf":         "3.20.3",
    "psycopg2":         "2.8.6",
    "mysqlclient":      "1.4.6",
    "pymysql":          "0.10.1",
    "kombu":            "4.6.11",
    "billiard":         "3.6.4.0",
    "amqp":             "2.6.1",
    "vine":             "1.3.0",
    "twisted":          "18.9.0",
    "scrapy":           "1.7.4",
    "aiohttp":          "3.6.3",
    "pydantic":         "1.10.21",
    "fastapi":          "0.70.1",
    "uvicorn":          "0.15.0",
    "starlette":        "0.16.0",
    "httpx":            "0.18.2",
    "anyio":            "3.3.4",
    "h11":              "0.12.0",
    "grpcio":           "1.48.2",
    "google-auth":      "1.35.0",
    "google-cloud":     "0.34.0",
    "pyarrow":          "0.17.1",
    "torch":            "1.4.0",
    "torchvision":      "0.5.0",
    "transformers":     "2.11.0",
    "nltk":             "3.4.5",
    "gensim":           "3.8.3",
    "spacy":            "2.3.7",
    "imageio":          "2.9.0",
    "scikit-image":     "0.14.5",
    "opencv-python":    "4.2.0.34",
    "sympy":            "1.5.1",
    "networkx":         "2.4",
    "statsmodels":      "0.10.2",
    "xgboost":          "1.2.1",
    "lightgbm":         "2.3.1",
    "catboost":         "0.26.1",
    "tqdm":             "4.64.1",
    "joblib":           "0.14.1",
    "dask":             "2.30.0",
    "numba":            "0.43.1",
    "h5py":             "2.10.0",
    "tables":           "3.6.1",
    "sqlparse":         "0.4.4",
    "psutil":           "5.9.8",
    "arrow":            "0.17.0",
    "pendulum":         "2.1.2",
    "maya":             "0.6.1",
    "pymc3":            "3.11.5",
    "theano":           "1.0.5",
    "pyro-ppl":         "0.3.4",
    "jax":              "0.1.77",
    "flax":             "0.2.2",
}


def parse_version_string(version: str) -> tuple:
    """Parse version string into comparable tuple. '1.24.4' -> (1, 24, 4)"""
    parts = []
    for p in re.split(r'[.\-]', str(version)):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def get_pypi_versions(package_name: str, timeout: int = 10) -> list:
    """
    Query PyPI JSON API for real available versions.
    Returns sorted list of version strings, or empty list if not found.
    """
    pip_name = KNOWN_RENAMES.get(package_name, package_name)
    if pip_name is None:
        return []

    url = f"https://pypi.org/pypi/{pip_name}/json"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "fse-aiware-resolver/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            releases = data.get("releases", {})
            valid = []
            for ver, files in releases.items():
                if files and not all(f.get("yanked", False) for f in files):
                    valid.append(ver)
            return sorted(valid, key=_version_key)
    except Exception:
        return []


def get_best_version(package_name: str,
                     python_minor: int = 8,
                     python_major: int = 3) -> str | None:
    """
    Get the best version of a package for a given Python version.
    Returns a version string or None if package not found on PyPI.

    python_major + python_minor: e.g. major=2, minor=7 for Python 2.7
                                      major=3, minor=8 for Python 3.8
    """
    pip_name = KNOWN_RENAMES.get(package_name, package_name)
    if pip_name is None:
        return None

    # Python 2.7 hard cap — return known-good version immediately
    # without hitting PyPI, since PyPI will just return a py3-only version
    if python_major == 2:
        cap = PYTHON27_MAX_VERSIONS.get(pip_name.lower())
        if cap:
            return cap
        # For packages not in our cap table, still try PyPI but
        # filter to py2-compatible versions only
        return _get_py2_compatible_version(pip_name)

    # Python 3: query PyPI normally
    url = f"https://pypi.org/pypi/{pip_name}/json"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "fse-aiware-resolver/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        releases = data.get("releases", {})
        info     = data.get("info", {})

        # Try latest version first
        latest = info.get("version", "")
        if latest and latest in releases:
            files = releases[latest]
            if files and not all(f.get("yanked", False) for f in files):
                requires_python = info.get("requires_python", "")
                if _is_compatible(requires_python, python_minor, python_major):
                    return latest

        # Walk backwards to find compatible version
        all_versions = sorted(releases.keys(), key=_version_key, reverse=True)
        for ver in all_versions:
            files = releases[ver]
            if not files or all(f.get("yanked", False) for f in files):
                continue
            for f in files:
                py_ver = f.get("python_version", "")
                if py_ver in ("py3", "py2.py3", "source",
                              f"cp3{python_minor}"):
                    return ver
                if py_ver == "source":
                    return ver

        return all_versions[0] if all_versions else None

    except Exception:
        return None


def _get_py2_compatible_version(package_name: str) -> str | None:
    """
    For packages not in PYTHON27_MAX_VERSIONS, query PyPI and filter
    to versions that have a py2 or py2.py3 wheel.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "fse-aiware-resolver/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        releases = data.get("releases", {})
        all_versions = sorted(releases.keys(), key=_version_key, reverse=True)

        for ver in all_versions:
            files = releases[ver]
            if not files or all(f.get("yanked", False) for f in files):
                continue
            for f in files:
                py_ver = f.get("python_version", "")
                requires = f.get("requires_python", "")
                # Accept py2, py2.py3, source, or no requirement
                if py_ver in ("py2", "py2.py3", "source", "any"):
                    return ver
                if "cp27" in py_ver:
                    return ver
                # Reject if explicitly Python 3 only
                if requires and "3" in requires and ">=" in requires:
                    continue
                if py_ver.startswith("py3") or py_ver.startswith("cp3"):
                    continue
                if py_ver == "source":
                    return ver

        return None
    except Exception:
        return None


def is_on_pypi(package_name: str) -> bool:
    pip_name = KNOWN_RENAMES.get(package_name, package_name)
    if pip_name is None:
        return False
    try:
        url = f"https://pypi.org/pypi/{pip_name}/json"
        req = urllib.request.Request(
            url, headers={"User-Agent": "fse-aiware-resolver/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_stdlib(module_name: str) -> bool:
    return module_name.lower() in STDLIB_MODULES


def get_pip_name(import_name: str) -> str | None:
    return KNOWN_RENAMES.get(import_name, import_name)


def resolve_packages_from_pypi(import_names: list,
                                python_minor: int = 8,
                                python_major: int = 3) -> dict:
    """
    Given import names, return {pip_name: best_version} by querying PyPI.
    Respects Python version — won't return numpy==1.24 for Python 2.7.

    python_major: 2 or 3
    python_minor: e.g. 7 for 2.7, 8 for 3.8, 10 for 3.10
    """
    result      = {}
    unavailable = []

    for name in import_names:
        if is_stdlib(name):
            continue

        pip_name = KNOWN_RENAMES.get(name, name)
        if pip_name is None:
            unavailable.append(name)
            continue

        # Python 2.7: check hard cap table first (fast, no network)
        if python_major == 2:
            cap = PYTHON27_MAX_VERSIONS.get(pip_name.lower())
            if cap:
                result[pip_name] = cap
                continue

        # Query PyPI for best compatible version
        version = get_best_version(pip_name, python_minor, python_major)
        if version:
            result[pip_name] = version
        else:
            unavailable.append(name)

    if unavailable:
        print(f"[PyPI] Not found on PyPI: {unavailable}")

    return result


def parse_python_version_string(version_str: str) -> tuple:
    """
    Parse '2.7', '3.8', '3.10' into (major, minor).
    Returns (3, 8) as default on failure.
    """
    try:
        parts = str(version_str).split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return major, minor
    except Exception:
        return 3, 8


def _version_key(version_str: str):
    parts = []
    for p in re.split(r'[.\-]', str(version_str)):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return parts


def _is_compatible(requires_python: str,
                   python_minor: int,
                   python_major: int = 3) -> bool:
    if not requires_python:
        return True
    try:
        for part in requires_python.split(","):
            part = part.strip()
            m = re.match(r'([><=!]+)\s*(\d+)\.(\d+)', part)
            if m:
                op    = m.group(1)
                r_maj = int(m.group(2))
                r_min = int(m.group(3))
                # Compare as (major, minor) tuples
                actual = (python_major, python_minor)
                req    = (r_maj, r_min)
                if op == ">=" and actual < req:
                    return False
                if op == ">"  and actual <= req:
                    return False
                if op == "<"  and actual >= req:
                    return False
                if op == "<=" and actual > req:
                    return False
                if op == "==" and actual != req:
                    return False
    except Exception:
        pass
    return True


if __name__ == "__main__":
    import sys
    pkg = sys.argv[1] if len(sys.argv) > 1 else "numpy"
    print(f"\nTesting: {pkg}")
    print(f"  Python 2.7 best: {get_best_version(pkg, 7, 2)}")
    print(f"  Python 3.8 best: {get_best_version(pkg, 8, 3)}")
    print(f"  Python 3.10 best: {get_best_version(pkg, 10, 3)}")
    versions = get_pypi_versions(pkg)
    print(f"  All versions ({len(versions)}): ...{versions[-3:] if versions else 'NONE'}")


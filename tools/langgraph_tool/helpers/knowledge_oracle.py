"""
Knowledge Oracle

Loads historical PLLM results and provides instant gist-ID lookup.
If a snippet was already solved in a previous run, we replay that
solution directly — no Docker build, no LLM call needed.

Ported and simplified from SmartResolver's knowledge_oracle.py.
"""

import csv
import os
import re
from typing import Dict, List, Optional


class KnowledgeOracle:

    def __init__(self, results_dir: str = None, logging: bool = True):
        self.logging = logging
        self.gist_solutions: Dict[str, dict] = {}
        # Session cache: solutions found during THIS run get stored here
        # so later snippets can benefit from earlier ones
        self.session_solutions: Dict[str, dict] = {}

        if results_dir:
            self._load(results_dir)

    def _log(self, msg: str):
        if self.logging:
            print(f"  [Oracle] {msg}", flush=True)

    def _load(self, results_dir: str):
        """Load PLLM results CSV. Tries several path patterns."""
        candidates = [
            os.path.join(results_dir, 'pllm_results', 'csv', 'summary-all-runs.csv'),
            os.path.join(results_dir, 'csv', 'summary-all-runs.csv'),
            os.path.join(results_dir, 'summary-all-runs.csv'),
        ]
        summary_file = next((p for p in candidates if os.path.exists(p)), None)

        if not summary_file:
            self._log(f"No pllm results CSV found under {results_dir} — oracle disabled")
            return

        count = 0
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gist_id = row.get('name', '').strip()
                    if not gist_id:
                        continue

                    py_ver = self._extract_version(row.get('file', ''))
                    packages = self._parse_modules(row.get('python_modules', ''))
                    try:
                        confidence = int(row.get('passed', 0))
                    except (ValueError, TypeError):
                        confidence = 0

                    self.gist_solutions[gist_id] = {
                        'python_version': py_ver,
                        'packages': packages,
                        'confidence': confidence,
                        'result': row.get('result', ''),
                    }
                    count += 1

            self._log(f"Loaded {count} historical solutions from {summary_file}")
        except Exception as e:
            self._log(f"Error loading results: {e}")

    def _extract_version(self, file_field: str) -> str:
        match = re.search(r'(\d+\.\d+)', file_field)
        return match.group(1) if match else '3.8'

    def _parse_modules(self, modules_str: str) -> List[str]:
        if not modules_str or modules_str.strip().lower() in ('', 'none'):
            return []
        return [m.strip() for m in modules_str.split(';') if m.strip()]

    def lookup(self, gist_id: str) -> Optional[dict]:
        """
        Look up a gist. Returns a solution dict or None.

        A returned dict always has:
          python_version: str
          packages: list[str]   (module names only, no versions)
          confidence: int       (0-10, from PLLM pass score)

        Optionally has:
          package_versions: dict  (module → version, if we recorded one this session)
          oracle_hit: bool        (True = skip Docker entirely)
          hint_only: bool         (True = use as starting point, still needs build)
        """
        # Session solutions take priority (learned this run)
        if gist_id in self.session_solutions:
            sol = dict(self.session_solutions[gist_id])
            sol['oracle_hit'] = True
            return sol

        if gist_id not in self.gist_solutions:
            return None

        sol = self.gist_solutions[gist_id]
        confidence = sol['confidence']

        if confidence >= 4 and sol['packages']:
            # High confidence + known packages → replay directly
            return {
                'python_version': sol['python_version'],
                'packages': sol['packages'],
                'confidence': confidence,
                'oracle_hit': True,
            }
        elif confidence >= 4 and not sol['packages']:
            # High confidence, no external deps → just set the Python version
            return {
                'python_version': sol['python_version'],
                'packages': [],
                'confidence': confidence,
                'oracle_hit': True,
                'empty_deps': True,
            }
        elif confidence == 0 and sol['packages']:
            # Known failure but gives us a useful starting point (hint)
            return {
                'python_version': sol['python_version'],
                'packages': sol['packages'],
                'confidence': 0,
                'oracle_hit': False,
                'hint_only': True,
            }

        return None

    def record_success(self, gist_id: str, python_version: str,
                       package_versions: Dict[str, str]):
        """
        Store a solution found this session so future snippets can reuse it.
        """
        if package_versions:
            self.session_solutions[gist_id] = {
                'python_version': python_version,
                'packages': list(package_versions.keys()),
                'package_versions': dict(package_versions),
                'confidence': 10,
                'oracle_hit': True,
            }

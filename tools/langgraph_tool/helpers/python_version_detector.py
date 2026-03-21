"""
Python Version Detector

Analyses Python source code to determine whether it is Python 2 or 3,
and which minor version is most likely.

Ported from SmartResolver (other competition entry) with minor adjustments.
Uses weighted regex scoring rather than asking the LLM to guess — more
reliable for Python 2 code which Gemma2 often misclassifies as Python 3.
"""

import re


class PythonVersionDetector:

    # Python 2 patterns with confidence weights
    PY2_INDICATORS = {
        r'#!/usr/bin/env python2':    100,
        r'#!/usr/bin/python2':        100,
        r'\bprint\s+["\']':            15,   # print "hello"
        r'\bprint\s+[^(\n]':           10,   # print something (no parens)
        r'\.iteritems\(\)':            15,
        r'\.itervalues\(\)':           15,
        r'\.iterkeys\(\)':             15,
        r'\bxrange\s*\(':              15,
        r'\.has_key\s*\(':             15,
        r'\bexecfile\s*\(':            15,
        r'\braw_input\s*\(':           15,
        r'\bunicode\s*\(':             10,
        r'\bbasestring\b':             10,
        r'\burllib2\b':                20,
        r'\burlparse\b':               15,
        r'\bConfigParser\b':           10,
        r'\bSocketServer\b':           10,
        r'\bSimpleHTTPServer\b':       15,
        r'\bBaseHTTPServer\b':         15,
        r'\bhttplib\b':                15,
        r'\bcookielib\b':              10,
        r'except\s+\w+\s*,\s*\w+':    15,   # except Exception, e:
        r'raise\s+\w+\s*,':           10,    # raise Exception, "msg"
        r'from\s+__future__\s+import':  5,   # py2 compat shim
    }

    # Python 3 patterns with confidence weights
    PY3_INDICATORS = {
        r'#!/usr/bin/env python3':    100,
        r'#!/usr/bin/python3':        100,
        r'\basync\s+def\b':            20,
        r'\bawait\s+':                 20,
        r'\basync\s+for\b':            20,
        r'\basync\s+with\b':           20,
        r'f"[^"]*\{':                  15,   # f-strings
        r"f'[^']*\{":                  15,
        r'\bnonlocal\s+':              10,
        r'->\s*\w+':                    5,   # return type hints
        r'\burllib\.request\b':        15,
        r'\burllib\.parse\b':          15,
        r'\bconfigparser\b':           10,
        r'\bpathlib\b':                10,
        r'\bdataclasses\b':            15,
        r'from\s+typing\s+import':     10,
        r'\bbreakpoint\s*\(':          10,
        r':=':                         15,   # walrus operator (3.8+)
    }

    def detect(self, code: str) -> str:
        """
        Detect Python version from source code.
        Returns '2.7', '3.7', or '3.8'.
        """
        version, _ = self.detect_with_confidence(code)
        return version

    def detect_with_confidence(self, code: str) -> tuple:
        """
        Returns (version_str, confidence) where confidence is 'high', 'medium', or 'low'.
        """
        py2_score = 0
        py3_score = 0

        for pattern, weight in self.PY2_INDICATORS.items():
            if re.search(pattern, code):
                py2_score += weight

        for pattern, weight in self.PY3_INDICATORS.items():
            if re.search(pattern, code):
                py3_score += weight

        total = py2_score + py3_score
        diff = abs(py2_score - py3_score)

        if py2_score > py3_score:
            version = '2.7'
        else:
            # Check for 3.8+ specific features
            if re.search(r':=', code) or re.search(r'\bdataclasses\b', code):
                version = '3.8'
            elif re.search(r'\basync\s+def\b', code) or re.search(r'\bawait\s+', code):
                version = '3.7'
            else:
                version = '3.8'  # safe default

        if total == 0:
            confidence = 'low'
        elif diff > 30:
            confidence = 'high'
        elif diff > 10:
            confidence = 'medium'
        else:
            confidence = 'low'

        return version, confidence

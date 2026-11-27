"""HPC runs Python 3.11.18. Fail on syntax that only parses on 3.12+.

Two layers:
1. If a Python 3.11 interpreter is available (the running one, or ``python3.11`` on PATH),
   byte-compile every hpc/*.py and tests/*.py with it. This is the authoritative check.
2. On 3.12+ also scan f-strings with ``tokenize`` for PEP 701 constructs that 3.11 rejects:
   a nested string reusing the enclosing f-string's quote, a backslash inside a replacement
   field, or a comment inside a replacement field. ``ast.parse(feature_version=(3, 11))`` does
   not catch these, which is how the a40_baselines.py break slipped through.
"""
import shutil
import subprocess
import sys
import tokenize
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(list((ROOT / "hpc").glob("*.py")) + list((ROOT / "tests").glob("*.py")))


def _quote(tok_string):
    s = tok_string.lstrip("rRbBfFuU")
    return s[:3] if s[:3] in ('"""', "'''") else s[:1]


def pep701_violations(path):
    """Return (line, reason) for f-string syntax 3.11 rejects. Needs Python 3.12+ tokenize."""
    out = []
    stack = []  # [quote, brace_depth]
    with open(path, "rb") as fh:
        toks = list(tokenize.tokenize(fh.readline))
    for tok in toks:
        name = tokenize.tok_name[tok.type]
        inside = any(depth > 0 for _, depth in stack)
        if name == "FSTRING_START":
            q = _quote(tok.string)
            if inside and any(q == oq or q in oq or oq in q for oq, d in stack if d > 0):
                out.append((tok.start[0], "nested f-string reuses an enclosing quote"))
            stack.append([q, 0])
        elif name == "FSTRING_END":
            stack.pop()
        elif not stack:
            continue
        elif name == "OP" and tok.string in "{[(":
            stack[-1][1] += tok.string == "{"
        elif name == "OP" and tok.string in "}])":
            stack[-1][1] -= tok.string == "}" and stack[-1][1] > 0
        elif inside and name == "STRING":
            q = _quote(tok.string)
            if any(q == oq or q in oq or oq in q for oq, d in stack if d > 0):
                out.append((tok.start[0], "string in replacement field reuses the f-string quote"))
            if "\\" in tok.string:
                out.append((tok.start[0], "backslash in replacement field"))
        elif inside and name == "COMMENT":
            out.append((tok.start[0], "comment in replacement field"))
    return out


def _python311():
    if sys.version_info[:2] == (3, 11):
        return sys.executable
    return shutil.which("python3.11")


class Py311SyntaxTest(unittest.TestCase):
    def test_compiles_under_python311(self):
        exe = _python311()
        if exe is None:
            self.skipTest("no Python 3.11 interpreter; the tokenize scan still runs")
        # compile() in memory, so no .pyc is written next to the sources
        code = ("import sys\nbad = []\nfor p in sys.argv[1:]:\n"
                "    try: compile(open(p, 'rb').read(), p, 'exec')\n"
                "    except SyntaxError as e: bad.append(f'{e.filename}:{e.lineno}: {e.msg}')\n"
                "print(chr(10).join(bad)); sys.exit(1 if bad else 0)\n")
        res = subprocess.run([exe, "-c", code, *map(str, FILES)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)

    @unittest.skipIf(sys.version_info < (3, 12), "FSTRING tokens exist only on 3.12+")
    def test_no_pep701_fstrings(self):
        bad = [f"{p.relative_to(ROOT)}:{ln}: {why}" for p in FILES for ln, why in pep701_violations(p)]
        self.assertEqual(bad, [], "\n".join(bad))

    @unittest.skipIf(sys.version_info < (3, 12), "FSTRING tokens exist only on 3.12+")
    def test_scanner_catches_known_breaks(self):
        import tempfile
        cases = {
            "x = {'a': 1}\ny = f\"{f'{x['a']}'}\"\n": True,   # the a40_baselines.py bug
            "x = {'a': 1}\ny = f\"{x[\"a\"]}\"\n": True,
            "x = 1\ny = f\"{'\\n'.join([])}\"\n": True,
            "x = {'a': 1}\ny = f\"{x['a']:.1f} {'-' if x else 'b'}\"\n": False,
            "y = f'{{literal}}' + f\"{1:>{3}}\"\n": False,
        }
        for src, expect in cases.items():
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
                fh.write(src)
            self.assertEqual(bool(pep701_violations(fh.name)), expect, src)
            Path(fh.name).unlink()


if __name__ == "__main__":
    unittest.main()

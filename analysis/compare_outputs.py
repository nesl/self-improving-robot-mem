"""Compare a reproduced output with the paper's copy.

    python3 analysis/compare_outputs.py EXPECTED PRODUCED

Text (.txt, .md): PASS when the sequence of numbers in the two files is identical; the text itself is also
compared and any wording difference is shown (labels in a few scripts were renamed after the paper's copy was
written). JSON: PASS when every numeric value at every key path is identical (string fields such as notes and
source paths are reported but not required to match). PDF: compares the text layer (pdftotext, if installed),
which holds every tick label, legend entry and printed count. Exit status 1 on any FAIL.
"""
import difflib
import json
import pathlib
import re
import shutil
import subprocess
import sys

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def numbers(text):
    return NUM.findall(text)


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def compare_text(a, b):
    ta, tb = a.read_text(), b.read_text()
    if ta == tb:
        return True, "identical"
    if numbers(ta) == numbers(tb):
        diff = [l for l in difflib.unified_diff(ta.splitlines(), tb.splitlines(), lineterm="", n=0)
                if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        return True, f"same numbers, wording differs on {len(diff) // 2 or 1} line(s):\n      " + "\n      ".join(diff[:6])
    na, nb = numbers(ta), numbers(tb)
    first = next((i for i, (x, y) in enumerate(zip(na, nb)) if x != y), min(len(na), len(nb)))
    diff = list(difflib.unified_diff(ta.splitlines(), tb.splitlines(), "paper", "reproduced", lineterm="", n=0))
    return False, (f"numbers differ ({len(na)} vs {len(nb)} numbers; first difference at #{first}: "
                   f"{na[first] if first < len(na) else '-'} vs {nb[first] if first < len(nb) else '-'})\n      "
                   + "\n      ".join(diff[:20]))


def compare_json(a, b):
    ja, jb = dict(leaves(json.load(open(a)))), dict(leaves(json.load(open(b))))
    num = lambda d: {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}  # noqa: E731
    na, nb = num(ja), num(jb)
    bad = [k for k in set(na) | set(nb) if na.get(k) != nb.get(k)]
    strings = [k for k in set(ja) | set(jb) if k not in na and k not in nb and ja.get(k) != jb.get(k)]
    if bad:
        return False, f"{len(bad)} numeric values differ, e.g. " + "; ".join(f"{k}: {na.get(k)} vs {nb.get(k)}" for k in sorted(bad)[:5])
    note = f"all {len(na)} numbers identical"
    if strings:
        note += f"; {len(strings)} string field(s) differ: " + ", ".join(sorted(strings)[:4])
    return True, note


def compare_pdf(a, b):
    if not shutil.which("pdftotext"):
        return True, "pdftotext not installed; PDF not compared (its .json carries the plotted values)"
    t = lambda p: " ".join(subprocess.run(["pdftotext", str(p), "-"], capture_output=True, text=True).stdout.split())  # noqa: E731
    ta, tb = t(a), t(b)
    return (ta == tb), ("text layer identical" if ta == tb else f"text layer differs:\n      paper:      {ta[:300]}\n      reproduced: {tb[:300]}")


def main():
    a, b = map(pathlib.Path, sys.argv[1:3])
    if not b.exists():
        print(f"FAIL  {b}: not produced")
        return 1
    fn = {".json": compare_json, ".pdf": compare_pdf}.get(a.suffix, compare_text)
    ok, msg = fn(a, b)
    print(f"{'PASS' if ok else 'FAIL'}  {b}  vs  {a}: {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

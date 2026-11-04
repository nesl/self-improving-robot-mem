"""Check that rerun cells reproduce the shipped ones, search row by search row.

    python3 analysis/compare_cells.py RERUN_CELLS SHIPPED_CELLS ARM_DIR HOME [HOME ...]
"""
import json
import pathlib
import sys


def rows(p):
    return [r for r in map(json.loads, open(p / "searches.jsonl")) if r.get("kind") == "search"]


rerun, shipped, arm, homes = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
bad = 0
for h in homes:
    a, b = rows(rerun / arm / h), rows(shipped / arm / h)
    same = a == b
    bad += not same
    print(f"{'PASS' if same else 'FAIL'}  rerun {arm}/{h}: {len(a)} searches, "
          f"{'identical to the shipped cell' if same else 'DIFFERENT from the shipped cell'}")
sys.exit(1 if bad else 0)

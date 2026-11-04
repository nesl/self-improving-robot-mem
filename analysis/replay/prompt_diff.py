#!/usr/bin/env python3
"""Diff the day-1 search, answer and night-1 update prompts of a new arm against `incremental edits` on hh_s2_t03.

    PYTHONPATH=src python3 analysis/replay/prompt_diff.py \\
        "claim store plus the log" OUT.md

No model calls: the client records each prompt and returns nothing, so both arms fall back to the
same rotation and the first prompts of day 1 are comparable. Cells are written to a scratch dir.
"""
import difflib
import json
import pathlib
import sys
import tempfile

from self_improve import overnight_wave as ow
from self_improve.frozen_household import FrozenHousehold

BANKS = pathlib.Path("results/self_improve/varied_homes/ten_homes/banks")


class Recorder:
    def __init__(self):
        self.calls = []
        self.model = "recorder"
        self.stats = {"calls": 0, "cached": 0}

    def complete(self, messages, schema, max_tokens):
        self.calls.append((messages, schema, max_tokens))
        return None, {}


def first_prompts(arm):
    rec = Recorder()
    with tempfile.TemporaryDirectory() as d:
        try:
            ow.run_one_arm(FrozenHousehold(BANKS / "hh_s2_t03.jsonl"), arm, rec,
                           pathlib.Path(d) / "wave" / "cells" / "arm" / "hh", 1, 8, 3, 0)
        except AssertionError:
            pass   # the gates refuse a cell whose calls all failed; only the prompts matter here
    props = lambda sc: set((sc or {}).get("properties", {}))  # noqa: E731
    choice = next(c for c in rec.calls if "room" in props(c[1]))
    answer = next(c for c in rec.calls if "location" in props(c[1]))
    night = [c for c in rec.calls if "room" not in props(c[1]) and "location" not in props(c[1])][-1]
    return choice, answer, night


def show(messages):
    return "\n".join(f"[{m['role']}]\n{m['content']}" for m in messages).splitlines()


def main() -> int:
    arm, out = sys.argv[1], pathlib.Path(sys.argv[2])
    base = first_prompts("incremental edits")
    new = first_prompts(arm)
    parts = [f"# Day-1 prompts: `{arm}` against `incremental edits`, hh_s2_t03, 8 a day\n"]
    for name, b, n in (("search (room choice)", base[0], new[0]), ("answer", base[1], new[1]),
                       ("nightly update (night 1)", base[2], new[2])):
        diff = list(difflib.unified_diff(show(b[0]), show(n[0]), "incremental edits", arm, lineterm="", n=2))
        parts += [f"## {name}", f"schema identical: {b[1] == n[1]}; max_tokens identical: {b[2] == n[2]}", "",
                  "```diff", *diff, "```", ""]
    out.write_text("\n".join(parts))
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())

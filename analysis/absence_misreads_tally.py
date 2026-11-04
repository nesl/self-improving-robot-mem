"""Tally of the hand coding of 40 wrong first-room choices (E6, "readers misjudge absences too").

The sample (`absence_misreads/sample.json`) was built by `analysis/absence_misreads_sample.py`: seed 0, wrong first
rooms during and after the illness, at most 3 per home, 20 per arm (history only = `wave_log_only`, history and notes
= `overnight_wave`), ten homes at 8 a day, Qwen. `coding.json` holds one independent coder's category and evidence
for each item: A = misreads or ignores where the object had NOT been found (failed looks), B = misreads something
else, C = a sound reading that was still wrong. The finer splits in the paper (5 of 7, 7 of 7) come from reading
the `evidence` text of the A items, not from a field.

    python3 analysis/absence_misreads_tally.py
"""
import collections
import json
import pathlib

P = pathlib.Path("results/self_improve/paper/absence_misreads")
coding = json.load(open(P / "coding.json"))
sample = {x["id"]: x for x in json.load(open(P / "sample.json"))}
assert set(sample) == {c["id"] for c in coding} and len(coding) == 40
arm = {"LO": "history only (log_only_no_notes)", "LN": "history and notes (the_log_and_notes_about_the_routine)"}
count = collections.Counter((c["id"][:2], c["category"]) for c in coding)
print(__doc__)
for k, name in arm.items():
    print(f"{name:58s} A {count[k, 'A']:2d}   B {count[k, 'B']:2d}   C {count[k, 'C']:2d}   of {sum(v for (a, _), v in count.items() if a == k)}")
tot = collections.Counter(c["category"] for c in coding)
print(f"{'both arms':58s} A {tot['A']:2d}   B {tot['B']:2d}   C {tot['C']:2d}   of {len(coding)}")

#!/usr/bin/env python3
"""Table 1 of the draft: six methods on the ten-household run, and the paired differences.

Arm directories from questions_answered.md. "First room right" is found_at_step == 1 and "found
within 3 rooms" is found_it; both are defined on all 2,480 rows of every arm, so the denominator
does not move between methods (exact-place accuracy does - see scored_counts.py).

    python3 analysis/table1_ten_homes.py
"""
import glob
import json
import statistics

WAVE = "results/self_improve/overnight_wave/cells"
NAME = {"the_log_and_notes_about_the_routine": "log and notes",
        "last_seen_no_model": "last-seen",
        "incremental_edits": "claim store",
        "claim_store_told_if_it_was_right": "reduced ACE",
        "a_small_working_memory_and_an_archive": "small working memory",
        "prior_only_no_notes": "notes hidden"}
PAIRS = [("last_seen_no_model", "incremental_edits"),
         ("last_seen_no_model", "claim_store_told_if_it_was_right"),
         ("last_seen_no_model", "a_small_working_memory_and_an_archive"),
         ("the_log_and_notes_about_the_routine", "last_seen_no_model")]
MEAS = {"first room right": lambda r: r.get("found_at_step") == 1,
        "found within 3": lambda r: bool(r.get("found_it"))}
FLOOR = {"first room right": 1.9, "found within 3": 2.2}


def rows(arm, home):
    out = []
    for line in open(f"{WAVE}/{arm}/{home}/searches.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "search":
            out.append(r)
    return out


def main() -> int:
    homes = sorted(p.split("/")[-1] for p in glob.glob(f"{WAVE}/last_seen_no_model/*"))
    data = {a: {h: rows(a, h) for h in homes} for a in NAME}
    print(f"ten households: {', '.join(homes)}\n")
    print(f"{'method':24s}" + "".join(f"{m:>18s}" for m in MEAS)
          + "   questions per method")
    for arm, label in NAME.items():
        line = f"{label:24s}"
        n = sum(len(data[arm][h]) for h in homes)
        for m, fn in MEAS.items():
            pooled = 100 * sum(1 for h in homes for r in data[arm][h] if fn(r)) / n
            line += f"{pooled:18.1f}"
        print(line + f"   {n}")
    print("\npooled over all questions above; per household below\n")
    for m, fn in MEAS.items():
        print(f"=== {m}: paired difference, 2 SE across the ten households, floor {FLOOR[m]}")
        for a, b in PAIRS:
            d = []
            for h in homes:
                x = 100 * sum(1 for r in data[a][h] if fn(r)) / len(data[a][h])
                y = 100 * sum(1 for r in data[b][h] if fn(r)) / len(data[b][h])
                d.append(x - y)
            mean = statistics.mean(d)
            tse = 2 * statistics.stdev(d) / len(d) ** 0.5
            same = sum(1 for v in d if v > 0)
            verdict = "clears" if abs(mean) > tse and abs(mean) > FLOOR[m] else "does NOT clear"
            print(f"   {NAME[a]} minus {NAME[b]:22s} {mean:+6.1f}  2 SE {tse:5.1f}  "
                  f"same sign in {same} of {len(d)}  {verdict}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The log-only arm beside the ten-household table: is the raw record enough on its own?

Same households, same banks, same seed, same eight questions a day as Table 1. The arm reads the
control's search-time prompt with the object's raw log in it and an empty notes block, and never
writes a note. So the difference between it and `log and notes` is exactly what the summaries add
on top of the record, and the difference between it and `last-seen` is what the model's reading of
the record adds over a rule that walks back through it.

Anything not finished is named and left out; partial households are never pooled with anything.

    PYTHONPATH=src python3 analysis/log_only_row.py
"""
import json
import pathlib
import statistics
import sys

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_data as D                                              # noqa: E402

TEN = pathlib.Path("results/self_improve/overnight_wave/cells")
LOG_ONLY = pathlib.Path("results/self_improve/wave_log_only/cells")
ARM = "log_only_no_notes"
OTHERS = {"log and notes": "the_log_and_notes_about_the_routine",
          "last-seen": "last_seen_no_model", "claim store": "incremental_edits"}
MEAS = (("first room right", "first", 1.9), ("found within 3", "found", 2.2))


def finished(home):
    c = LOG_ONLY / ARM / home / "cell.json"
    if not c.exists():
        return False
    d = json.load(open(c))
    return d.get("last_day") == 31 and (LOG_ONLY / ARM / home / "searches.jsonl").exists()


def main() -> int:
    done = [h for h in D.HOMES_10 if finished(h)]
    missing = [h for h in D.HOMES_10 if h not in done]
    print(f"log-only finished in {len(done)} of {len(D.HOMES_10)} households")
    if missing:
        print(f"  NOT finished, and not pooled with anything: {', '.join(missing)}")
    if not done:
        return 0
    print(f"  households used: {', '.join(done)}\n")

    for label, key, floor in MEAS:
        pooled = []
        for home in done:
            pooled += D.rows(LOG_ONLY, ARM, home)
        print(f"=== {label}")
        print(f"  {'log only, no notes':24s} {D.share(pooled, key):6.1f}   "
              f"{len(pooled)} questions")
        for name, arm_dir in OTHERS.items():
            other = []
            for home in done:
                other += D.rows(TEN, arm_dir, home)
            print(f"  {name:24s} {D.share(other, key):6.1f}   {len(other)} questions")
        print(f"  --- paired within household, 2 SE across the {len(done)} households, "
              f"floor {floor}")
        for name, arm_dir in OTHERS.items():
            diffs = []
            for home in done:
                a = D.share(D.rows(LOG_ONLY, ARM, home), key)
                b = D.share(D.rows(TEN, arm_dir, home), key)
                if a is not None and b is not None:
                    diffs.append(a - b)
            mean = statistics.mean(diffs)
            two_se = (2 * statistics.stdev(diffs) / len(diffs) ** 0.5) if len(diffs) > 1 else 0
            same = sum(1 for d in diffs if (d > 0) == (mean > 0))
            clears = abs(mean) > two_se and abs(mean) > floor
            print(f"    log only minus {name:16s} {mean:+6.1f}  2 SE {two_se:5.1f}  "
                  f"same sign {same} of {len(diffs)}  "
                  f"{'CLEARS' if clears else 'does not clear'}")
            if not clears and abs(mean) <= floor:
                print(f"      {abs(mean):.1f} is inside the {floor} noise floor: on this "
                      f"measure the two are indistinguishable")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

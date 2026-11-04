#!/usr/bin/env python3
"""Camera-ready report for the new arms: first room right and found within three rooms, overall and on
moved objects, by window, with paired differences per household against last-seen and the claim store.

    PYTHONPATH=src python3 analysis/camera_ready_report.py > OUT.txt

Measures and pairing are the paper's: MEAS from table1_ten_homes.py (found_at_step == 1, found_it),
per-household shares, the mean paired difference and 2 SE = 2 stdev / sqrt(households).
An arm or comparator that has not run on a setting is skipped and named.
"""
import importlib.util
import json
import pathlib
import statistics
import sys

R = pathlib.Path("results/self_improve")
spec = importlib.util.spec_from_file_location("table1", pathlib.Path(__file__).resolve().parent / "table1_ten_homes.py")
table1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table1)
MEAS = table1.MEAS

TEN = ["hh_s2_t03", "hh_s19_t03", "hh_s20_t03", "hh_s32_t03", "hh_s48_t03", "hh_s63_t03",
       "hh_s93_t03", "hh_s109_t03", "hh_s123_t03", "hh_s151_t03"]
FIVE = ["hh_s32_t03", "hh_s48_t03", "hh_s63_t03", "hh_s93_t03", "hh_s151_t03"]
THREE = ["hh_s2_t03", "hh_s32_t03", "hh_s48_t03"]
W31 = {"days 1-13": range(1, 14), "day 14": [14], "days 14-23": range(14, 24), "day 24": [24],
       "days 24-31": range(24, 32), "days 1-31": range(1, 32)}
W49 = {**W31, "day 32": [32], "days 32-41": range(32, 42), "days 42-49": range(42, 50),
       "days 1-49": range(1, 50)}
SETTINGS = [  # (name, wave dirs by arm (default first), homes, windows)
    ("ten households, 8 a day, 31 days", {"": "overnight_wave", "log_only_no_notes": "wave_log_only"}, TEN, W31),
    ("three households, 24 a day, 50 days", {"": "wave_the_second_illness"}, THREE, W49),
    ("five households, 4 a day", {"": "wave_the_budget_sweep_q4"}, FIVE, W31),
    ("five households, 8 a day", {"": "wave_the_budget_sweep_q8"}, FIVE, W31),
    ("five households, 24 a day", {"": "wave_wider_five"}, FIVE, W31),
]
NEW = ["last_seen_updated_each_night", "claim_store_plus_the_log", "claim_store_plus_today's_sightings",
       "log_only_no_notes"]
BASE = {"last-seen": "last_seen_no_model", "claim store": "incremental_edits"}


def rows(waves, arm, home):
    f = R / waves.get(arm, waves[""]) / "cells" / arm / home / "searches.jsonl"
    if not f.exists():
        return None
    return [r for r in (json.loads(l) for l in f.open()) if r.get("kind") == "search"]


def share(rs, fn):
    return None if not rs else 100 * sum(1 for r in rs if fn(r)) / len(rs)


def main() -> int:
    for name, waves, homes, wins in SETTINGS:
        print(f"\n{'#' * 100}\n# {name}\n{'#' * 100}")
        arms = [a for a in NEW + list(BASE.values())
                if all(rows(waves, a, h) is not None for h in homes)]
        missing = [a for a in NEW if a not in arms]
        if missing:
            print("not run on this setting (or incomplete): " + ", ".join(missing))
        for movers in (False, True):
            print(f"\n=== {'MOVED OBJECTS ONLY' if movers else 'EVERY QUESTION'}")
            data = {a: {h: [r for r in rows(waves, a, h) if r.get("is_a_mover") or not movers]
                        for h in homes} for a in arms}
            for m, fn in MEAS.items():
                print(f"\n--- {m}: pooled share (%) by window")
                print(f"{'arm':38s}" + "".join(f"{w:>12s}" for w in wins))
                for a in arms:
                    line = f"{a:38s}"
                    for w, days in wins.items():
                        rs = [r for h in homes for r in data[a][h] if r["day"] in days]
                        v = share(rs, fn)
                        line += f"{'' if v is None else f'{v:.1f}':>12s}"
                    print(line)
                for a in arms:
                    if a not in NEW:
                        continue
                    for bname, b in BASE.items():
                        if b not in arms:
                            continue
                        print(f"\n  {a} minus {bname}: mean paired difference over {len(homes)} households, 2 SE")
                        print(f"  {'window':14s}{'diff':>8s}{'2 SE':>8s}   per household")
                        for w, days in wins.items():
                            d = []
                            for h in homes:
                                x = share([r for r in data[a][h] if r["day"] in days], fn)
                                y = share([r for r in data[b][h] if r["day"] in days], fn)
                                if x is not None and y is not None:
                                    d.append(x - y)
                            if len(d) < 2:
                                continue
                            se2 = 2 * statistics.stdev(d) / len(d) ** 0.5
                            print(f"  {w:14s}{statistics.mean(d):+8.1f}{se2:8.1f}   "
                                  + " ".join(f"{v:+.0f}" for v in d))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""The shape of a household, read from the frozen question banks the study actually ran on.

The numbers a methods section needs: residents, rooms, places, objects, the objects the questions
ask about, and how many of those change room while a resident is ill. Everything here is counted
from the bank file, never from a docstring - the loader's own docstring says "6 to 9 rooms" and
"70-odd objects" and only one of those turns out to be right for these households.

    PYTHONPATH=src python3 analysis/what_a_household_is_made_of.py
"""
import collections
import pathlib
import statistics
import sys

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_data as D                                              # noqa: E402
from self_improve import search_driven as sd                        # noqa: E402

WIDER_5 = ("hh_s32_t03", "hh_s48_t03", "hh_s63_t03", "hh_s93_t03", "hh_s151_t03")
SETS = (("ten homes, 31 days", "ten_homes", D.HOMES_10),
        ("three homes, 49 days", "all_generated_v2_twice", D.HOMES_50),
        ("five wider homes, 31 days", "headline_five", WIDER_5))


def spread(values):
    """min to max, and the mean - a range is what a methods section should print."""
    lo, hi = min(values), max(values)
    mean = statistics.mean(values)
    return f"{lo} to {hi}" if lo != hi else f"{lo}", f"{mean:.1f}"


def main() -> int:
    every_home = {}
    for label, banks_name, homes in SETS:
        # headline_five/banks held links to the same five files in all_generated_v3/banks; the release ships those files
        bank_dir = {"headline_five": "all_generated_v3"}.get(banks_name, banks_name)
        banks = pathlib.Path(f"results/self_improve/varied_homes/{bank_dir}/banks")
        print(f"\n===== {label}   ({len(homes)} households, banks {banks_name})")
        rows = []
        for name in homes:
            h = D.home(name, banks)
            # every object with a place history in the bank, which is every object the simulator
            # moves - not only the ones the questions ask about
            objects = set(h.moves) | set(h.object_class)
            # an object "moves" if its place history has more than one entry: the rest are the
            # fixed things of the house, which the bank records once and never touches again
            ever_moves = {o for o, hist in h.moves.items() if len({p for _, p in hist}) > 1}
            per_day_moves = len([1 for hist in h.moves.values() for _ in hist[1:]]) / h.n_days
            movers = sd.the_movers(h)
            days = sorted({q["day_index"] for q in h.questions})
            per_day = collections.Counter(q["day_index"] for q in h.questions)
            rows.append(dict(name=name, residents=len(h.resident_ids), rooms=len(h.rooms),
                             places=len(h.places), objects=len(objects),
                             ever_moves=len(ever_moves), moves_a_day=per_day_moves,
                             asked=len(h.asked_objects), movers=len(movers),
                             days=len(days), questions=len(h.questions),
                             per_day=statistics.mean(per_day.values())))
            every_home[(banks_name, name)] = rows[-1]
        print(f"  {'household':14s} {'residents':>9s} {'rooms':>6s} {'places':>7s} "
              f"{'objects':>8s} {'moved':>6s} {'a day':>6s} {'asked':>6s} {'movers':>7s} "
              f"{'days':>5s} {'questions':>10s}")
        for r in rows:
            print(f"  {r['name']:14s} {r['residents']:9d} {r['rooms']:6d} {r['places']:7d} "
                  f"{r['objects']:8d} {r['ever_moves']:6d} {r['moves_a_day']:6.0f} "
                  f"{r['asked']:6d} {r['movers']:7d} {r['days']:5d} {r['questions']:10d}")
        for field in ("residents", "rooms", "places", "objects", "ever_moves", "asked", "movers"):
            rng, mean = spread([r[field] for r in rows])
            print(f"    {field:10s} {rng:>12s}   mean {mean}")

    print("\n===== every distinct household in the three sets")
    uniq = {}
    for (banks_name, name), r in every_home.items():
        uniq.setdefault(name, []).append((banks_name, r))
    print(f"  {len(uniq)} household names, {len(every_home)} (bank, household) pairs")
    for name, entries in sorted(uniq.items()):
        shapes = {(r["residents"], r["rooms"], r["places"], r["objects"]) for _, r in entries}
        if len(shapes) > 1:
            print(f"  {name}: DIFFERENT SHAPES per bank: {sorted(shapes)}")
    flat = [r for r in every_home.values()]
    for field in ("residents", "rooms", "places", "objects", "ever_moves", "asked", "movers"):
        rng, mean = spread([r[field] for r in flat])
        print(f"    {field:10s} {rng:>12s}   mean {mean}   over {len(flat)} household runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

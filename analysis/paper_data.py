#!/usr/bin/env python3
"""The row loaders and the object sets every figure and analysis shares.

BOTH-SPELL MOVERS. `search_driven.the_movers` fixes the mover set from the FIRST illness only
(days 14 to 23, at spot level), so on the 50-day bank it includes objects the second illness leaves
where they normally are - 8 of its 22, four of them belonging to a resident who never falls ill.
Every day-14-against-day-32 number therefore needs the objects that move in BOTH spells, which is
what `both_spell_movers` returns: recomputed from the bank rather than hard-coded, and checked
against the list in the brief.

    python3 analysis/paper_data.py     # prints what it found
"""
import collections
import json
import os
import pathlib
import sys

sys.path.insert(0, "src")
from baselines.types import DAY_SECONDS, ON_PERSON, OUT_OF_HOUSE  # noqa: E402
from self_improve import search_driven as sd  # noqa: E402
from self_improve.frozen_household import FrozenHousehold  # noqa: E402

BANKS_50 = pathlib.Path("results/self_improve/varied_homes/all_generated_v2_twice/banks")
BANKS_10 = pathlib.Path("results/self_improve/varied_homes/ten_homes/banks")
# Overridable so the same analysis can be pointed at a rerun without editing anything:
#   WAVE_50=results/self_improve/wave_the_second_illness_ACE_rerun/cells python3 <script>
WAVE_50 = pathlib.Path(os.environ.get(
    "WAVE_50", "results/self_improve/wave_the_second_illness/cells"))
WAVE_10 = pathlib.Path("results/self_improve/overnight_wave/cells")

HOMES_50 = ("hh_s2_t03", "hh_s32_t03", "hh_s48_t03")
HOMES_10 = ("hh_s2_t03", "hh_s19_t03", "hh_s20_t03", "hh_s32_t03", "hh_s48_t03",
            "hh_s63_t03", "hh_s93_t03", "hh_s109_t03", "hh_s123_t03", "hh_s151_t03")

# cell directory -> the name the paper uses
ARM_50 = {"last_seen_no_model": "last-seen",
          "the_log_and_notes_about_the_routine": "log and notes",
          "incremental_edits": "claim store",
          "ACE_as_published": "ACE"}
ARM_10 = {"the_log_and_notes_about_the_routine": "log and notes",
          "last_seen_no_model": "last-seen",
          "incremental_edits": "claim store",
          "claim_store_told_if_it_was_right": "reduced ACE",
          "a_small_working_memory_and_an_archive": "small working memory",
          "prior_only_no_notes": "notes hidden"}

SPELL_1 = range(14, 24)
SPELL_2 = range(32, 42)
SETTLED = range(0, 14)

_homes: dict = {}


def home(name, banks=BANKS_50):
    key = (str(banks), name)
    if key not in _homes:
        _homes[key] = FrozenHousehold(banks / f"{name}.jsonl")
    return _homes[key]


def _commonest_room(h, object_id, days):
    counts: collections.Counter = collections.Counter()
    for day in days:
        for hour in range(8, 23):
            place = h.place_of_object(object_id, day * DAY_SECONDS + hour * 3600)
            if place and place not in (OUT_OF_HOUSE, ON_PERSON):
                counts[h.place_room.get(place)] += 1
    return counts.most_common(1)[0][0] if counts else None


def both_spell_movers(name):
    """Asked-about objects whose commonest daytime ROOM differs from settled in BOTH spells."""
    h = home(name)
    out = set()
    for object_id in sd.the_movers(h):
        base = _commonest_room(h, object_id, SETTLED)
        if (base and _commonest_room(h, object_id, SPELL_1) != base
                and _commonest_room(h, object_id, SPELL_2) != base):
            out.add(object_id)
    return out


def spell_1_movers(name, banks=BANKS_10):
    return sd.the_movers(home(name, banks))


# When WAVE_50 points at a rerun that contains only the arm that was rerun, every other arm falls
# back to the wave it was originally run in. That makes a mixed comparison, and a mixed comparison
# has to be labelled as one wherever it is reported.
WAVE_50_ORIGINAL = pathlib.Path("results/self_improve/wave_the_second_illness/cells")


def rows(wave, arm, name):
    out = []
    if not (wave / arm / name / "searches.jsonl").exists() and \
            (WAVE_50_ORIGINAL / arm / name / "searches.jsonl").exists():
        wave = WAVE_50_ORIGINAL
    for line in (wave / arm / name / "searches.jsonl").open():
        r = json.loads(line)
        if r.get("kind") == "search":
            out.append(r)
    return out


def share(rs, measure):
    if not rs:
        return None
    fn = ((lambda r: r.get("found_at_step") == 1) if measure == "first"
          else (lambda r: bool(r.get("found_it"))))
    return 100 * sum(1 for r in rs if fn(r)) / len(rs)


def main() -> int:
    total = 0
    for name in HOMES_50:
        both = sorted(both_spell_movers(name))
        total += len(both)
        print(f"{name}: {len(both)} both-spell movers of {len(sd.the_movers(home(name)))} "
              f"spell-1 movers\n   " + ", ".join(both))
    print(f"\n{total} both-spell movers over the three homes "
          f"(the brief says 14: {'agrees' if total == 14 else 'DISAGREES'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

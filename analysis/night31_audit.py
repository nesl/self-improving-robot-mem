#!/usr/bin/env python3
"""Night 31: live notes that name an object and one of its illness-time places, by condition.

The matcher is the structural one the rest of the study uses - `where_an_object_is_named` from
write_the_notes, which knows the rewrite arm writes "Nora's book" for book_nora - and the places
come from the BANK, not from text: for each asked object, the set of rooms and spots it occupied
during days 14 to 23. A note counts when it names the object and one of those places.

    PYTHONPATH=src python3 analysis/night31_audit.py
"""
import collections
import os
import json
import pathlib
import re
import sys

sys.path.insert(0, "src")
from baselines.types import DAY_SECONDS, ON_PERSON, OUT_OF_HOUSE          # noqa: E402
from self_improve.frozen_household import FrozenHousehold, plain_place_name  # noqa: E402
from self_improve.write_the_notes import where_an_object_is_named         # noqa: E402

WAVE = pathlib.Path("results/self_improve/overnight_wave/cells")
BANKS = pathlib.Path("results/self_improve/varied_homes/ten_homes/banks")
ARMS = {"incremental_edits": "claim store",
        "the_log_and_notes_about_the_routine": "log and notes",
        "claim_store_told_if_it_was_right": "reduced ACE",
        "a_small_working_memory_and_an_archive": "tight working memory"}
ILLNESS = re.compile(r"\b(unwell|ill|illness|sick|sickness|poorly|not well|under the weather)\b",
                     re.I)
PERSON_HOME = re.compile(r"\b(at home|is home|home all day|stays? home|staying home|in the house|"
                         r"is out|away|not home|goes out)\b", re.I)
A_TIME = re.compile(r"\b(morning|afternoon|evening|night|midday|noon|daytime|"
                    r"\d{1,2}:\d{2}|\d{1,2}\s?(am|pm))\b", re.I)
ALWAYS = re.compile(r"^\s*(always|any time|anytime|all day|current|general|generally)\s*\.?\s*$",
                    re.I)


def illness_places(home):
    """Per object: the places and rooms it occupied during days 14 to 23, in plain words."""
    hh = FrozenHousehold(BANKS / f"{home}.jsonl")
    out = {}
    for obj in hh.asked_objects:
        places, rooms = set(), set()
        for day in range(14, 24):
            for hour in range(8, 23):
                p = hh.place_of_object(obj, day * DAY_SECONDS + hour * 3600)
                if p and p not in (OUT_OF_HOUSE, ON_PERSON):
                    places.add(plain_place_name(p).lower())
                    rooms.add((hh.place_room.get(p) or "").lower().replace("_", " "))
        out[obj] = {w for w in (places | rooms) if w}
    return hh, out


def condition_of(text):
    t = (text or "").strip()
    if not t or ALWAYS.match(t):
        return "always"
    if PERSON_HOME.search(t):
        return "a person home or away"
    if A_TIME.search(t):
        return "a time of day"
    return "other"


def main() -> int:
    totals = collections.Counter()
    by_arm = collections.defaultdict(collections.Counter)
    live_by_arm = collections.Counter()
    mention_illness = collections.Counter()
    examples = []
    for arm_dir, label in ARMS.items():
        for cell in sorted((WAVE / arm_dir).iterdir()):
            if not cell.is_dir():
                continue
            hh, places = illness_places(cell.name)
            notes = json.loads((cell / "notes.json").read_text())
            for c in notes.get("claims", []):
                if c.get("folded_into") is not None or c.get("standing") == "set aside":
                    continue
                live_by_arm[label] += 1
                text = (c.get("statement") or "")
                named = [o for o in places if where_an_object_is_named(text, o) >= 0]
                hit = any(any(w in text.lower() for w in places[o]) for o in named)
                if not hit:
                    continue
                cond = condition_of(c.get("holds_under"))
                by_arm[label][cond] += 1
                totals[label] += 1
                if ILLNESS.search(text) or ILLNESS.search(c.get("holds_under") or ""):
                    mention_illness[label] += 1
                if cond == "a person home or away" and len(examples) < 12:
                    examples.append((label, cell.name, c.get("holds_under"), text[:90]))
    print(f"{'method':22s}{'live notes':>11s}{'describing an illness place':>29s}"
          f"{'always':>8s}{'a time':>8s}{'a person':>10s}{'other':>7s}{'mention illness':>17s}")
    grand = 0
    for label in ARMS.values():
        c = by_arm[label]
        grand += totals[label]
        print(f"{label:22s}{live_by_arm[label]:11d}{totals[label]:29d}{c['always']:8d}"
              f"{c['a time of day']:8d}{c['a person home or away']:10d}{c['other']:7d}"
              f"{mention_illness[label]:17d}")
    print(f"{'TOTAL':22s}{sum(live_by_arm.values()):11d}{grand:29d}"
          f"{sum(by_arm[l]['always'] for l in ARMS.values()):8d}"
          f"{sum(by_arm[l]['a time of day'] for l in ARMS.values()):8d}"
          f"{sum(by_arm[l]['a person home or away'] for l in ARMS.values()):10d}"
          f"{sum(by_arm[l]['other'] for l in ARMS.values()):7d}"
          f"{sum(mention_illness.values()):17d}")
    print("\nevery note whose condition names a person being home or away:")
    for label, home, cond, text in examples:
        print(f"  {label} / {home}\n     condition: {cond!r}\n     {text}")
    json.dump({"by_arm": {k: dict(v) for k, v in by_arm.items()},
               "live": dict(live_by_arm), "illness": dict(mention_illness)},
              open(pathlib.Path(os.environ.get("OUT_DIR", "out")) / "night31_audit.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

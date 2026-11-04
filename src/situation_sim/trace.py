"""Writers: trace.md (human), events.jsonl (machine), hidden_state.json."""
from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional

from situation_sim import timing_constants as tc
from situation_sim.household import Household, ON_PERSON, OUT_OF_HOUSE
from situation_sim.placement import PARAMS
from situation_sim.simulate import RunResult, hhmm
from situation_sim.situation import (DaySituation, STATE_CARRYOVER, STATE_NOISE_SD,
                                     THRESH_LOW_ENERGY, THRESH_HURRIED, THRESH_DISTRACTED)

ROLE_WORDS = {"worker_out": "works outside the home", "worker_home": "works from home",
              "retired": "retired", "student": "a student with classes out", "shift_worker": "works an afternoon-to-night shift"}


def _trait_words(r) -> str:
    tidy = ("very tidy" if r.tidiness > 0.75 else "fairly tidy" if r.tidiness > 0.5
            else "untidy" if r.tidiness > 0.3 else "very untidy")
    punct = ("punctual" if r.jitter_scale < 0.85 else "average timing" if r.jitter_scale < 1.25
             else "erratic timing")
    mood = ("strongly" if r.mood_sensitivity > 0.7 else "moderately" if r.mood_sensitivity > 0.4
            else "barely")
    return (f"{ROLE_WORDS[r.role]}; {tidy} (tidiness {r.tidiness:.2f}); {punct} "
            f"(jitter scale {r.jitter_scale:.2f}); {r.forget_level} forgets pocket items; "
            f"mood {mood} affects behaviour (sensitivity {r.mood_sensitivity:.2f}); "
            f"sleeps in {r.bedroom}, works at the {r.workspace} desk")


def write_trace(path: pathlib.Path, hh: Household, res: RunResult, events: Dict[str, dict]) -> None:
    out: List[str] = []
    days = [sit for sit, _ in res.trace_days]
    out.append(f"# Household {hh.id} — {len(days)} days, {days[0].weekday} to {days[-1].weekday}\n")
    out.append("Seed-generated household. Times are clock times; ids are the receptacle and "
               "object ids used in events.jsonl. A line indented under an activity says where an "
               "object went when the activity ended and why. WHIM marks a placement that landed "
               "somewhere other than where the decision was heading. Text in [brackets] names the "
               "hidden cause the decision is attributed to.\n")
    out.append("## Household\n")
    out.append(f"Type: {hh.household_type}. Rooms: {', '.join(hh.rooms)}.\n")
    out.append("Residents:\n")
    for r in sorted(hh.residents.values(), key=lambda r: r.id):
        out.append(f"- **{r.name.capitalize()}** ({r.id}): {_trait_words(r)}. Hobbies: "
                   f"{', '.join(r.hobbies) or 'none'}. Chores they take on: {', '.join(r.chores) or 'none'}.")
    if hh.pet:
        out.append(f"\nPet: a {hh.pet}, mainly looked after by {hh.residents[hh.carer].name.capitalize()}.")
    n_static = sum(1 for o in hh.objects.values() if o.static)
    out.append(f"\nObjects ({len(hh.objects)}, of which {n_static} are fixtures that never move) and their usual place, primary slot first:\n")
    for r_id in sorted(hh.residents) + [None]:
        objs = [o for o in hh.objects_of(r_id) if not o.static]
        owner = hh.residents[r_id].name.capitalize() + "'s" if r_id else "Shared"
        out.append(f"- {owner}: " + "; ".join(f"{o.id} → {' / '.join(o.home)}" for o in objs))
    statics = [o for o in hh.objects.values() if o.static]
    if statics:
        out.append("- Fixtures: " + "; ".join(f"{o.id} at {o.home[0]}" for o in sorted(statics, key=lambda o: o.id)))
    if hh.groups:
        out.append("\nObject groups (things that travel together):\n")
        for g in sorted(hh.groups.values(), key=lambda g: g.name):
            inside = ", ".join(g.members) if g.members else "nothing in particular"
            trip = "trips to work and errands" if g.kind == "bag" else "gym trips"
            out.append(f"- {g.leader} carries {inside} on {trip}")
    out.append("")
    for sit, lines in res.trace_days:
        out.append(f"\n## Day {sit.day_index} — {sit.weekday}\n")
        out.append("**Active causes**\n")
        for c in sorted(sit.causes, key=lambda c: (c.kind, c.id)):
            if c.kind == "event":
                scope = "household" if c.resident is None else hh.residents[c.resident].name.capitalize()
                out.append(f"- `{c.id}` ({scope}): {events[c.event]['description']} "
                           f"{events[c.event]['trace_words']}.")
            else:
                out.append(f"- `{c.id}`: {c.words}.")
        st = "; ".join(f"{hh.residents[r].name.capitalize()} energy {v['energy']:.2f}, "
                       f"hurriedness {v['hurriedness']:.2f}, distraction {v['distraction']:.2f}"
                       for r, v in sorted(sit.states.items()))
        out.append(f"\nInternal states (0–1): {st}\n")
        out.append("**Timeline**\n")
        out.append("```")
        for l in lines:
            if l.indent:
                out.append(f"          {l.text}")
            else:
                out.append(f"{hhmm(l.minute)}  {l.text}")
        out.append("```")
    path.write_text("\n".join(out) + "\n")


def write_events(path: pathlib.Path, hh: Household, res: RunResult, n_days: int,
                 episode_id: str) -> None:
    header = {
        "kind": "episode_header", "episode_id": episode_id, "household_id": hh.id,
        "receptacle_ids": sorted(hh.receptacles) + [ON_PERSON, OUT_OF_HOUSE],
        "object_classes": {o.id: o.cls for o in sorted(hh.objects.values(), key=lambda o: o.id)},
        "budget_per_day": 0, "n_days": n_days,
        "household_type": hh.household_type,
        "unsensable_receptacles": [OUT_OF_HOUSE],
        "receptacle_rooms": {**{r.id: r.room for r in sorted(hh.receptacles.values(), key=lambda r: r.id)},
                             ON_PERSON: "person_check"},
        "home_base_room": _home_base(hh),
        "person_sensing": True,
        "resident_ids": sorted(hh.residents),
        "generator": "situation_sim",
    }
    rows = [header] + sorted(res.truth_rows, key=lambda r: (r["t"], r["object_id"])) \
        + sorted(res.resident_rows, key=lambda r: (r["t"], r["resident_id"]))
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=False) + "\n")


def _home_base(hh: Household) -> str:
    counts: Dict[str, int] = {}
    for r in hh.receptacles.values():
        counts[r.room] = counts.get(r.room, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def write_hidden_state(path: pathlib.Path, hh: Household, sits: List[DaySituation],
                       res: RunResult, events: Dict[str, dict], seed: int,
                       episodes: Optional[Dict[str, dict]] = None) -> None:
    state = {
        "seed": seed,
        "household": hh.to_json(),
        "days": [s.to_json() for s in sits],
        "placement_params": PARAMS,
        "internal_state_params": {
            "carryover": STATE_CARRYOVER, "noise_sd": STATE_NOISE_SD,
            "thresh_low_energy": THRESH_LOW_ENERGY, "thresh_hurried": THRESH_HURRIED,
            "thresh_distracted": THRESH_DISTRACTED},
        "timing_constants": {
            "jitter_classes_min": tc.JITTER_CLASSES_MIN,
            "jitter_scale_bounds": [tc.JITTER_SCALE_MIN, tc.JITTER_SCALE_MAX],
            "keep_block_share": tc.KEEP_BLOCK_SHARE,
            "mean_bouts_bounds": [tc.MIN_MEAN_BOUTS, tc.MAX_MEAN_BOUTS],
            "min_bout_minutes": tc.MIN_BOUT_MINUTES, "max_skip_p": tc.MAX_SKIP_P,
            "carry_p": tc.CARRY_P, "forget_levels": tc.FORGET_LEVELS},
        "event_library": events,
        "episode_library": episodes or {},
        "visitors": res.visitors,
        "stats": res.stats,
    }
    path.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Structured trace for the inspection page
# ---------------------------------------------------------------------------
import re as _re

_TAG_RE = _re.compile(r" \[(?:because of |shifted by )?([^\]]+)\]")


def _line_kind(text: str, indent: bool) -> str:
    if indent:
        if "WHIM" in text:
            return "whim"
        if text.startswith("keeps "):
            return "keep"
        if text.startswith("forgets "):
            return "forget"
        if "→" in text:
            return "move"
        return "note"
    if " leaves for " in text:
        return "trip"
    if " is back from " in text:
        return "back"
    if " finishes " in text:
        return "end"
    return "start"


def write_trace_json(path: pathlib.Path, hh: Household, res: RunResult) -> None:
    """Same content as trace.md, but one record per line with the cause tags
    split out (so a viewer can hide them) and the object ids mentioned."""
    obj_ids = sorted(hh.objects, key=len, reverse=True)
    obj_re = _re.compile(r"\b(" + "|".join(_re.escape(o) for o in obj_ids) + r")\b")
    rec_re = _re.compile(r"\b(" + "|".join(_re.escape(r) for r in sorted(hh.receptacles, key=len, reverse=True)) + r")\b")
    days = []
    for sit, lines in res.trace_days:
        recs = []
        for l in lines:
            tags: List[str] = []
            for m in _TAG_RE.finditer(l.text):
                tags += [t.strip() for t in m.group(1).split(",")]
            text = _TAG_RE.sub("", l.text)
            recs.append({
                "minute": l.minute, "indent": l.indent, "text": text,
                "kind": _line_kind(text, l.indent), "tags": tags,
                "objects": sorted(set(obj_re.findall(text))),
                "receptacles": sorted(set(rec_re.findall(text))),
                "resident": next((r.id for r in sorted(hh.residents.values(), key=lambda r: r.id)
                                  if not l.indent and text.startswith(r.name.capitalize() + " ")), None),
            })
        days.append({"day_index": sit.day_index, "weekday": sit.weekday, "lines": recs})
    path.write_text(json.dumps({"days": days}, indent=None, sort_keys=True) + "\n")

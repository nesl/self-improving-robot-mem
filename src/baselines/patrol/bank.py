"""Build a patrol bank from a situation_sim run.

    python3 -m baselines.patrol.bank --run ../data/situation_sim/week8/hh_s0 \
        --patrol-hours 4 --out ../results/overnight/banks/hh_s0_p4.jsonl

What goes into the bank (all rows the loader in baselines.bank understands):

* the sim's header, plus a ``protocol`` block every agent may read (the
  resident intro cards, patrol density, walkthrough time, day 0's weekday)
  and harness-only fields (``day_causes``, ``shift_days``,
  ``hint_messages``) that never reach an agent's context;
* the sim's truth and resident rows (ground truth, harness-only);
* the walkthrough: one ``initial_tour`` observation per object found in
  the home on Tuesday at 18:00, plus one ``room_visit`` per room listing
  the EMPTY spots (so the walkthrough's exclusions reach the beliefs);
* the patrol: every ``patrol_hours`` hours after the walkthrough, one
  ``room_visit`` per room with the full contents of every spot in it
  (residents present are stamped on by the loader from the resident rows);
* 32 questions per scored day (Wednesday .. next Tuesday), uniform over
  movable objects, at times uniform over 07:00-23:00.

Same seed and patrol density -> byte-identical file.
"""
from __future__ import annotations

import argparse
import bisect
import json
import pathlib
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from baselines.types import DAY_SECONDS, ON_PERSON, OUT_OF_HOUSE

WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKEND = {"Saturday", "Sunday"}
MAJOR_EVENTS = ("guest_visit", "sick_day")
"""Generator events that mark a day as a shift (besides the weekend)."""
QUESTIONS_PER_DAY = 32
OVERSAMPLE = 60
"""Candidate questions drawn per day in ``sensable`` mode before the
OUT_OF_HOUSE / ON_PERSON truths are dropped and 32 are kept."""
WAKING_MINUTES = (7 * 60, 23 * 60)
TOUR_MINUTE = 18 * 60

ROLE_WORDS = {
    "worker_out": "works in an office in town",
    "worker_home": "works from home",
    "student": "is a student",
    "shift_worker": "works afternoon-to-night shifts",
    "retired": "is retired",
}
ROLE_WEEKDAY = {
    "worker_out": "out at work from about 8 in the morning to 5:30, then home for the evening",
    "worker_home": "at the desk at home from 9 to about 5:30, sometimes out for lunch",
    "student": "at classes from about 8:40 to 2:30, then studying at home",
    "shift_worker": "home in the morning, out at work from about 1:40 in the afternoon until 11 at night",
    "retired": "home most of the day, with a morning walk and afternoon errands",
}
ROLE_WEEKEND = {
    "worker_out": "sleeps in, errands around midday, home the rest of the day",
    "worker_home": "sleeps in, errands around midday, home the rest of the day",
    "student": "sleeps in, studies in the afternoon, home in the evening",
    "shift_worker": "off work; errands around midday, home the rest of the day",
    "retired": "home most of the day, with a late-morning walk",
}
ROLE_AGE_BANDS = {
    "worker_out": ["late twenties", "thirties", "forties", "early fifties"],
    "worker_home": ["late twenties", "thirties", "forties", "early fifties"],
    "student": ["early twenties"],
    "shift_worker": ["late twenties", "thirties", "forties"],
    "retired": ["sixties", "seventies"],
}
HOBBY_WORDS = {"reading_living": "reading", "family_calls": "calls with family",
               "friends": "seeing friends", "plants": "houseplants", "music": "music",
               "movies": "movies", "board_games": "board games", "puzzles": "jigsaw puzzles"}


def _rows(path: pathlib.Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class _Truth:
    """Piecewise-constant object locations from the truth rows."""

    def __init__(self, truth_rows: List[dict]) -> None:
        self._t: Dict[str, List[int]] = defaultdict(list)
        self._r: Dict[str, List[str]] = defaultdict(list)
        for r in sorted(truth_rows, key=lambda r: (r["object_id"], r["t"])):
            self._t[r["object_id"]].append(r["t"])
            self._r[r["object_id"]].append(r["receptacle_id"])

    def at(self, obj: str, t: int) -> Optional[str]:
        i = bisect.bisect_right(self._t[obj], t) - 1
        return self._r[obj][i] if i >= 0 else None


def resident_cards(state: dict) -> List[dict]:
    """The intro card of every resident: name, age band, occupation, one
    sentence each for the weekday and the weekend, hobbies, pet."""
    hh = state["household"]
    seed = int(state["seed"])
    cards = []
    for rid, r in sorted(hh["residents"].items()):
        rng = random.Random(f"intro_card:{seed}:{rid}")
        name = r["name"].capitalize()
        hobbies = [HOBBY_WORDS.get(h, h.replace("_", " ")) for h in sorted(r["hobbies"])]
        card = {
            "resident_id": rid, "name": name,
            "age_band": rng.choice(ROLE_AGE_BANDS[r["role"]]),
            "occupation": ROLE_WORDS[r["role"]],
            "weekday": f"Weekdays: {ROLE_WEEKDAY[r['role']]}.",
            "weekend": f"Weekends: {ROLE_WEEKEND[r['role']]}.",
            "hobbies": hobbies,
            "bedroom": r["bedroom"],
        }
        if hh.get("pet") and hh.get("carer") == rid:
            card["pet"] = f"looks after the household's {hh['pet']}"
        cards.append(card)
    return cards


def card_sentences(card: dict) -> str:
    s = (f"{card['name']} (in their {card['age_band']}) {card['occupation']}. "
         f"{card['weekday']} {card['weekend']}")
    if card["hobbies"]:
        s += f" Hobbies: {', '.join(card['hobbies'])}."
    if card.get("pet"):
        s += f" {card['name']} {card['pet']}."
    return s


def shift_info(state: dict, scored_days: List[int]) -> Tuple[List[int], Dict[int, List[str]], List[dict]]:
    """(shift_days, day -> active cause ids, hint messages).

    A shift day is a weekend day or a scored day with a major event. The
    hint is one dated sentence a resident would say, written here from the
    cause type, never from the internal cause name."""
    names = {rid: r["name"].capitalize() for rid, r in state["household"]["residents"].items()}
    shift_days: List[int] = []
    day_causes: Dict[int, List[str]] = {}
    hints: List[dict] = []
    for d in state["days"]:
        di = int(d["day_index"])
        day_causes[di] = sorted(c["id"] for c in d["causes"])
        if di not in scored_days:
            continue
        weekday = d["weekday"]
        sentences: List[str] = []
        if d.get("is_weekend", weekday in WEEKEND):   # a calendar may put a weekday on the weekend schedule
            sentences.append("It's the weekend, so we're off our usual routine and around the house more.")
        for c in sorted(d["causes"], key=lambda c: c["id"]):
            if c["kind"] != "event" or c["event"] not in MAJOR_EVENTS:
                continue
            if c["event"] == "guest_visit":
                sentences.append("We have friends coming over this evening.")
            elif c["event"] == "sick_day":
                sentences.append(f"{names[c['resident']]} is home sick today.")
        if sentences:
            shift_days.append(di)
            hints.append({"day_index": di, "weekday": weekday,
                          "text": f"{weekday}: " + " ".join(sentences)})
    return shift_days, day_causes, hints


def question_rows(seed: int, objects: dict, scored_days: List[int], eid: str) -> List[dict]:
    """The overnight bank's questions: 32 per day, object uniform over movable
    objects, any truth (OUT_OF_HOUSE and ON_PERSON included)."""
    movable = sorted(o for o, spec in objects.items() if not spec.get("static"))
    rng = random.Random(f"patrol_questions:{seed}")
    rows = []
    for d in scored_days:
        minutes = sorted(rng.randrange(WAKING_MINUTES[0], WAKING_MINUTES[1]) for _ in range(QUESTIONS_PER_DAY))
        for k, minute in enumerate(minutes):
            obj = rng.choice(movable)
            rows.append({"kind": "question", "episode_id": eid, "question_id": f"d{d}q{k + 1:02d}",
                         "object_id": obj, "t_query": d * DAY_SECONDS + minute * 60, "day_index": d,
                         "object_class": objects[obj]["cls"]})
    return rows


def sensable_question_rows(seed: int, objects: dict, scored_days: List[int], eid: str, truth: "_Truth",
                           classes: Optional[List[str]] = None, per_day: int = QUESTIONS_PER_DAY,
                           oversample: int = OVERSAMPLE) -> Tuple[List[dict], dict]:
    """Questions whose truth is an in-house spot: ``oversample`` candidates
    per day (object uniform over the eligible classes, minute uniform over
    waking hours), the OUT_OF_HOUSE / ON_PERSON truths dropped, then
    ``per_day`` kept by a seeded draw and ordered by time. Returns the rows
    and a small count summary (per day: drawn, kept after the filter)."""
    movable = sorted(o for o, spec in objects.items()
                     if not spec.get("static") and (classes is None or spec["cls"] in classes))
    if not movable:
        raise ValueError("no eligible objects")
    rng = random.Random(f"patrol_questions_sensable:{seed}")
    rows: List[dict] = []
    counts = {}
    for d in scored_days:
        cands = []
        for _ in range(oversample):
            minute = rng.randrange(WAKING_MINUTES[0], WAKING_MINUTES[1])
            obj = rng.choice(movable)
            cands.append((minute, obj))
        ok = [(m, o) for m, o in cands if truth.at(o, d * DAY_SECONDS + m * 60) not in (ON_PERSON, OUT_OF_HOUSE, None)]
        counts[d] = {"drawn": len(cands), "sensable": len(ok)}
        if len(ok) < per_day:
            raise ValueError(f"day {d}: only {len(ok)} sensable candidates of {oversample}; raise oversample")
        keep = sorted(rng.sample(ok, per_day))
        for k, (minute, obj) in enumerate(keep):
            rows.append({"kind": "question", "episode_id": eid, "question_id": f"d{d}q{k + 1:02d}",
                         "object_id": obj, "t_query": d * DAY_SECONDS + minute * 60, "day_index": d,
                         "object_class": objects[obj]["cls"]})
    return rows, counts


TOKEN_CLASSES = {"pocket", "outdoor", "bag", "gym", "bag_leader"}
CHORE_ACTIVITIES = ("vacuum", "dust", "laundry", "iron", "water_plants", "feed_dog", "fix_something", "wash_dishes", "tidy")
CHORE_SHARE = 0.15
"""Share of question moments that are chores the robot could do while a
resident is out (drawn after a trip line), the rest are activity starts."""
START_WINDOW = (-5, 15)
"""Minutes around an activity start at which the question lands."""
QUESTION_MOMENT = os.environ.get("PATROL_QUESTION_MOMENT", "start")
"""Regime search (2026-09-21): where inside an activity the question lands. ``start`` = START_WINDOW
around the start line (the object is often mid-move); ``during`` = uniformly inside the activity
(between its start and the resident's next end line, 2 min margins: the object is in use at the
activity surface); ``after`` = 5-30 min after the end (put back, or left where it was used)."""
CHORE_WINDOW = (10, 90)


def load_activities() -> Dict[str, dict]:
    import yaml
    path = pathlib.Path(__file__).resolve().parents[2] / "situation_sim" / "activities.yaml"
    return yaml.safe_load(path.read_text())["activities"]


def activity_of(text: str, by_words: List[Tuple[str, str]]) -> Optional[str]:
    """The activity named in a trace start line ('Name — words in the ...')."""
    if " — " not in text:
        return None
    seg = text.split(" — ", 1)[1]
    for words, name in by_words:
        if seg.startswith(words) and (len(seg) == len(words) or not seg[len(words)].isalpha()):
            return name
    return None


def uses_classes(spec: dict) -> List[str]:
    out: List[str] = []
    for u in spec.get("uses", []):
        for alt in u.split("|"):
            if alt not in TOKEN_CLASSES and alt not in out:
                out.append(alt)
    return out


def activity_question_rows(seed: int, objects: dict, scored_days: List[int], eid: str, truth: "_Truth",
                           trace: dict, classes: Optional[List[str]] = None, per_day: int = QUESTIONS_PER_DAY,
                           oversample: int = OVERSAMPLE, shift_focus: float = 0.0,
                           shift_days: Optional[List[int]] = None) -> Tuple[List[dict], dict]:
    """Questions tied to what the residents are doing: a resident starting an
    activity asks for one of the objects that activity uses (question a few
    minutes around the start); after someone leaves the house the robot is
    asked about a chore object. Candidates whose truth is OUT_OF_HOUSE /
    ON_PERSON are dropped, ``per_day`` kept by a seeded draw, ordered by
    time. Returns the rows and per-day counts."""
    acts = load_activities()
    by_words = sorted(((spec["words"], name) for name, spec in acts.items()), key=lambda w: -len(w[0]))
    owners = [x for x in os.environ.get("PATROL_QUESTION_OWNERS", "").split(",") if x]   # regime search: e.g. resident_1
    eligible = lambda o: (not objects[o].get("static")) and (classes is None or objects[o]["cls"] in classes) \
        and (not owners or objects[o].get("owner") in owners)
    by_class: Dict[str, List[str]] = defaultdict(list)
    for o in sorted(objects):
        if eligible(o):
            by_class[objects[o]["cls"]].append(o)
    chore_classes = sorted({c for a in CHORE_ACTIVITIES for c in uses_classes(acts[a])})
    # regime search: PATROL_QUESTION_HOURS="17-23" keeps only questions whose local hour is in [lo, hi]; unset (the
    # default) leaves every candidate, so a bank built without it is byte-identical to before this knob existed.
    hours_env = os.environ.get("PATROL_QUESTION_HOURS", "").strip()
    hour_lo = hour_hi = None
    if hours_env:
        lo_s, _, hi_s = hours_env.partition("-")
        hour_lo, hour_hi = int(lo_s), int(hi_s or lo_s)
        if not (0 <= hour_lo <= hour_hi <= 23):
            raise ValueError(f"PATROL_QUESTION_HOURS={hours_env!r}: want 'LO-HI' with 0 <= LO <= HI <= 23")
    rng = random.Random(f"patrol_questions_activity:{seed}" + (f":focus{shift_focus:g}" if shift_focus else "")
                        + (f":owners{','.join(owners)}" if owners else "")
                        + (f":hours{hours_env}" if hours_env else ""))
    days = {int(d["day_index"]): d for d in trace["days"]}
    rows: List[dict] = []
    counts = {}
    shift_days = set(shift_days or [])
    # activities each resident does on Wed-Fri (the weekday baseline): a weekend start of an
    # activity outside this set is a weekend-only activity for that resident
    weekday_acts: Dict[Optional[str], set] = defaultdict(set)
    for d in scored_days[:3]:
        for l in days[d]["lines"]:
            if l["kind"] == "start":
                a = activity_of(l["text"], by_words)
                if a:
                    weekday_acts[l.get("resident")].add(a)
    for d in scored_days:
        lines = days[d]["lines"]
        starts = []   # (minute, activity, resident)
        shift_starts = []
        trips = []    # minute
        # end minute of each start: the resident's next end line (the activity's own end)
        ends: Dict[Tuple[int, Optional[str]], int] = {}
        pending: Dict[Optional[str], int] = {}
        for l in lines:
            if l["kind"] == "start":
                pending[l.get("resident")] = int(l["minute"])
            elif l["kind"] == "end" and l.get("resident") in pending:
                ends[(pending.pop(l.get("resident")), l.get("resident"))] = int(l["minute"])
        for l in lines:
            if l["kind"] == "start":
                a = activity_of(l["text"], by_words)
                if a and acts[a].get("room") != "ELSEWHERE":
                    starts.append((int(l["minute"]), a, l.get("resident")))
                    tagged = any(t.split(":")[0] in MAJOR_EVENTS for t in l.get("tags", []))
                    weekend_only = d in shift_days and a not in weekday_acts.get(l.get("resident"), set())
                    if tagged or weekend_only:
                        shift_starts.append((int(l["minute"]), a, l.get("resident")))
            elif l["kind"] == "trip":
                trips.append(int(l["minute"]))
        cands = []
        n_start = n_chore = 0
        for _ in range(oversample):
            if trips and rng.random() < CHORE_SHARE:
                m0 = rng.choice(trips)
                pool = [o for c in chore_classes for o in by_class.get(c, [])]
                minute = m0 + rng.randint(*CHORE_WINDOW)
                kind = "chore"
            else:
                if not starts:
                    break
                pool_starts = shift_starts if (d in shift_days and shift_starts and rng.random() < shift_focus) else starts
                m0, a, res = rng.choice(pool_starts)
                pool = [o for c in uses_classes(acts[a]) for o in by_class.get(c, [])
                        if objects[o].get("owner") in (None, res)]
                end = ends.get((m0, res))
                moment = os.environ.get("PATROL_QUESTION_MOMENT", QUESTION_MOMENT)
                if moment == "during" and end is not None and end - m0 >= 6:
                    minute = rng.randint(m0 + 2, end - 2)
                elif moment == "after" and end is not None:
                    minute = end + rng.randint(5, 30)
                else:
                    minute = m0 + rng.randint(*START_WINDOW)
                kind = a
            if not pool:
                continue
            minute = min(max(minute, WAKING_MINUTES[0]), WAKING_MINUTES[1] - 1)
            if hour_lo is not None and not (hour_lo <= minute // 60 <= hour_hi):
                continue   # regime search: keep only questions asked in the hours the studied event actually touches
            obj = rng.choice(sorted(pool))
            t = d * DAY_SECONDS + minute * 60
            if truth.at(obj, t) in (ON_PERSON, OUT_OF_HOUSE, None):
                continue
            cands.append((minute, obj, kind))
            n_start += kind != "chore"; n_chore += kind == "chore"
        counts[d] = {"drawn": oversample, "sensable": len(cands), "start": n_start, "chore": n_chore,
                     "shift_starts": len(shift_starts)}
        if len(cands) < per_day:
            raise ValueError(f"day {d}: only {len(cands)} sensable candidates of {oversample}; raise oversample")
        # regime search: no two questions about the same object within MIN_GAP minutes (a repeat inside one
        # activity block is the same question twice, and the found-it feedback of the first answers the second)
        min_gap = int(os.environ.get("PATROL_QUESTION_MIN_GAP_MIN", "0"))
        if min_gap > 0:
            # greedy in time order over a seeded shuffle of ties: keeps the most questions per day
            order = list(range(len(cands))); rng.shuffle(order); order.sort(key=lambda i: cands[i][0])
            pool_taken: List[Tuple[int, str, str]] = []
            last_t: Dict[str, int] = {}
            for i in order:
                minute, obj, kind = cands[i]
                if obj not in last_t or minute - last_t[obj] >= min_gap:
                    pool_taken.append(cands[i]); last_t[obj] = minute
            # with a gap the day's natural number of distinct questions may be below per_day: take what there is
            taken = rng.sample(pool_taken, min(per_day, len(pool_taken)))
            counts[d]["distinct"] = len(pool_taken)
            keep = sorted(taken)
        else:
            keep = sorted(rng.sample(cands, per_day))
        for k, (minute, obj, kind) in enumerate(keep):
            rows.append({"kind": "question", "episode_id": eid, "question_id": f"d{d}q{k + 1:02d}",
                         "object_id": obj, "t_query": d * DAY_SECONDS + minute * 60, "day_index": d,
                         "object_class": objects[obj]["cls"], "moment": kind})
    return rows, counts


def parse_times(spec: str) -> List[int]:
    """'03:00,11:00,19:00' -> minutes of day, sorted."""
    out = []
    for part in spec.split(","):
        hh, mm = part.strip().split(":")
        out.append(int(hh) * 60 + int(mm))
    return sorted(set(out))


def patrol_label(patrol_hours: int, patrol_times: Optional[List[int]]) -> str:
    if patrol_times:
        return "t" + "-".join(f"{m // 60:02d}{m % 60:02d}" if m % 60 else f"{m // 60:02d}" for m in patrol_times)
    return f"p{patrol_hours}"


def build_bank(run_dir: pathlib.Path, out: pathlib.Path, patrol_hours: int,
               patrol_times: Optional[List[int]] = None, questions: str = "legacy",
               classes: Optional[List[str]] = None, per_day: int = QUESTIONS_PER_DAY,
               oversample: int = OVERSAMPLE, shift_focus: float = 0.0,
               feedback_delay_min: Optional[int] = None) -> pathlib.Path:
    """``shift_focus``: on a shift day, the probability that a question moment is drawn
    from the day's shift activities (starts tagged with a major event, or activities the
    resident does not do on Wed-Fri) instead of all activities.
    ``feedback_delay_min``: found-it feedback. ``delay`` minutes after each question the
    robot learns where the object turned out to be (its true spot at the question instant),
    as an ordinary sighting every agent receives. None = no feedback; -1 = at the next
    nightly review (one minute before the 03:00 round)."""
    """``patrol_times`` (minutes of day) replaces the every-``patrol_hours``
    patrol with one pass at each listed clock time every day. ``questions``
    is ``legacy`` (the overnight bank) or ``sensable`` (in-house truths
    only, see :func:`sensable_question_rows`)."""
    rows = _rows(run_dir / "events.jsonl")
    header = dict(rows[0])
    assert header["kind"] == "episode_header"
    state = json.loads((run_dir / "hidden_state.json").read_text())
    seed = int(state["seed"])
    n_days = int(header["n_days"])
    objects = state["household"]["objects"]
    truth = _Truth([r for r in rows if r["kind"] == "truth"])
    eid = header["episode_id"]
    day0 = state["days"][0]["weekday"]
    scored_days = list(range(1, n_days))
    tour_t = TOUR_MINUTE * 60

    rooms: Dict[str, List[str]] = defaultdict(list)
    for rec, room in sorted(header["receptacle_rooms"].items()):
        if rec != ON_PERSON:
            rooms[room].append(rec)
    room_list = sorted(rooms)

    def visit(room: str, t: int, only_empty: bool) -> Optional[dict]:
        contents = {rec: sorted(o for o in sorted(objects) if truth.at(o, t) == rec) for rec in sorted(rooms[room])}
        if only_empty:
            contents = {rec: objs for rec, objs in contents.items() if not objs}
        if not contents:
            return None
        return {"kind": "room_visit", "episode_id": eid, "t": t, "room": room, "contents": contents}

    # walkthrough
    tour_obs = [{"kind": "observation", "episode_id": eid, "object_id": o, "receptacle_id": truth.at(o, tour_t),
                 "t": tour_t, "source": "initial_tour"}
                for o in sorted(objects) if truth.at(o, tour_t) not in (ON_PERSON, OUT_OF_HOUSE, None)]
    tour_visits = [v for room in room_list for v in [visit(room, tour_t, only_empty=True)] if v]
    # patrol instants: every patrol_hours hours, or the listed clock times every day
    if patrol_times:
        instants = [d * DAY_SECONDS + m * 60 for d in range(n_days) for m in patrol_times]
    else:
        step = patrol_hours * 3600
        instants = list(range(step, n_days * DAY_SECONDS, step))
    patrol_rows = []
    for t in instants:
        if t > tour_t:
            for room in room_list:
                v = visit(room, t, only_empty=False)
                if v:
                    patrol_rows.append(v)
    shift_days, day_causes, hints = shift_info(state, scored_days)
    feedback_rows: List[dict] = []
    if questions == "sensable":
        question_list, qcounts = sensable_question_rows(seed, objects, scored_days, eid, truth, classes, per_day, oversample)
    elif questions == "activity":
        trace = json.loads((run_dir / "trace.json").read_text())
        question_list, qcounts = activity_question_rows(seed, objects, scored_days, eid, truth, trace, classes, per_day, oversample,
                                                        shift_focus, shift_days)
    else:
        question_list, qcounts = question_rows(seed, objects, scored_days, eid), {}
    if feedback_delay_min is not None:
        seen_at: Dict[Tuple[str, int], bool] = {}
        for q in question_list:
            if feedback_delay_min < 0:
                # feedback at the next nightly review: the robot learns the day's outcomes at its 03:00 round
                night = 3 * 3600
                t_fb = (q["t_query"] // DAY_SECONDS + 1) * DAY_SECONDS + night - 60
            else:
                t_fb = q["t_query"] + feedback_delay_min * 60
            key = (q["object_id"], t_fb)
            if key in seen_at:      # two questions at the same moment about the same object: one feedback
                continue
            seen_at[key] = True
            feedback_rows.append({"kind": "observation", "episode_id": eid, "object_id": q["object_id"],
                                  "receptacle_id": truth.at(q["object_id"], q["t_query"]), "t": t_fb, "source": "scripted"})
    # regime search: the calendar stage of each day (None outside any stage) as a per-question tag
    stage_of_day = {int(d["day_index"]): d.get("stage") for d in state["days"]}
    if any(v is not None for v in stage_of_day.values()):
        for q in question_list:
            q["stage"] = stage_of_day.get(int(q["day_index"]))
    cards = resident_cards(state)
    label = patrol_label(patrol_hours, patrol_times)
    times_text = [f"{m // 60:02d}:{m % 60:02d}" for m in (patrol_times or [])]

    header["budget_per_day"] = 1
    header["tour_t"] = tour_t
    header["patrol_hours"] = 0 if patrol_times else patrol_hours
    header["patrol_times"] = times_text
    header["patrol_label"] = label
    header["day0_weekday"] = day0
    header["day_names"] = {str(d["day_index"]): d["weekday"] for d in state["days"]}
    header["scored_days"] = scored_days
    header["shift_days"] = shift_days
    if any(v is not None for v in stage_of_day.values()):
        header["stages"] = {str(k): v for k, v in sorted(stage_of_day.items())}       # calendar stage per day
        header["day_kinds"] = {str(d["day_index"]): ("weekend" if d.get("is_weekend") else "weekday") for d in state["days"]}
    header["day_causes"] = {str(k): v for k, v in sorted(day_causes.items())}   # harness-only
    header["hint_messages"] = hints                                              # LLM told arm only
    header["question_mode"] = questions
    header["question_counts"] = {str(k): v for k, v in sorted(qcounts.items())}
    header["question_classes"] = sorted(classes) if classes else None
    header["shift_focus"] = shift_focus
    header["protocol"] = {
        "walkthrough_t": tour_t, "questions_per_day": per_day, "first_question_day": 1,
        "patrol_hours": header["patrol_hours"], "patrol_times": times_text,
        "free_look": questions == "legacy", "room_level_looks": questions == "legacy",
        "patrol": questions != "legacy",     # the passive patrol stream, no looks (read by llm_hypotheses.protocol_text)
        "question_moments": questions,       # 'activity': questions arise around what residents do (told to the LLM agents)
        "feedback_delay_min": feedback_delay_min,   # found-it feedback after each question (told to the LLM agents)
        "pockets_visible": False, "day0_weekday": day0,
        "residents": cards,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(json.dumps(header, sort_keys=True) + "\n")
        for r in rows[1:]:
            f.write(json.dumps(r, sort_keys=True) + "\n")
        for r in tour_obs + tour_visits + patrol_rows + feedback_rows + question_list:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--patrol-hours", type=int, default=4)
    ap.add_argument("--patrol-times", default=None, help="clock times, e.g. 03:00,11:00,19:00 (replaces --patrol-hours)")
    ap.add_argument("--questions", default="legacy", choices=("legacy", "sensable", "activity"))
    ap.add_argument("--classes", nargs="*", default=None, help="eligible object classes (sensable mode)")
    ap.add_argument("--per-day", type=int, default=QUESTIONS_PER_DAY)
    ap.add_argument("--oversample", type=int, default=OVERSAMPLE)
    a = ap.parse_args(argv)
    p = build_bank(a.run, a.out, a.patrol_hours, parse_times(a.patrol_times) if a.patrol_times else None,
                   a.questions, a.classes, a.per_day, a.oversample)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

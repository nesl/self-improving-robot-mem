"""Rare movers with a visible cause: a post-pass over a finished simulation.

The base simulator moves an object only when an activity uses it, and the activities that use
the asked-about classes run every day, so a home has fixtures and daily movers and nothing in
between. This module adds, AFTER the base run has finished, two objects that move a few times
a month and whose every move is caused by something the robot can see when it opens a room:
the resident is recorded in that room by the base run's own `resident` rows.

  * ``pill_box_<ill resident>`` (class ``pill_box``). At its usual place (a kitchen counter or
    the bathroom cabinet, one seeded draw per home). On the first day an ``unwell_spell``
    cause is active it moves to the ill resident's nightstand, at the start of that
    resident's first bedroom bout of the day (the morning dose). On the first day after the
    spell it goes back, at the start of that resident's first bout outside the bedroom (they
    are up and about again).
  * ``car_keys_shared`` (class ``car_keys``). On the entry table at the start of the month.
    One to three times a week (a seeded draw per week) one resident's trip out is a car trip:
    the keys leave the house with them and, when they get back, land where that person first
    sits: the living-room coffee table if they settle there before their next activity, the
    surface of the next activity otherwise (kitchen table, desk, dining table...). They stay
    there until the next car trip.
  * reading glasses are NOT built here: with every calendar day a weekday the only
    non-working days are the sick days, which would make them either a daily mover or a copy
    of the illness movers. See the study folder's FINDINGS_generator.md.

Nothing in the base run changes. The pass runs after ``Simulator.run`` has returned, draws
from its own ``random.Random``, and only appends: objects to ``hh.objects``, truth rows to
``res.truth_rows``, indented lines to the trace. Every existing object's trajectory, every
resident row and every base question are therefore byte-identical to a run without it.
Checked when the ten-home rare-mover banks were built (2026-10-02).

Questions: ``rare_mover_questions`` draws questions about these objects concentrated after each
place change: five stratified over the first hour (so the robot is asked before it has had much
chance to see the thing by accident) and two over the rest of the 24 hours, plus a thin
background outside the windows, and stamps every row with ``within_24h_of_rare_move`` so
scoring needs no re-derivation. Whether the robot had already sighted the object at its new
place when asked depends on its own looks, so that flag is added at scoring time
(``times_seen_there_before == 0`` on the search row), not here.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from situation_sim.household import Household, Obj, ON_PERSON, OUT_OF_HOUSE, catalogue
from situation_sim.schedule import Bout
from situation_sim.simulate import RunResult, TraceLine, DAY_SECONDS, hhmm
from situation_sim.situation import DaySituation

DEFAULTS = {
    "pill_box": True,
    "car_keys": True,
    "arrivals_per_week": [1, 3],        # inclusive range, one draw per 7-day week
    "questions_first_hour": 5,          # after each place change: stratified over the first window,
    "first_window_minutes": [0, 60],    # ... which is this many minutes after the move (the first hour),
    "questions_rest_of_day": 2,         # ... and over the rest of the 24 h after the window
    "min_questions_within_24h": 30,     # raise questions_rest_of_day until this is met
    "max_questions_rest_of_day": 10,
    "background": "match_first_hour",   # background questions at random times when the object has NOT
                                        # moved in the last 24 h: as many per home as the first-hour burst
                                        # questions (or an integer), so a question about a rare mover does
                                        # not by itself say that it just moved
    "same_object_gap_min": 30,          # the bank's own rule for repeats about one object
    "waking_minutes": [7 * 60, 23 * 60],
}

# trips one would drive to; walks, runs, rides and dog walks are not car trips
CAR_TRIPS = ("errands", "groceries", "visit_friends", "gym", "lunch_out",
             "commute_work", "classes", "shift")
SETTLE_MINUTES = 5          # simulate.end_bout: a gap longer than this and the resident settles


@dataclass
class RareMove:
    object_id: str
    day: int
    minute: int
    t: int
    src: str
    dest: str
    kind: str              # "pill_box_out" | "pill_box_back" | "keys_depart" | "keys_arrive"
    cause_id: str          # a cause id for the truth row (check 2 counts objects per cause)
    resident: str          # whose action it is
    cause_room: Optional[str]   # where that resident is seen at the time (None when away)
    words: str
    place_changed: bool    # dest differs from where it lay before the car trip / spell
    counts_as_rare_move: bool   # a window of questions follows this move

    def to_json(self) -> dict:
        d = asdict(self)
        d["hhmm"] = hhmm(self.minute)
        return d


def settings(cfg: Optional[dict]) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None})
    return out


# --------------------------------------------------------------------------- helpers

def _bouts_of(bouts: Sequence[Bout], resident: str) -> List[Bout]:
    return sorted((b for b in bouts if b.resident == resident), key=lambda b: (b.start, b.end))


def _spell(sits: Sequence[DaySituation], event: str = "unwell_spell") -> Optional[Tuple[str, int, int]]:
    """(ill resident, first sick day, first well day after the spell) or None."""
    first = None
    who = None
    for s in sits:
        hit = [c for c in s.causes if c.kind == "event" and c.event == event and c.resident]
        if hit and first is None:
            first, who = s.day_index, hit[0].resident
        if first is not None and who is not None and not any(c.resident == who for c in hit):
            return who, first, s.day_index
    if first is not None:
        return who, first, len(sits)      # the spell runs to the end of the month: no way back
    return None


def _holds_small(kind: str) -> bool:
    """May a spot of this kind hold a small thing like a set of keys? Read straight from the
    catalogue's size table: the new classes are not in objects.yaml (adding them there would
    change every household's random draws), so ``household.accepts`` cannot be asked."""
    if "floor" in kind:          # nobody puts the car keys down on the floor
        return False
    cat = catalogue()
    ok = cat["size_accepts"].get(kind)
    return ok is None or "*" in ok or "small" in ok or kind in cat["class_overrides"].get("keys", [])


def _small_spot_in(hh: Household, room: str, prefer: Sequence[str]) -> Optional[str]:
    for kind in prefer:
        r = hh.rec_of_kind(kind, room)
        if r is not None and _holds_small(kind):
            return r
    for r in hh.recs_in(room):
        if _holds_small(r.kind):
            return r.id
    return None


def _first_sits(hh: Household, trip: Bout, nxt: Optional[Bout]) -> Tuple[int, str, str]:
    """Where the resident is first recorded after a trip, mirroring simulate.end_bout:
    (minute, room, receptacle for small things in that room)."""
    res = hh.residents[trip.resident]
    if nxt is not None and not nxt.away and nxt.start - trip.end <= SETTLE_MINUTES:
        minute = max(nxt.start, trip.end)
        room = nxt.room
        rec = nxt.surface if (nxt.surface and _holds_small(hh.receptacles[nxt.surface].kind)) \
            else _small_spot_in(hh, room, ("kitchen_table", "desk", "dining_table", "coffee_table",
                                           "counter", "side_table", "nightstand"))
    else:
        minute = min(trip.end + SETTLE_MINUTES, 1439)
        room = "living" if "living" in hh.rooms else res.bedroom
        rec = _small_spot_in(hh, room, ("coffee_table", "side_table", "couch", "nightstand", "desk"))
    if rec is None:
        rec = hh.rec_of_kind("entry_table", "entry")
        room = "entry"
    # one minute after the resident is recorded there, so the cause strictly precedes the move
    return min(minute + 1, 1439), room, rec


# ----------------------------------------------------------------------- the post-pass

def add_rare_movers(hh: Household, sits: Sequence[DaySituation], res: RunResult,
                    cfg: Optional[dict], seed: int) -> dict:
    """Append the rare movers to a finished run. Returns the manifest written beside the bank."""
    st = settings(cfg)
    rng = random.Random(f"rare_movers:{seed}")
    eid = res.truth_rows[0]["episode_id"]
    n_days = len(sits)
    moves: List[RareMove] = []
    added: Dict[str, Obj] = {}
    names = {r.id: r.name.capitalize() for r in hh.residents.values()}

    def truth(obj: str, day: int, minute: int, rec: str, cause: str, causes: List[str],
              reason: str, carrier: Optional[str] = None) -> int:
        t = day * DAY_SECONDS + minute * 60
        last = max((r["t"] for r in res.truth_rows if r["object_id"] == obj), default=-1)
        if t <= last:
            t = last + 1
        row = {"kind": "truth", "episode_id": eid, "object_id": obj, "t": t, "receptacle_id": rec}
        if rec == ON_PERSON:
            row["carrier"] = carrier
        row.update({"cause": cause, "causes": sorted(set(causes)), "whim": False, "reason": reason,
                    "rare_mover": True})
        res.truth_rows.append(row)
        return t

    def trace(day: int, minute: int, text: str) -> None:
        sit, lines = res.trace_days[day]
        lines.append(TraceLine(minute, 2, text, True))
        lines.sort(key=lambda l: (l.minute, l.order))

    # ---------------------------------------------------------------- pill box
    spell = _spell(sits)
    if st["pill_box"] and spell is not None:
        who, first_sick, first_well = spell
        ill = hh.residents[who]
        usual_opts = [r for r in (hh.rec_of_kind("counter", "kitchen"),
                                  hh.rec_of_kind("medicine_cabinet", "bathroom")) if r]
        usual = rng.choice(usual_opts)
        bedside = hh.rec_of_kind("nightstand", ill.bedroom)
        oid = f"pill_box_{ill.name}"
        obj = Obj(oid, "pill_box", who, [usual], size="toiletry",
                  allowed=sorted({usual, bedside, ON_PERSON, OUT_OF_HOUSE}))
        added[oid] = obj
        truth(oid, 0, 0, usual, "initial", [], "")
        cause_id = f"unwell_spell:{who}"
        # out: the first bedroom bout of the first sick day, after waking
        day_bouts = _bouts_of(res.bouts_by_day[first_sick], who)
        out_b = next((b for b in day_bouts if not b.away and b.room == ill.bedroom
                      and b.activity not in ("sleep", "bedtime") and b.start >= 6 * 60), None)
        m_out = (out_b.start if out_b else 8 * 60 + 15) + 1     # the resident is seen there first
        t_out = truth(oid, first_sick, m_out, bedside, "rare_mover:pill_box_out", [cause_id],
                      f"{names[who]} is unwell and resting: the pill box is kept by the bed")
        moves.append(RareMove(oid, first_sick, m_out, t_out, usual, bedside, "pill_box_out", cause_id,
                              who, ill.bedroom, f"{names[who]} is unwell and starts the day in bed "
                              f"({out_b.activity if out_b else 'morning dose'})", True, True))
        trace(first_sick, m_out, f"{oid} → {bedside}: {names[who]} is unwell, the pill box is kept by the bed")
        if first_well < n_days:
            day_bouts = _bouts_of(res.bouts_by_day[first_well], who)
            back_b = next((b for b in day_bouts if not b.away and b.room != ill.bedroom
                           and b.start >= 6 * 60), None)
            m_back = (back_b.start if back_b else 8 * 60 + 30) + 1
            t_back = truth(oid, first_well, m_back, usual, "rare_mover:pill_box_back", [cause_id],
                           f"{names[who]} is up and about again: the pill box goes back")
            moves.append(RareMove(oid, first_well, m_back, t_back, bedside, usual, "pill_box_back",
                                  cause_id, who, back_b.room if back_b else None,
                                  f"{names[who]} is well again and out of the bedroom "
                                  f"({back_b.activity if back_b else 'morning'})", True, True))
            trace(first_well, m_back, f"{oid} → {usual}: {names[who]} is up and about again, the pill box goes back")

    # ---------------------------------------------------------------- car keys
    if st["car_keys"]:
        entry = hh.rec_of_kind("entry_table", "entry")
        oid = "car_keys_shared"
        allowed = {entry, ON_PERSON, OUT_OF_HOUSE}
        for r in hh.receptacles.values():
            if _holds_small(r.kind):
                allowed.add(r.id)
        obj = Obj(oid, "car_keys", None, [entry], size="small", allowed=sorted(allowed))
        added[oid] = obj
        truth(oid, 0, 0, entry, "initial", [], "")
        lo, hi = st["arrivals_per_week"]
        # candidate car outings: a trip in CAR_TRIPS plus any away bout that follows it directly
        outings: List[Tuple[int, Bout, Bout, Optional[Bout]]] = []   # (day, first leg, last leg, next home bout)
        for day in range(1, n_days):            # day 0 has no questions; keep it quiet
            for rid in sorted(hh.residents):
                bl = _bouts_of(res.bouts_by_day[day], rid)
                i = 0
                while i < len(bl):
                    b = bl[i]
                    if b.away and b.activity in CAR_TRIPS:
                        last = b
                        j = i + 1
                        while j < len(bl) and bl[j].away and bl[j].start - last.end <= SETTLE_MINUTES:
                            last = bl[j]
                            j += 1
                        nxt = bl[j] if j < len(bl) else None
                        if last.end < 23 * 60 + 30:
                            outings.append((day, b, last, nxt))
                        i = j
                    else:
                        i += 1
        where = entry
        for w0 in range(0, n_days, 7):
            week = [o for o in outings if w0 <= o[0] < w0 + 7]
            k = rng.randint(lo, hi)
            chosen: List[Tuple[int, Bout, Bout, Optional[Bout]]] = []
            for o in sorted(rng.sample(week, min(k, len(week))), key=lambda o: (o[0], o[1].start)):
                clash = any(c[0] == o[0] and not (o[2].end < c[1].start or o[1].start > c[2].end) for c in chosen)
                if not clash:
                    chosen.append(o)
            for day, first_leg, last_leg, nxt in chosen:
                rid = first_leg.resident
                cause_id = f"rare_mover:car_trip:{rid}:d{day}"
                t_dep = truth(oid, day, first_leg.start, OUT_OF_HOUSE, "rare_mover:keys_depart", [cause_id],
                              f"{names[rid]} takes the car to {first_leg.activity.replace('_', ' ')}")
                moves.append(RareMove(oid, day, first_leg.start, t_dep, where, OUT_OF_HOUSE, "keys_depart",
                                      cause_id, rid, None, f"{names[rid]} leaves by car for "
                                      f"{first_leg.activity.replace('_', ' ')}", True, False))
                trace(day, first_leg.start, f"{oid} → OUT_OF_HOUSE: {names[rid]} takes the car")
                minute, room, rec = _first_sits(hh, last_leg, nxt)
                t_arr = truth(oid, day, minute, rec, "rare_mover:keys_arrive", [cause_id],
                              f"{names[rid]} is back and sits down in the {room}: the keys stay there")
                moves.append(RareMove(oid, day, minute, t_arr, where, rec, "keys_arrive", cause_id, rid, room,
                                      f"{names[rid]} is back from {last_leg.activity.replace('_', ' ')} "
                                      f"and first sits in the {room}", rec != where, rec != where))
                trace(day, minute, f"{oid} → {rec}: {names[rid]} is back and sits down in the {room}")
                where = rec

    for oid, obj in added.items():
        hh.objects[oid] = obj
    hh.objects = dict(sorted(hh.objects.items()))
    return {
        "settings": st,
        "objects": {oid: {"cls": o.cls, "owner": o.owner, "usual_place": o.home[0], "allowed": o.allowed}
                    for oid, o in added.items()},
        "moves": [m.to_json() for m in moves],
        "n_place_changes": sum(1 for m in moves if m.counts_as_rare_move),
        "not_built": {"reading_glasses": "every calendar day is a weekday here, so the only "
                      "non-working days are the sick days; the glasses would be a daily mover "
                      "or a copy of the illness movers, not a rare mover"},
    }


# ------------------------------------------------------------------------ questions

def rare_mover_questions(manifest: dict, truth_at: Callable[[str, int], Optional[str]],
                         object_class: Dict[str, str], scored_days: Sequence[int], eid: str,
                         seed: int, stage_of_day: Dict[int, Optional[str]],
                         cfg: Optional[dict] = None) -> Tuple[List[dict], dict]:
    """Question rows about the rare movers. Returns (rows, counts)."""
    st = settings(cfg)
    rng = random.Random(f"rare_mover_questions:{seed}")
    wake_lo, wake_hi = st["waking_minutes"]
    gap = st["same_object_gap_min"] * 60
    scored = set(int(d) for d in scored_days)
    windows = [m for m in manifest["moves"] if m["counts_as_rare_move"]]
    objects = sorted(manifest["objects"])
    moves_of = {o: sorted(m["t"] for m in manifest["moves"] if m["object_id"] == o and m["counts_as_rare_move"])
                for o in objects}
    cause_of = {(m["object_id"], m["t"]): m["words"] for m in manifest["moves"]}

    def askable(obj: str, t: int) -> bool:
        day, minute = divmod(t, DAY_SECONDS)
        minute //= 60
        return (day in scored and wake_lo <= minute < wake_hi
                and truth_at(obj, t) not in (ON_PERSON, OUT_OF_HOUSE, None))

    def stratified(r: random.Random, grid: List[int], n: int, free: Callable[[int], bool]) -> List[int]:
        """One draw from each of n equal slices of the grid (skipping slices with nothing free)."""
        out: List[int] = []
        if not grid or n <= 0:
            return out
        for i in range(n):
            b = grid[int(i * len(grid) / n): int((i + 1) * len(grid) / n)]
            cands = [t for t in b if free(t) and all(abs(t - u) >= gap for u in out)]
            if cands:
                out.append(r.choice(cands))
        return out

    def draw(n_rest: int) -> List[Tuple[str, int]]:
        n_first = int(st["questions_first_hour"])
        r = random.Random(f"rare_mover_questions:{seed}:{n_first}:{n_rest}")
        taken: Dict[str, List[int]] = {o: [] for o in objects}

        def free(obj: str, t: int) -> bool:
            return all(abs(t - u) >= gap for u in taken[obj])

        for m in windows:
            obj, t0 = m["object_id"], m["t"]
            # the first window after the move, on a minute grid, then the rest of the 24 h
            w0, w1 = (int(x) * 60 for x in st["first_window_minutes"])
            first = [t for t in range(t0 + max(60, w0), t0 + w1 + 1, 60) if askable(obj, t)]
            rest = [t for t in range(t0 + w1 + 60, t0 + DAY_SECONDS + 1, 60) if askable(obj, t)]
            # the first-hour questions are closer together than the bank's 30-minute same-object
            # gap allows; inside this hour the gap is 10 minutes, outside it the usual 30 applies
            picked_first: List[int] = []
            for i in range(n_first):
                b = first[int(i * len(first) / n_first): int((i + 1) * len(first) / n_first)]
                cands = [t for t in b if all(abs(t - u) >= 600 for u in picked_first) and free(obj, t)]
                if cands:
                    picked_first.append(r.choice(cands))
            taken[obj] += picked_first
            taken[obj] += stratified(r, rest, n_rest, lambda t, o=obj: free(o, t))
        # background: random waking instants at which the object has not moved in the last 24 h,
        # as many per home as the first-hour questions just drawn (split evenly over the objects)
        w1 = int(st["first_window_minutes"][1]) * 60
        n_first = sum(1 for o, ts in taken.items() for t in ts
                      if any(0 < t - tm <= w1 for tm in moves_of[o]))
        n_bg = n_first if st["background"] == "match_first_hour" else int(st["background"])
        quiet = {o: [day * DAY_SECONDS + mnt * 60 for day in sorted(scored) for mnt in range(wake_lo, wake_hi, 5)]
                 for o in objects}
        quiet = {o: [t for t in ts if askable(o, t) and not any(0 < t - tm <= DAY_SECONDS for tm in moves_of[o])]
                 for o, ts in quiet.items()}
        background: Dict[str, List[int]] = {o: [] for o in objects}
        for i in range(n_bg):
            obj = objects[i % len(objects)]
            cands = [t for t in quiet[obj] if free(obj, t) and all(abs(t - u) >= gap for u in background[obj])]
            if cands:
                background[obj].append(r.choice(cands))
        for obj in objects:
            taken[obj] += background[obj]
        return sorted((o, t) for o, ts in taken.items() for t in ts)

    n_rest = int(st["questions_rest_of_day"])
    rows_t = draw(n_rest)
    def n_within(rows):
        return sum(1 for o, t in rows if any(0 < t - tm <= DAY_SECONDS for tm in moves_of[o]))
    while n_within(rows_t) < st["min_questions_within_24h"] and n_rest < st["max_questions_rest_of_day"]:
        n_rest += 1
        rows_t = draw(n_rest)

    rows: List[dict] = []
    per_day_k: Dict[int, int] = {}
    for obj, t in rows_t:
        day = t // DAY_SECONDS
        per_day_k[day] = per_day_k.get(day, 0) + 1
        prev = [tm for tm in moves_of[obj] if tm < t]
        last_move = max(prev) if prev else None
        within = last_move is not None and t - last_move <= DAY_SECONDS
        rank = None
        if within:
            rank = 1 + sum(1 for o2, t2 in rows_t if o2 == obj and last_move < t2 < t)
        rows.append({
            "kind": "question", "episode_id": eid, "question_id": f"d{day}r{per_day_k[day]:02d}",
            "object_id": obj, "t_query": t, "day_index": day, "object_class": object_class[obj],
            "moment": "rare_mover", "stage": stage_of_day.get(day),
            "rare_mover": True,
            "within_24h_of_rare_move": bool(within),
            "rare_move_t": last_move if within else None,
            "hours_since_rare_move": round((t - last_move) / 3600, 2) if last_move is not None else None,
            "rare_move_rank": rank,
            "in_first_hour_after_rare_move": bool(within and t - last_move <= int(st["first_window_minutes"][1]) * 60),
            "rare_mover_background_random": not within,
            "rare_move_cause": cause_of.get((obj, last_move)) if within else None,
        })
    counts = {"questions": len(rows), "within_24h": sum(1 for q in rows if q["within_24h_of_rare_move"]),
              "background_random": sum(1 for q in rows if q["rare_mover_background_random"]),
              "first_hour": sum(1 for q in rows if q["in_first_hour_after_rare_move"]),
              "questions_first_hour": int(st["questions_first_hour"]), "questions_rest_of_day_used": n_rest,
              "place_changes": len(windows),
              "per_object_within_24h": {o: sum(1 for q in rows if q["object_id"] == o and q["within_24h_of_rare_move"])
                                        for o in objects}}
    return rows, counts

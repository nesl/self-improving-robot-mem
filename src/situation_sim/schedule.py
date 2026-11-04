"""Daily schedule realization: role templates, habit-filled slots, pet
routines, event edits, jitter, anchors, fragmentation with kitchen breaks.

Timing constants are the copied values in ``timing_constants``.
"""
from __future__ import annotations

import copy
import math
import pathlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from situation_sim import timing_constants as tc
from situation_sim.household import Household, Resident
from situation_sim.situation import DaySituation

ELSEWHERE = "ELSEWHERE"
MIN_PER_DAY = 1440


def load_activities(path: Optional[pathlib.Path] = None) -> dict:
    path = path or pathlib.Path(__file__).with_name("activities.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


@dataclass
class Bout:
    resident: str
    activity: str
    start: int
    end: int
    room: str
    surface: Optional[str]
    uses: List[str]
    tidies: bool = False
    tidy_rooms: List[str] = field(default_factory=list)
    tidy_sizes: List[str] = field(default_factory=list)
    added_by: Optional[str] = None
    edited_by: List[str] = field(default_factory=list)
    habit: Optional[str] = None
    surface_by: Optional[str] = None      # episode that moved this activity to another surface
    bout_index: int = 0
    n_bouts: int = 1

    @property
    def away(self) -> bool:
        return self.room == ELSEWHERE


def resolve_room(token: str, res: Resident, hh: Household) -> str:
    if token == ELSEWHERE:
        return ELSEWHERE
    room = hh.resolve_room(token, res)
    if room is None:
        # room kind missing in this home: fall back to living
        return "living" if token not in ("kitchen", "bathroom", "entry") else token
    return room


def resolve_surface(token: Optional[str], room: str, hh: Household) -> Optional[str]:
    if token is None or room == ELSEWHERE:
        return None
    if token == "dining_or_kitchen_table":
        token = "dining_table" if room.startswith("dining") else "kitchen_table"
    if token == "balcony_or_side_table":
        token = "balcony_table" if room.startswith("balcony") else "side_table"
    return hh.rec_of_kind(token, room)


def _poisson(rng: random.Random, lam: float) -> int:
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def _home_at(blocks: List[dict], minute: int) -> bool:
    for b in blocks:
        if b["_room"] == ELSEWHERE and b["start"] <= minute < b["start"] + b["duration"]:
            return False
    return True


def _mk(activity: str, start: int, duration: int, res: Resident, hh: Household, templates: dict, **kw) -> dict:
    tmpl = templates[activity]
    b = {"activity": activity, "start": start, "duration": duration,
         "_room": resolve_room(tmpl["room"], res, hh), "_edited": [], "_added_by": None,
         "_habit": None, "skip_p": 0.0}
    b.update(kw)
    return b


def fill_slot(block: dict, res: Resident, hh: Household, acts: dict, daytype: str,
              rng: random.Random, used: set, hh_used: set,
              bias: Optional[Dict[str, float]] = None) -> List[dict]:
    """Choose what a resident does in a free slot from their habits. A habit
    fires at most once per resident per day; a household_once chore at most
    once per home per day. ``bias`` (from episodes active when the slot
    starts) multiplies a habit's or activity's weight by name."""
    templates = acts["activities"]
    bias = bias or {}
    slot = block["slot"]
    start, dur = _hhmm(block["start"]), int(block["duration"])
    hob, cho = acts["habits"]["hobbies"], acts["habits"]["chores"]

    def tilt(name: str, activity: str, p: float) -> float:
        return min(1.0, p * bias.get(name, 1.0) * (bias.get(activity, 1.0) if activity != name else 1.0))

    cands = []
    for h in res.hobbies:
        spec = hob[h]
        if slot in spec["slots"] and h not in used:
            cands.append((h, spec, tilt(h, spec["activity"], spec["p"][daytype])))
    for c in res.chores:
        spec = cho[c]
        if slot in spec["slots"] and c not in used and not (spec.get("household_once") and c in hh_used):
            cands.append((c, spec, tilt(c, spec["activity"], spec["p"][daytype] * (0.4 + res.tidiness))))
    # needs_shared: the home must own that shared object
    cands = [(n, s, p) for n, s, p in cands
             if not s.get("needs_shared") or any(o.cls == s["needs_shared"] for o in hh.objects.values())]
    cands.sort(key=lambda x: x[0])
    rng.shuffle(cands)
    for name, spec, p in cands:
        if rng.random() < p:
            used.add(name)
            if spec.get("household_once"):
                hh_used.add(name)
            d = min(int(spec["duration"]), dur)
            out = [_mk(spec["activity"], start, d, res, hh, templates, _habit=name)]
            if spec.get("then") and start + d < start + dur + 30:
                out.append(_mk(spec["then"], start + d, 15, res, hh, templates, _habit=name))
            return out
    # default
    opts = acts["slot_defaults"][slot]
    ws = [o["w"] * bias.get(o["activity"], 1.0) for o in opts]
    u = rng.random() * sum(ws)
    acc = 0.0
    for o, w in zip(opts, ws):
        acc += w
        if u <= acc:
            if o["activity"] == "none":
                return []
            return [_mk(o["activity"], start, dur, res, hh, templates)]
    return []


def _episode_edits(blocks: List[dict], res: Resident, sit: DaySituation) -> None:
    """Episode effects on blocks that start inside an active window:
    extend / shorten / shift / skip_p / surface. In place."""
    for b in blocks:
        for ep in sit.episodes_at(res.id, b["start"]):
            fx = ep.effects
            act = b["activity"]
            hit = False
            if act in fx.get("extend", {}):
                b["duration"] += int(fx["extend"][act]); hit = True
            if act in fx.get("shorten", {}):
                b["duration"] = max(5, int(round(b["duration"] * float(fx["shorten"][act])))); hit = True
            if act in fx.get("shift", {}):
                b["start"] += int(fx["shift"][act]); hit = True
            if act in fx.get("skip_p", {}):
                b["skip_p"] = min(1.0, float(b.get("skip_p", 0.0)) + float(fx["skip_p"][act])); hit = True
            if act in fx.get("surface", {}):
                b["_surface_kind"] = fx["surface"][act]; hit = True
            if hit and ep.id not in b["_edited"]:
                b["_edited"].append(ep.id)


def build_day(hh: Household, sit: DaySituation, events: Dict[str, dict],
              acts: dict, seed: int) -> List[Bout]:
    rng = random.Random(f"schedule:{seed}:{sit.day_index}")
    templates = acts["activities"]
    daytype = "weekend" if sit.is_weekend else "weekday"
    residents = sorted(hh.residents.values(), key=lambda r: r.id)

    # 1. nominal blocks: template + slots + pet routine, then skip, then event edits
    nominal: Dict[str, List[dict]] = {}
    hh_used: set = set()
    for res in residents:
        blocks: List[dict] = []
        used: set = set()
        # an event may take a whole free slot away ("slot:evening" in its remove list)
        gone_slots = {tok[5:] for c in sit.events_for(res.id)
                      for tok in events[c.event].get("schedule", {}).get("remove", []) if tok.startswith("slot:")}
        for raw in acts["schedules"][sit.role_of(res)][daytype]:
            if "slot" in raw:
                if raw["slot"] in gone_slots:
                    continue
                blocks += fill_slot(raw, res, hh, acts, daytype, rng, used, hh_used,
                                    bias=sit.slot_bias(res.id, _hhmm(raw["start"])))
                continue
            b = _mk(raw["activity"], _hhmm(raw["start"]), int(raw["duration"]), res, hh, templates)
            for k in ("skip_p", "mean_bouts"):
                if k in raw:
                    b[k] = raw[k]
            if raw.get("skip_by_tidiness"):
                b["skip_p"] = 0.8 * (1 - res.tidiness)
            blocks.append(b)
        _episode_edits(blocks, res, sit)
        nominal[res.id] = blocks
    if hh.pet:
        pet_cfg = acts["habits"]["pet"][hh.pet]
        others = [r.id for r in residents if r.id != hh.carer]
        for raw in pet_cfg["blocks"]:
            who = hh.carer
            if raw["who"] == "carer_or_other" and others and rng.random() < 0.4:
                who = others[rng.randrange(len(others))]
            res = hh.residents[who]
            b = _mk(raw["activity"], _hhmm(raw["start"]), int(raw["duration"]), res, hh, templates,
                    skip_p=float(raw.get("skip_p", 0.0)), _habit="pet")
            nominal[who].append(b)
    for res in residents:
        kept = []
        for b in nominal[res.id]:
            skip_p = min(float(b.get("skip_p", 0.0)), tc.MAX_SKIP_P)
            if skip_p and rng.random() < skip_p:
                continue
            kept.append(b)
        if any(b["activity"] == "lunch_out" for b in kept):
            kept = [b for b in kept if b["activity"] != "lunch"]
        # event edits
        for c in sit.events_for(res.id):
            sch = events[c.event].get("schedule", {})
            kept = [b for b in kept if b["activity"] not in sch.get("remove", [])]
            for act, mins in sorted(sch.get("delay", {}).items()):
                for b in kept:
                    if b["activity"] == act:
                        b["start"] += mins
                        b["_edited"].append(c.id)
            for act, factor in sorted(sch.get("shorten", {}).items()):
                for b in kept:
                    if b["activity"] == act:
                        b["duration"] = int(round(b["duration"] * factor))
                        b["_edited"].append(c.id)
            for act, mins in sorted(sch.get("extend", {}).items()):
                matches = [b for b in kept if b["activity"] == act]
                if matches:
                    matches[-1]["duration"] += mins
                    matches[-1]["_edited"].append(c.id)
        nominal[res.id] = kept

    # 2. event-added blocks with "who" resolved against nominal schedules
    for c in sorted(sit.causes, key=lambda c: c.id):
        if c.kind != "event":
            continue
        for add in events[c.event].get("schedule", {}).get("add", []):
            start = _hhmm(add["start"])
            who = add["who"]
            if who == "owner":
                targets = [c.resident]
            elif who == "all":
                targets = [r.id for r in residents]
            elif who == "all_home":
                targets = [r.id for r in residents if _home_at(nominal[r.id], start)]
            elif who in ("first_home", "one_home"):
                home = [r.id for r in residents if _home_at(nominal[r.id], start)]
                if home:
                    targets = [home[0]]
                elif who == "first_home":
                    def back(rid: str) -> int:
                        ends = [b["start"] + b["duration"] for b in nominal[rid]
                                if b["_room"] == ELSEWHERE and b["start"] <= start]
                        return max(ends) if ends else start
                    targets = [min((r.id for r in residents), key=lambda rid: (back(rid), rid))]
                else:
                    targets = []
            else:
                raise ValueError(who)
            for rid in targets:
                res = hh.residents[rid]
                tmpl = templates[add["activity"]]
                b = _mk(add["activity"], start, int(add["duration"]), res, hh, templates,
                        mean_bouts=add.get("mean_bouts", tmpl.get("mean_bouts", 1)))
                b["_room"] = resolve_room(add.get("room", tmpl["room"]), res, hh)
                b["_added_by"] = c.id
                nominal[rid].append(b)
    # 2b. episode-added bouts (a snack at onset, an evening work session)
    for ep in sorted(sit.episodes, key=lambda e: (e.start, e.id)):
        res = hh.residents[ep.resident]
        for add in ep.effects.get("add", []):
            at = add.get("at", 0)
            start = _hhmm(at) if isinstance(at, str) else ep.start + int(at)
            if start >= MIN_PER_DAY:
                continue
            tmpl = templates[add["activity"]]
            b = _mk(add["activity"], start, int(add["duration"]), res, hh, templates,
                    mean_bouts=add.get("mean_bouts", 1))
            b["_room"] = resolve_room(add.get("room", tmpl["room"]), res, hh)
            b["_added_by"] = ep.id
            nominal[ep.resident].append(b)

    # 3. jitter, anchors, fragmentation
    out: List[Bout] = []
    for res in residents:
        kept = sorted(nominal[res.id], key=lambda b: (b["start"], b["activity"]))
        realized = []
        for b in kept:
            tmpl = templates[b["activity"]]
            sigma = tc.JITTER_CLASSES_MIN[tmpl["jitter"]] * res.jitter_scale
            start = int(round(b["start"] + rng.gauss(0, sigma)))
            realized.append((start, int(b["duration"]), b))
        cursor = 0
        placed: List[list] = []
        for start, dur, b in realized:
            is_anchor = templates[b["activity"]]["jitter"] == "external"
            if is_anchor:
                while placed and placed[-1][1] > start:
                    ps, pe, pb = placed[-1]
                    if start - ps >= math.ceil((pe - ps) * tc.KEEP_BLOCK_SHARE):
                        placed[-1][1] = start
                        break
                    placed.pop()
            else:
                start = max(start, cursor)
            placed.append([start, start + dur, b])
            cursor = start + dur
        for s, e, b in placed:
            if s >= MIN_PER_DAY:
                continue
            e = min(e, MIN_PER_DAY)
            if b["activity"] == "sleep":
                e = MIN_PER_DAY
            tmpl = templates[b["activity"]]
            room = b["_room"]
            surface = resolve_surface(b.get("_surface_kind") or tmpl.get("surface"), room, hh)
            mean_bouts = max(tc.MIN_MEAN_BOUTS, min(tc.MAX_MEAN_BOUTS,
                                                   float(b.get("mean_bouts", tmpl.get("mean_bouts", 1)))))
            n = 1
            if mean_bouts > 1.0:
                n = max(1, _poisson(rng, mean_bouts))
            window = e - s
            while n > 1 and window < (2 * n - 1) * tc.MIN_BOUT_MINUTES:
                n -= 1
            if n == 1:
                pieces = [(s, e)]
            else:
                gap_total = int(window * 0.3)
                bout_total = window - gap_total
                cuts = sorted(rng.random() for _ in range(n - 1))
                lens, prev = [], 0.0
                for c in cuts + [1.0]:
                    lens.append(c - prev)
                    prev = c
                bout_lens = [max(tc.MIN_BOUT_MINUTES, int(bout_total * l)) for l in lens]
                gap = max(tc.MIN_BOUT_MINUTES, gap_total // (n - 1))
                pieces, t = [], s
                for bl in bout_lens:
                    pieces.append((t, min(t + bl, e)))
                    t += bl + gap
                pieces = [(a, bb) for a, bb in pieces if bb - a >= tc.MIN_BOUT_MINUTES and a < e]
                n = len(pieces)
            common = dict(tidies=bool(tmpl.get("tidies", False)),
                          tidy_rooms=[r for r in tmpl.get("tidy_rooms", [room]) if r in hh.rooms],
                          tidy_sizes=list(tmpl.get("tidy_sizes", [])),
                          added_by=b["_added_by"], edited_by=list(b["_edited"]), habit=b.get("_habit"),
                          surface_by=(next((e for e in b["_edited"] if e.startswith("ep:")), None)
                                      if b.get("_surface_kind") else None))
            for i, (a, bb) in enumerate(pieces):
                out.append(Bout(res.id, b["activity"], a, bb, room, surface, list(tmpl.get("uses", [])),
                                bout_index=i, n_bouts=n, **common))
                if i + 1 < len(pieces) and room != ELSEWHERE:
                    ga, gb = bb, pieces[i + 1][0]
                    if gb - ga >= tc.MIN_BOUT_MINUTES:
                        out.append(Bout(res.id, "break", ga, gb, "kitchen",
                                        hh.rec_of_kind("counter", "kitchen"), list(templates["break"]["uses"]),
                                        added_by=b["_added_by"], habit=b.get("_habit")))
    out.sort(key=lambda b: (b.start, b.resident, b.activity))
    return out

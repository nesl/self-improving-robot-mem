"""The simulation loop: walks every resident's bouts through the day, brings
objects to where they are used, decides where they go afterwards, and
records truth rows plus a human-readable trace.

Object model: while an activity runs, the objects it uses sit at the
activity's surface (a laptop on the desk); pocket items (phone, keys,
wallet) ride ON_PERSON. When the activity ends every used object that the
next activity does not need goes through the placement decision. A decision
whose destination is where the object already is produces no move (the
laptop stays on the desk) but still counts as a decision.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from situation_sim import timing_constants as tc
from situation_sim.household import Household, Obj, ON_PERSON, OUT_OF_HOUSE
from situation_sim.placement import World, decide, Decision
from situation_sim.schedule import Bout, ELSEWHERE, build_day
from situation_sim.situation import DaySituation

AWAY = "AWAY"
DAY_SECONDS = 86_400

WORDS: Dict[str, str] = {}   # activity -> plain words, filled from activities.yaml at init


def hhmm(minute: int) -> str:
    minute = min(minute, 1439)
    return f"{minute // 60:02d}:{minute % 60:02d}"


def words(activity: str) -> str:
    return WORDS.get(activity, activity.replace("_", " "))


@dataclass
class TraceLine:
    minute: int
    order: int
    text: str
    indent: bool = False


@dataclass
class RunResult:
    truth_rows: List[dict]
    resident_rows: List[dict]
    trace_days: List[Tuple[DaySituation, List[TraceLine]]]
    bouts_by_day: List[List[Bout]]
    stats: Dict[str, object] = field(default_factory=dict)
    visitors: List[dict] = field(default_factory=list)   # guests present: {day, start, end, room, n}


class Simulator:
    def __init__(self, hh: Household, sits: List[DaySituation], events: Dict[str, dict],
                 acts: dict, seed: int, episode_id: str):
        self.hh = hh
        self.sits = sits
        self.events = events
        self.acts = acts
        for name, tmpl in acts["activities"].items():
            WORDS[name] = tmpl.get("words", name.replace("_", " "))
        self.took_out: Dict[str, str] = {}   # object -> resident who carried it out
        self.seed = seed
        self.episode_id = episode_id
        self.rng = random.Random(f"simulate:{seed}")
        self.loc: Dict[str, str] = {}
        self.carrier: Dict[str, Optional[str]] = {}
        self.res_room: Dict[str, str] = {}
        self.in_use: Dict[str, List[str]] = {}      # resident -> objects used by the running bout
        self.truth: List[dict] = []
        self.resident_rows: List[dict] = []
        self.last_t: Dict[str, int] = {}
        self.n_decisions = 0
        self.n_moves = 0
        self.n_whim = 0
        self.notes: List[dict] = []   # non-move effects of causes (a forgotten pickup)

    # ---- recording ------------------------------------------------------
    def _t(self, day: int, minute: int) -> int:
        return day * DAY_SECONDS + minute * 60

    def move(self, obj_id: str, day: int, minute: int, rec: str, cause: str,
             causes: List[str], whim: bool = False, reason: str = "",
             carrier: Optional[str] = None, intended: Optional[str] = None,
             whim_p: float = 0.0) -> None:
        t = self._t(day, minute)
        if obj_id in self.last_t and t <= self.last_t[obj_id]:
            t = self.last_t[obj_id] + 1          # keep rows strictly ordered per object
        self.last_t[obj_id] = t
        row = {"kind": "truth", "episode_id": self.episode_id, "object_id": obj_id,
               "t": t, "receptacle_id": rec}
        if rec == ON_PERSON:
            row["carrier"] = carrier
        row["cause"] = cause
        row["causes"] = sorted(set(causes))
        row["whim"] = whim
        if whim:
            row["whim_p"] = whim_p
            row["intended"] = intended
        if reason:
            row["reason"] = reason
        self.truth.append(row)
        self.loc[obj_id] = rec
        self.carrier[obj_id] = carrier if rec == ON_PERSON else None

    def resident_at(self, res_id: str, day: int, minute: int, room: str) -> None:
        if self.res_room.get(res_id) == room:
            return
        self.res_room[res_id] = room
        self.resident_rows.append({"kind": "resident", "episode_id": self.episode_id,
                                   "resident_id": res_id, "t": self._t(day, minute),
                                   "room": room})

    # ---- helpers ----------------------------------------------------------
    def name(self, res_id: str) -> str:
        return self.hh.residents[res_id].name.capitalize()

    def expand_uses(self, res_id: str, tokens: List[str], sit: DaySituation,
                    bout: Optional[Bout] = None) -> List[Obj]:
        hh = self.hh
        own = hh.objects_of(res_id)
        shared = hh.objects_of(None)
        out: List[Obj] = []
        extra_carry = set()
        for c in sit.events_for(res_id):
            extra_carry.update(self.events[c.event].get("carry", []))
        tokens = list(tokens)
        if bout is not None:
            # episodes: extra things brought to this activity (the laptop to dinner)
            for ep in sit.episodes_at(res_id, bout.start):
                tokens += [cls for cls in ep.effects.get("bring", {}).get(bout.activity, [])
                           if cls not in tokens]
        for tok in tokens:
            if tok == "pocket":
                out += [o for o in own if o.pocket]
            elif tok == "outdoor":
                out += [o for o in own if o.outdoor and (not o.rain_only or o.cls in extra_carry)]
            elif tok in ("bag", "gym", "bag_leader"):
                kind = "bag" if tok != "gym" else "gym"
                for g in sorted(hh.groups.values(), key=lambda g: g.name):
                    if g.owner == res_id and g.kind == kind:
                        out.append(hh.objects[g.leader])
                        if tok != "bag_leader":
                            out += [hh.objects[m] for m in g.members]
            else:
                for alt in tok.split("|"):        # "laptop|tablet": first one the resident has
                    mine = [o for o in own if o.cls == alt]
                    found = mine if mine else [o for o in shared if o.cls == alt]
                    if found:
                        out += found
                        break
        seen = set()
        uniq = []
        for o in out:
            if o.id not in seen:
                seen.add(o.id)
                uniq.append(o)
        return uniq

    def blocked_now(self, bouts: List[Bout], minute: int, sit: DaySituation) -> Dict[str, str]:
        blocked: Dict[str, str] = {}
        for c in sorted(sit.causes, key=lambda c: c.id):
            if c.kind != "event":
                continue
            for rule in self.events[c.event].get("block", []):
                active = any(b.activity == rule["during"] and b.start <= minute <= b.end
                             for b in bouts)
                if active:
                    rec = self.hh.rec_of_kind(rule["receptacle"])
                    if rec:
                        blocked[rec] = c.id
        return blocked

    SHORT_TRIPS = ("walk", "run", "dog_walk", "cycle", "photo_walk")

    def standing_omission(self, obj: Obj, activity: str) -> bool:
        """Stable per (item, trip type): the resident never takes this on that
        trip. Phone and keys are never a standing omission (nobody habitually
        leaves them behind; only the per-departure forgetting applies); the
        wallet can be one only on short local trips; optional outdoor
        accessories (hat, scarf, sunglasses, umbrella, headphones...) can be
        one on any trip."""
        if obj.cls in ("phone", "keys"):
            return False
        if obj.cls == "wallet" and activity not in self.SHORT_TRIPS:
            return False
        r = random.Random(f"omission:{self.seed}:{obj.id}:{activity}")
        return r.random() >= tc.CARRY_P

    def is_leader(self, oid: str) -> bool:
        return any(g.leader == oid for g in self.hh.groups.values())

    # ---- core -------------------------------------------------------------
    def run(self) -> RunResult:
        hh = self.hh
        # t=0: everything at its primary home slot, residents asleep
        for o in sorted(hh.objects.values(), key=lambda o: o.id):
            self.move(o.id, 0, 0, o.home[0], "initial", [])
        for r in sorted(hh.residents.values(), key=lambda r: r.id):
            self.resident_at(r.id, 0, 0, r.bedroom)
        trace_days = []
        bouts_by_day = []
        visitors: List[dict] = []
        for sit in self.sits:
            bouts = build_day(hh, sit, self.events, self.acts, self.seed)
            lines = self.run_day(sit, bouts)
            trace_days.append((sit, lines))
            bouts_by_day.append(bouts)
            hosting = [b for b in bouts if b.activity == "host_guest"]
            if hosting:
                # guests are in the hosting room for the whole visit and can be seen there
                n = 2 + (self.seed + sit.day_index) % 2
                visitors.append({"day": sit.day_index, "start": min(b.start for b in hosting),
                                 "end": max(b.end for b in hosting), "room": hosting[0].room,
                                 "n": n, "cause": hosting[0].added_by})
        stats = {"placement_decisions": self.n_decisions,
                 "placement_moves": self.n_moves,
                 "whims": self.n_whim,
                 "whim_share_of_decisions": round(self.n_whim / max(1, self.n_decisions), 4),
                 "whim_share_of_moves": round(self.n_whim / max(1, self.n_moves), 4),
                 "notes": self.notes}
        return RunResult(self.truth, self.resident_rows, trace_days, bouts_by_day, stats, visitors)

    def run_day(self, sit: DaySituation, bouts: List[Bout]) -> List[TraceLine]:
        day = sit.day_index
        lines: List[TraceLine] = []
        by_res: Dict[str, List[Bout]] = {}
        for b in bouts:
            by_res.setdefault(b.resident, []).append(b)
        timeline = []
        for res_id, bl in sorted(by_res.items()):
            for i, b in enumerate(bl):
                nxt = bl[i + 1] if i + 1 < len(bl) else None
                timeline.append((b.start, 1, res_id, i, b, nxt))
                timeline.append((b.end, 0, res_id, i, b, nxt))
        # ends before starts at the same minute
        timeline.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        for minute, kind, res_id, i, b, nxt in timeline:
            if kind == 1:
                self.start_bout(sit, bouts, b, lines)
            else:
                self.end_bout(sit, bouts, b, nxt, lines)
        # end of day: anything still on a person is put down at home
        for res_id in sorted(by_res):
            carried = sorted(o for o, c in self.carrier.items() if c == res_id)
            for oid in carried:
                obj = self.hh.objects[oid]
                self.move(oid, day, 1439, obj.home[0], "end_of_day", [],
                          reason="put down before sleep")
                lines.append(TraceLine(1439, 2, f"{oid} → {obj.home[0]}: put down before sleep", True))
        lines.sort(key=lambda l: (l.minute, l.order))
        return lines

    def start_bout(self, sit: DaySituation, bouts: List[Bout], b: Bout,
                   lines: List[TraceLine]) -> None:
        day = sit.day_index
        who = self.name(b.resident)
        needed = self.expand_uses(b.resident, b.uses, sit, b)
        if b.away:
            self.start_trip(sit, b, needed, lines)
            return
        self.resident_at(b.resident, day, b.start, b.room)
        brought = []
        used: List[str] = []
        for o in needed:
            where = self.loc[o.id]
            if where == OUT_OF_HOUSE:
                continue
            if where == ON_PERSON and self.carrier[o.id] != b.resident:
                continue          # someone else has it
            causes = [b.added_by] if b.added_by else []
            if o.pocket:
                if where != ON_PERSON:
                    self.move(o.id, day, b.start, ON_PERSON, f"pickup:{b.activity}", causes,
                              carrier=b.resident)
                    brought.append(f"{o.id} from {where}")
            elif b.surface is not None and where != b.surface:
                # the activity surface, or the first receptacle in the room
                # that may hold this class; failing that it stays in hand
                target = b.surface if b.surface in o.allowed else None
                if target is None:
                    cands = [r.id for r in self.hh.recs_in(b.room) if r.id in o.allowed]
                    target = cands[0] if cands else ON_PERSON
                if target == ON_PERSON:
                    self.move(o.id, day, b.start, ON_PERSON, f"carry:{b.activity}", causes,
                              carrier=b.resident)
                else:
                    self.move(o.id, day, b.start, target, f"bring:{b.activity}", causes,
                              reason=f"brought to the {self.hh.receptacles[target].kind} for {words(b.activity)}")
                brought.append(f"{o.id} from {where}")
            used.append(o.id)
        self.in_use[b.resident] = used
        frag = f" (bout {b.bout_index + 1}/{b.n_bouts})" if b.n_bouts > 1 else ""
        txt = f"{who} — {words(b.activity)} in the {b.room}{frag}"
        if b.added_by:
            txt += f" [because of {b.added_by}]"
        elif b.habit and b.habit != "pet":
            txt += f" ({b.habit} habit)"
        elif b.edited_by:
            txt += f" [shifted by {', '.join(sorted(set(b.edited_by)))}]"
        if brought:
            txt += "; brings " + ", ".join(brought)
        lines.append(TraceLine(b.start, 1, txt))

    def start_trip(self, sit: DaySituation, b: Bout, needed: List[Obj],
                   lines: List[TraceLine]) -> None:
        day = sit.day_index
        who = self.name(b.resident)
        res = self.hh.residents[b.resident]
        self.resident_at(b.resident, day, b.start, AWAY)
        taken, forgotten, omitted = [], [], []
        for o in needed:
            where = self.loc[o.id]
            if where == OUT_OF_HOUSE:
                continue
            if where == ON_PERSON and self.carrier[o.id] != b.resident:
                continue
            optional = o.pocket or (o.outdoor and o.cls not in ("shoes", "jacket", "running_shoes") and not o.rain_only)
            if optional and self.standing_omission(o, b.activity):
                omitted.append(o.id)
                if where == ON_PERSON and self.carrier[o.id] == b.resident:
                    # it was in their pocket from the last activity but never
                    # goes on this kind of trip: put down before leaving
                    self.move(o.id, day, b.start, o.home[0], f"leave_home:{b.activity}", [],
                              reason=f"left at home, never taken on {words(b.activity)}")
                    lines.append(TraceLine(b.start, 1, f"{o.id} → {o.home[0]}: left at home before {words(b.activity)}", True))
                continue
            if o.pocket and where != ON_PERSON:
                p_f = res.forget_p
                distracted_src = sit.flag_source(b.resident, "distracted", b.start)
                p_extra = 2 * res.forget_p * res.mood_sensitivity if distracted_src else 0.0
                u = self.rng.random()
                if u < p_f + p_extra:
                    forgotten.append((o.id, where, u >= p_f))
                    continue
            causes = [c.id for c in sit.events_for(b.resident)
                      if o.cls in self.events[c.event].get("carry", [])]
            if b.added_by:
                causes.append(b.added_by)
            self.move(o.id, day, b.start, OUT_OF_HOUSE, f"trip:{b.activity}", causes)
            self.took_out[o.id] = b.resident
            taken.append(o.id)
        self.in_use[b.resident] = []
        trip = words(b.activity)
        txt = f"{who} leaves for {trip} (back {hhmm(b.end)})"
        if taken:
            txt += "; takes " + self.describe_taken(taken)
        if omitted:
            txt += f"; never takes {', '.join(omitted)} on {trip}"
        lines.append(TraceLine(b.start, 1, txt))
        for oid, where, distracted in forgotten:
            why = "distracted today" if distracted else "just forgot"
            lines.append(TraceLine(b.start, 1, f"forgets {oid} (still at {where}, {why})", True))
            if distracted:
                self.notes.append({"object_id": oid, "t": self._t(day, b.start),
                                   "event": "forgotten_on_departure",
                                   "causes": [distracted_src]})

    def describe_taken(self, taken: List[str]) -> str:
        hh = self.hh
        parts = []
        used = set()
        for g in sorted(hh.groups.values(), key=lambda g: g.name):
            if g.leader in taken:
                inside = [m for m in g.members if m in taken]
                used.add(g.leader)
                used.update(inside)
                parts.append(g.leader + (f" (with {', '.join(inside)})" if inside else ""))
        parts += [t for t in taken if t not in used]
        return ", ".join(parts)

    def place(self, oid: str, res_id: str, b: Bout, nxt: Optional[Bout], sit: DaySituation,
              world: World, sub: List[TraceLine], src_note: bool = False) -> Decision:
        hh = self.hh
        obj = hh.objects[oid]
        d = decide(hh, obj, hh.residents[res_id], b, nxt, sit, self.events, world, self.rng)
        self.n_decisions += 1
        if d.whim:
            self.n_whim += 1
        src = self.loc[oid]
        if d.dest != src:
            self.n_moves += 1
            self.move(oid, sit.day_index, b.end, d.dest, f"placement:{d.rule}", d.causes,
                      whim=d.whim, reason=d.reason, intended=d.intended, whim_p=d.whim_p)
            sub.append(TraceLine(b.end, 0, self.describe(oid, src if src_note else None, d), True))
        elif d.whim:
            # whim that happened to pick the current spot cannot occur (alts exclude it)
            pass
        return d

    def end_bout(self, sit: DaySituation, bouts: List[Bout], b: Bout, nxt: Optional[Bout],
                 lines: List[TraceLine]) -> None:
        day = sit.day_index
        hh = self.hh
        who = self.name(b.resident)
        sub: List[TraceLine] = []
        kept: List[str] = []
        needed_next = {o.id for o in self.expand_uses(b.resident, nxt.uses, sit, nxt)} if nxt else set()
        if nxt is not None:
            # episodes: classes that stay on the person between activities (the phone on a rough day)
            keep_cls = {cls for ep in sit.episodes_at(b.resident, b.end) for cls in ep.effects.get("keep", [])}
            if keep_cls:
                needed_next |= {o.id for o in hh.objects_of(b.resident) if o.cls in keep_cls
                                and (self.loc[o.id] == ON_PERSON and self.carrier[o.id] == b.resident
                                     or o.id in self.in_use.get(b.resident, []))}
        world = World(hh, self.loc, self.blocked_now(bouts, b.end, sit))
        if b.away:
            # everything of this resident's that went out comes back now
            if nxt is not None and nxt.start <= b.end:
                arrive = AWAY if nxt.away else nxt.room     # straight on to the next trip
            else:
                arrive = "entry"
            self.resident_at(b.resident, day, b.end, arrive)
            if arrive == "entry" and (nxt is None or nxt.start - b.end > 5):
                # nobody stands in the hallway for an hour: after putting
                # things down they settle somewhere until the next activity
                settle = "living" if "living" in hh.rooms else hh.residents[b.resident].bedroom
                self.resident_at(b.resident, day, min(b.end + 5, 1439), settle)
            back = sorted(o for o in hh.objects if self.loc[o] == OUT_OF_HOUSE
                          and self.took_out.get(o) == b.resident)
            order = sorted(back, key=lambda o: (0 if self.is_leader(o) else 1, o))
            placed_leader: Dict[str, Decision] = {}
            for oid in order:
                obj = hh.objects[oid]
                if oid in needed_next:
                    self.move(oid, day, b.end, ON_PERSON, f"keep:{nxt.activity}", [],
                              carrier=b.resident)
                    kept.append(oid)
                    continue
                g = hh.groups[obj.group] if obj.group else None
                if g and g.leader != oid and g.leader in placed_leader:
                    lead = placed_leader[g.leader]
                    if lead.dest in obj.allowed:
                        self.n_decisions += 1
                        self.n_moves += 1
                        self.move(oid, day, b.end, lead.dest, f"group:{g.name}", lead.causes,
                                  reason=f"stays in {g.leader}")
                        sub.append(TraceLine(b.end, 0, f"{oid} → {lead.dest}: stays in the {hh.objects[g.leader].cls}", True))
                        continue
                d = self.place(oid, b.resident, b, nxt, sit, world, sub)
                if self.is_leader(oid):
                    placed_leader[oid] = d
            if kept:
                sub.append(TraceLine(b.end, 0, f"keeps {', '.join(kept)} for {words(nxt.activity)}", True))
            lines.append(TraceLine(b.end, 0, f"{who} is back from {words(b.activity)}"))
            lines.extend(sub)
            return

        # a home activity ended: used objects still where this bout put
        # them, plus whatever is on the person
        used = [o for o in self.in_use.get(b.resident, [])
                if (self.loc[o] not in (ON_PERSON, OUT_OF_HOUSE)
                    and hh.receptacles[self.loc[o]].room == b.room)
                or (self.loc[o] == ON_PERSON and self.carrier[o] == b.resident)]
        used += [o for o, c in self.carrier.items() if c == b.resident and o not in used]
        self.in_use[b.resident] = []
        handled = set()
        for oid in sorted(used):
            handled.add(oid)
            if oid in needed_next:
                if self.loc[oid] == ON_PERSON:
                    kept.append(oid)
                continue
            self.place(oid, b.resident, b, nxt, sit, world, sub)
        # a tidy pass also returns whatever is lying around the rooms it covers
        if b.tidies:
            stray = sorted(o for o, r in self.loc.items()
                           if r not in (ON_PERSON, OUT_OF_HOUSE) and o not in handled
                           and hh.receptacles[r].room in b.tidy_rooms
                           and r != hh.objects[o].home[0]
                           and not hh.objects[o].static
                           and (not b.tidy_sizes or hh.objects[o].size in b.tidy_sizes))
            for oid in stray:
                self.place(oid, b.resident, b, nxt, sit, world, sub, src_note=True)
        if kept and nxt is not None and nxt.away:
            sub.append(TraceLine(b.end, 0, f"keeps {', '.join(kept)} for {words(nxt.activity)}", True))
        if sub:
            lines.append(TraceLine(b.end, 0, f"{who} finishes {words(b.activity)}"))
            lines.extend(sub)

    def describe(self, oid: str, src: Optional[str], d: Decision) -> str:
        frm = f" (from {src})" if src else ""
        if d.whim:
            txt = (f"{oid}{frm} → {d.dest}: WHIM — was heading for {d.intended} ({d.reason}) "
                   f"but landed on {d.dest} instead")
        else:
            txt = f"{oid}{frm} → {d.dest}: {d.reason}"
        if d.causes:
            txt += f" [{', '.join(d.causes)}]"
        return txt

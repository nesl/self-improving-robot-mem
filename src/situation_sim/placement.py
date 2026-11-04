"""Placement decision function.

Replaces the per-activity destination distribution. Given the activity that
just ended, what the resident does next, the day's active causes, where
things are, whether the usual spot is full or blocked, and the resident's
traits, it returns a destination plus a small, context-dependent whim.

Attribution is exact: a random draw that lands in the probability mass a
cause ADDED is attributed to that cause; a draw in the base mass is not.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from situation_sim.household import Household, Obj, Resident, ON_PERSON, OUT_OF_HOUSE, accepts
from situation_sim.schedule import Bout, ELSEWHERE
from situation_sim.situation import DaySituation

# Placement parameters (judgment defaults; written to hidden_state.json)
PARAMS = {
    # whim: p = clamp(base + untidy*(1-tidiness) + mood*(hurry*hurriedness + distract*distraction))
    "whim_base": 0.04,
    "whim_untidy": 0.12,
    "whim_hurry": 0.12,
    "whim_distract": 0.08,
    "whim_min": 0.02,
    "whim_max": 0.35,
    # leave the object where it was used instead of putting it back
    "leave_untidy": 0.55,
    "leave_low_energy": 0.30,
    # coming home: dump things at the entry instead of walking them home
    "dump_untidy": 0.45,
    "dump_hurry": 0.40,
    # distracted: carry the object into the next activity's room
    "carry_next_distract": 0.30,
}

# entry spot a dumped object lands on, by object size class (class overrides first)
_DUMP_KIND = {"jacket": "entry_hook", "scarf": "entry_hook", "hat": "entry_hook", "handbag": "entry_table",
              "shoes": "entry_floor", "running_shoes": "entry_floor", "umbrella": "entry_floor"}
_DUMP_SIZE = {"wearable": "entry_floor", "bag": "entry_floor", "bulky": "entry_floor",
              "dish": "entry_table", "small": "entry_table", "device": "entry_table",
              "paper": "entry_table", "toiletry": "entry_table", "textile": "entry_hook",
              "kitchen_tool": "entry_table"}


@dataclass
class Decision:
    dest: str
    reason: str
    causes: List[str] = field(default_factory=list)
    whim: bool = False
    whim_p: float = 0.0
    intended: Optional[str] = None      # where it was heading before the whim
    rule: str = ""


class World:
    """What the decision can see of the current state (read-only view)."""

    def __init__(self, hh: Household, loc: Dict[str, str],
                 blocked: Dict[str, str]):
        self.hh = hh
        self.loc = loc              # object -> receptacle / ON_PERSON / OUT_OF_HOUSE
        self.blocked = blocked      # receptacle -> cause id blocking it now

    def count(self, rec: str) -> int:
        # an object riding inside its group leader (same receptacle as the
        # leader) does not take a slot of its own
        n = 0
        for o, r in self.loc.items():
            if r != rec:
                continue
            g = self.hh.objects[o].group
            if g and self.hh.groups[g].leader != o and self.loc.get(self.hh.groups[g].leader) == rec:
                continue
            n += 1
        return n

    def ok_for(self, rec: str, obj: Obj) -> bool:
        return rec in obj.allowed and self.free(rec)

    def full(self, rec: str) -> bool:
        return self.count(rec) >= self.hh.receptacles[rec].capacity

    def free(self, rec: str) -> bool:
        return rec not in self.blocked and not self.full(rec)


def _resolve_kind(hh: Household, kind: str, obj: Obj, res: Resident,
                  ended: Bout) -> Optional[str]:
    if kind == "home":
        return obj.home[0]
    prefer = []
    if ended.room != ELSEWHERE:
        prefer.append(ended.room)
    prefer += [res.bedroom, res.workspace]
    for room in prefer:
        r = hh.rec_of_kind(kind, room)
        if r is not None and r in obj.allowed:
            return r
    r = hh.rec_of_kind(kind)
    return r if (r is not None and r in obj.allowed) else None


def decide(hh: Household, obj: Obj, res: Resident, ended: Bout, nxt: Optional[Bout],
           sit: DaySituation, events: Dict[str, dict], world: World,
           rng: random.Random) -> Decision:
    now = ended.end                       # episodes and flags are read at the putdown minute
    st = sit.state_at(res.id, now)
    mood = res.mood_sensitivity
    d: Optional[Decision] = None
    active = sit.episodes_at(res.id, now)

    # 1. event placement rules (first matching rule of the first matching
    #    event in sorted order wins)
    for c in sorted(sit.events_for(res.id), key=lambda c: c.id):
        for rule in events[c.event].get("placement", []):
            if rule["class"] != obj.cls:
                continue
            after = rule["after"]
            hit = (ended.activity in after or "any" in after
                   or ("any_return" in after and ended.away))
            if not hit:
                continue
            dest = _resolve_kind(hh, rule["to"], obj, res, ended)
            if dest is None:
                continue
            d = Decision(dest, rule["note"], [c.id], rule="event")
            break
        if d:
            break

    # 1b. episode placement preferences (windowed, same shape as event rules)
    if d is None:
        for ep in active:
            for rule in ep.effects.get("prefer", []):
                if rule["class"] != obj.cls:
                    continue
                after = rule["after"]
                hit = (ended.activity in after or "any" in after
                       or ("any_return" in after and ended.away))
                if not hit:
                    continue
                dest = _resolve_kind(hh, rule["to"], obj, res, ended)
                if dest is None:
                    continue
                d = Decision(dest, rule["note"], [ep.id], rule="episode")
                break
            if d:
                break

    # 2. tidy pass: everything goes home
    if d is None and ended.tidies:
        d = Decision(obj.home[0], "tidied away to its usual place",
                     [ended.added_by] if ended.added_by else [], rule="tidy")

    # 3. coming home from a trip
    if d is None and ended.away:
        home = obj.home[0]
        p_base = PARAMS["dump_untidy"] * (1 - res.tidiness)
        late_src = sit.flag_source(res.id, "running_late", now)
        p_hurry = PARAMS["dump_hurry"] * mood if late_src else 0.0
        u = rng.random()
        dump_kind = _DUMP_KIND.get(obj.cls, _DUMP_SIZE.get(obj.size, "entry_table"))
        dump_rec = hh.rec_of_kind(dump_kind, "entry")
        if (u < p_base + p_hurry and dump_rec is not None and dump_rec != home
                and dump_rec in obj.allowed):
            if u < p_base:
                d = Decision(dump_rec, "dropped at the door instead of being put away", [],
                             rule="dump")
            else:
                d = Decision(dump_rec, "dumped at the door, running late",
                             [late_src], rule="dump")
        else:
            d = Decision(home, "put away in its usual place after the trip", [], rule="home")

    # 4. a home activity ended
    if d is None:
        # distracted: carry it into the next room
        distracted_src = sit.flag_source(res.id, "distracted", now)
        if (distracted_src and nxt is not None and not nxt.away
                and nxt.room != ended.room and nxt.surface is not None
                and nxt.surface in obj.allowed):
            if rng.random() < PARAMS["carry_next_distract"] * mood:
                d = Decision(nxt.surface, f"carried along absent-mindedly into the {nxt.room}",
                             [distracted_src], rule="carry_next")
        if d is None:
            tired_src = sit.flag_source(res.id, "low_energy", now)
            p_base = PARAMS["leave_untidy"] * (1 - res.tidiness)
            p_tired = PARAMS["leave_low_energy"] * mood if tired_src else 0.0
            # episodes: a per-class nudge, positive (left out) or negative (tidier)
            ep_leave = [(ep, float(ep.effects["leave_p"][obj.cls])) for ep in active
                        if obj.cls in ep.effects.get("leave_p", {})]
            p_ep = sum(v for _, v in ep_leave)
            if p_ep < 0:                    # a tidy mood eats into the base chance
                p_base = max(0.0, p_base + p_ep)
                p_ep = 0.0
            u = rng.random()
            surface = ended.surface
            if (u < p_base + p_tired + p_ep and surface is not None and surface in obj.allowed
                    and surface != obj.home[0]):
                if u < p_base:
                    d = Decision(surface, "left where it was used", [], rule="leave")
                elif u < p_base + p_tired:
                    d = Decision(surface, "left where it was used, too tired to put it away",
                                 [tired_src], rule="leave")
                else:
                    ep = max((e for e, v in ep_leave if v > 0), key=lambda e: e.start)
                    d = Decision(surface, "left where it was used", [ep.id], rule="leave")
            elif obj.after_use and ended.activity not in ("chores", "tidy"):
                d = Decision(obj.after_use, "used, so it goes in the sink", [], rule="after_use")
            else:
                d = Decision(obj.home[0], "put back in its usual place", [], rule="home")
        if ended.added_by and ended.added_by not in d.causes:
            # the activity itself only exists because of an event
            d.causes.append(ended.added_by)

    # the activity happened somewhere unusual because of an episode (armchair, not couch)
    if ended.surface_by and d.dest == ended.surface and ended.surface_by not in d.causes:
        d.causes.append(ended.surface_by)

    # 5. usual destination occupied or blocked -> nearest alternative
    if not world.free(d.dest):
        why = (f"blocked ({world.blocked[d.dest]})" if d.dest in world.blocked
               else "full")
        blocker = world.blocked.get(d.dest)
        alts = [r for r in obj.home[1:] if world.ok_for(r, obj)]
        room = hh.receptacles[d.dest].room
        alts += [r.id for r in hh.recs_in(room) if world.ok_for(r.id, obj) and r.id not in alts]
        if alts:
            d.reason += f"; {d.dest} was {why}, so it went to {alts[0]}"
            d.dest = alts[0]
            if blocker and blocker not in d.causes:
                d.causes.append(blocker)

    # 6. whim: a small chance of landing somewhere else in the same room
    p = (PARAMS["whim_base"] + PARAMS["whim_untidy"] * (1 - res.tidiness)
         + mood * (PARAMS["whim_hurry"] * st["hurriedness"]
                   + PARAMS["whim_distract"] * st["distraction"]))
    p = max(PARAMS["whim_min"], min(PARAMS["whim_max"], p))
    d.whim_p = round(p, 3)
    if rng.random() < p:
        room = hh.receptacles[d.dest].room
        alts = [r.id for r in hh.recs_in(room) if r.id != d.dest and world.ok_for(r.id, obj)]
        if alts:
            d.intended = d.dest
            d.dest = rng.choice(alts)
            d.whim = True
    return d


def _res_activities(hh: Household, res, acts: dict, events: Dict[str, dict],
                    extra_roles: Sequence[str] = ()) -> List[str]:
    """Every activity this resident could perform: template blocks, slot
    defaults, their habits (and follow-ups), pet routine, event-added, break.
    ``extra_roles``: roles the calendar may switch this resident to."""
    names = set()
    for role in [res.role, *extra_roles]:
        for dt in ("weekday", "weekend"):
            for b in acts["schedules"][role][dt]:
                if "activity" in b:
                    names.add(b["activity"])
    for opts in acts["slot_defaults"].values():
        names.update(o["activity"] for o in opts if o["activity"] != "none")
    hob, cho = acts["habits"]["hobbies"], acts["habits"]["chores"]
    for h in res.hobbies:
        names.add(hob[h]["activity"])
    for c in res.chores:
        names.add(cho[c]["activity"])
        if cho[c].get("then"):
            names.add(cho[c]["then"])
    if hh.pet:
        names.update(b["activity"] for b in acts["habits"]["pet"][hh.pet]["blocks"])
    for ev in events.values():
        for add in ev.get("schedule", {}).get("add", []):
            names.add(add["activity"])
    names.add("break")
    return sorted(names)


def uses_obj(tmpl_uses: List[str], obj: Obj, hh: Household) -> bool:
    for tok in tmpl_uses:
        alts = tok.split("|")
        if obj.cls in alts:
            return True
    if obj.pocket and "pocket" in tmpl_uses:
        return True
    if obj.outdoor and "outdoor" in tmpl_uses:
        return True
    if obj.group:
        g = hh.groups[obj.group]
        if g.kind in tmpl_uses:
            return True
        if "bag_leader" in tmpl_uses and g.kind == "bag" and g.leader == obj.id:
            return True
    return False


def compute_allowed(hh: Household, acts: dict, events: Dict[str, dict],
                    extra_roles: Optional[Dict[str, Sequence[str]]] = None) -> None:
    """Fill ``obj.allowed``: every spot that accepts the object's class in the
    rooms where it lives or is used, plus event-rule destinations, plus
    ON_PERSON / OUT_OF_HOUSE. Static objects are allowed only at home.
    Written to hidden_state.json and used by the independent check that no
    object lands somewhere its rules forbid."""
    from situation_sim.schedule import resolve_room
    templates = acts["activities"]
    res_acts = {r.id: _res_activities(hh, r, acts, events, (extra_roles or {}).get(r.id, ()))
                for r in hh.residents.values()}
    for obj in sorted(hh.objects.values(), key=lambda o: o.id):
        if obj.static:
            obj.allowed = list(obj.home)
            continue
        rooms = {hh.receptacles[r].room for r in obj.home}
        users = [hh.residents[obj.owner]] if obj.owner else sorted(hh.residents.values(), key=lambda r: r.id)
        for res in users:
            for name in res_acts[res.id]:
                tmpl = templates[name]
                if uses_obj(tmpl.get("uses", []), obj, hh):
                    room = resolve_room(tmpl["room"], res, hh)
                    if room != ELSEWHERE:
                        rooms.add(room)
            for ev in events.values():
                for rule in ev.get("placement", []):
                    if rule["class"] == obj.cls and rule["to"] != "home":
                        for room in [res.bedroom, res.workspace] + hh.rooms:
                            r = hh.rec_of_kind(rule["to"], room)
                            if r is not None:
                                rooms.add(hh.receptacles[r].room)
                                break
        if obj.pocket or obj.outdoor or obj.group:
            rooms.add("entry")
        recs = sorted(r.id for r in hh.receptacles.values()
                      if r.room in rooms and (accepts(r.kind, obj.cls) or r.id in obj.home))
        obj.allowed = recs + [ON_PERSON, OUT_OF_HOUSE]
    for g in hh.groups.values():
        lead = set(hh.objects[g.leader].allowed)
        for m in g.members:
            o = hh.objects[m]
            o.allowed = sorted(set(o.allowed) | lead)
    for obj in hh.objects.values():
        obj.allowed = sorted(set(obj.allowed) | set(obj.home))

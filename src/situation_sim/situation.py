"""Day situation sampler: the hidden causes active on each day.

Three kinds of cause:
  * external events from ``events.yaml`` (household- or resident-scoped):
    they rewrite the day's schedule and move many objects at once
  * internal resident states (energy, hurriedness, distraction), resampled
    daily with carryover, and counted as an ACTIVE cause only in their
    extreme band (low energy, running late, distracted). Mild states still
    scale whim size but are not listed as causes.
  * episodes from ``episodes.yaml``: small moods and intentions that hold
    for a window of a few hours (``on_a_roll`` 13:40-17:20) and only tilt
    choices the simulator already makes. An episode may have several
    phases on consecutive days (a deadline tomorrow -> due today -> relief),
    so what starts tonight is still true tomorrow.

Nothing here depends on the day index except the weekday/weekend split of
event rates and schedules, and the day offsets of episode phases.
"""
from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import yaml

from situation_sim.household import Household

WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAYS = ["Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]   # the old five-day default
WEEKEND = {"Saturday", "Sunday"}
DEFAULT_DAY0 = "Wednesday"


def weekday_of(day_index: int, day0: str = DEFAULT_DAY0) -> str:
    """Calendar weekday of day ``day_index`` when day 0 is ``day0``."""
    return WEEK[(WEEK.index(day0) + day_index) % 7]


MIN_PER_DAY = 1440
WAKE_MINUTE = 6 * 60 + 30      # "wake" in an episode window

# Internal state dynamics (judgment defaults, recorded in hidden_state.json).
# Day-long flags are meant to be rare now that episodes carry the
# within-day moods; the noise is lower than the first checkpoint's 0.22.
STATE_CARRYOVER = 0.5      # weight of yesterday's value
STATE_NOISE_SD = 0.14
THRESH_LOW_ENERGY = 0.25
THRESH_HURRIED = 0.72
THRESH_DISTRACTED = 0.72
FLAGS = ("low_energy", "running_late", "distracted")
STATE_KEYS = ("energy", "hurriedness", "distraction")


def load_events(path: Optional[pathlib.Path] = None) -> Dict[str, dict]:
    path = path or pathlib.Path(__file__).with_name("events.yaml")
    with open(path) as f:
        return yaml.safe_load(f)["events"]


def load_episodes(path: Optional[pathlib.Path] = None) -> Dict[str, dict]:
    path = path or pathlib.Path(__file__).with_name("episodes.yaml")
    with open(path) as f:
        return yaml.safe_load(f)["episodes"]


def hhmm(minute: int) -> str:
    minute = max(0, min(MIN_PER_DAY, minute))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


@dataclass
class Cause:
    id: str                    # e.g. "rain", "late_work:resident_1", "low_energy:resident_2", "ep:on_a_roll:resident_1:d2"
    kind: str                  # "event" | "internal" | "episode"
    event: Optional[str]       # event type name for events, episode type name for episodes
    resident: Optional[str]    # None for household scope
    words: str                 # plain words for the trace


@dataclass
class Episode:
    """One phase of an episode, on one day, for one resident."""
    id: str                    # "ep:<type>:<resident>:d<day>[:<phase>]"
    type: str
    phase: str
    resident: str
    day: int
    start: int                 # minute of day, inclusive
    end: int                   # minute of day, exclusive
    effects: dict
    words: str
    chain: str                 # shared by every phase of one rolled episode

    def active(self, minute: int) -> bool:
        return self.start <= minute < self.end

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class DaySituation:
    day_index: int
    weekday: str
    is_weekend: bool
    causes: List[Cause]
    states: Dict[str, Dict[str, float]]   # resident -> {energy, hurriedness, distraction}
    flags: Dict[str, List[str]] = field(default_factory=dict)  # resident -> day-long internal flags
    episodes: List[Episode] = field(default_factory=list)
    stage: Optional[str] = None            # calendar stage name (regime search), None outside any stage
    roles: Dict[str, str] = field(default_factory=dict)   # resident -> role override for this day (calendar)

    def role_of(self, resident) -> str:
        """The role whose schedule the resident follows today (calendar override or base)."""
        return self.roles.get(resident.id, resident.role)

    def events_for(self, resident: str) -> List[Cause]:
        """Event causes that apply to this resident (household ones and their own)."""
        return [c for c in self.causes if c.kind == "event"
                and (c.resident is None or c.resident == resident)]

    # --- time-aware views: minute=None means "the day-long part only" ----
    def episodes_at(self, resident: str, minute: Optional[int]) -> List[Episode]:
        if minute is None:
            return []
        return [e for e in self.episodes if e.resident == resident and e.active(minute)]

    def has_flag(self, resident: str, flag: str, minute: Optional[int] = None) -> bool:
        return self.flag_source(resident, flag, minute) is not None

    def flag_source(self, resident: str, flag: str, minute: Optional[int] = None) -> Optional[str]:
        """The cause id that makes ``flag`` active for this resident at
        ``minute``: the day-long internal cause, else the first active
        episode carrying the flag, else None."""
        if flag in self.flags.get(resident, []):
            return f"{flag}:{resident}"
        for e in self.episodes_at(resident, minute):
            if flag in e.effects.get("flags", []):
                return e.id
        return None

    def state_at(self, resident: str, minute: Optional[int] = None) -> Dict[str, float]:
        st = dict(self.states[resident])
        for e in self.episodes_at(resident, minute):
            for k, dv in e.effects.get("state", {}).items():
                st[k] = st.get(k, 0.0) + float(dv)
        return {k: round(_clamp(v), 3) for k, v in st.items()}

    def slot_bias(self, resident: str, minute: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for e in self.episodes_at(resident, minute):
            for k, f in e.effects.get("slot_bias", {}).items():
                out[k] = out.get(k, 1.0) * float(f)
        return out

    def to_json(self) -> dict:
        out = {"day_index": self.day_index, "weekday": self.weekday,
               "is_weekend": self.is_weekend,
               "causes": [asdict(c) for c in self.causes],
               "states": self.states, "flags": self.flags,
               "episodes": [e.to_json() for e in self.episodes]}
        if self.stage is not None or self.roles:   # only calendar runs carry these keys
            out["stage"] = self.stage
            out["roles"] = dict(self.roles)
        return out


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _window(phase: dict, rng: random.Random) -> Tuple[int, int]:
    start = phase.get("start", "wake")
    if start == "wake":
        s = WAKE_MINUTE
    elif isinstance(start, list):
        a, b = _hhmm(start[0]), _hhmm(start[1])
        s = a + int(rng.random() * max(0, b - a))
    else:
        s = _hhmm(start)
    if "until" in phase:
        e = MIN_PER_DAY if phase["until"] == "sleep" else _hhmm(phase["until"])
    else:
        dur = phase.get("duration", 60)
        if isinstance(dur, list):
            dur = dur[0] + int(rng.random() * max(0, dur[1] - dur[0]))
        e = s + int(dur)
    return s, max(s + 1, min(MIN_PER_DAY, e))


def _eligible(spec: dict, r) -> bool:
    if "roles" in spec and r.role not in spec["roles"]:
        return False
    if "needs_hobby" in spec and spec["needs_hobby"] not in r.hobbies:
        return False
    return True


def stage_for(calendar: Optional[List[dict]], day_index: int) -> Optional[dict]:
    """The calendar stage covering ``day_index`` (inclusive ``days: [a, b]``),
    the last one listed if several overlap; None outside every stage."""
    hit = None
    for st in calendar or []:
        a, b = st["days"]
        if int(a) <= day_index <= int(b):
            hit = st
    return hit


def forced_causes(stage: dict, hh: Household, events: Dict[str, dict]) -> List[Cause]:
    """``force_events`` of a stage as the same Cause objects the random roll
    makes: household scope -> id = event name; resident scope -> id =
    ``event:resident_id`` with the resident's name in the words. ``resident``
    may be an id, a name, or ``household``/omitted for household scope."""
    out: List[Cause] = []
    residents = sorted(hh.residents.values(), key=lambda r: r.id)
    for fe in stage.get("force_events") or []:
        ename = fe["event"]
        ev = events[ename]
        who = fe.get("resident")
        if who in (None, "household") or ev["scope"] == "household":
            out.append(Cause(ename, "event", ename, None, ev["description"]))
            continue
        if str(who).lower() == "all":       # regime search: every resident the event can hit
            for r in residents:
                if r.role in ev.get("roles", []):
                    out.append(Cause(f"{ename}:{r.id}", "event", ename, r.id, f"{r.name.capitalize()}: {ev['description']}"))
            continue
        r = next((r for r in residents if r.id == who or r.name.lower() == str(who).lower()), None)
        if r is None:
            raise ValueError(f"calendar: unknown resident {who!r} for forced event {ename}")
        out.append(Cause(f"{ename}:{r.id}", "event", ename, r.id,
                         f"{r.name.capitalize()}: {ev['description']}"))
    return out


def sample_situations(hh: Household, seed: int, n_days: int,
                      events: Dict[str, dict],
                      episodes: Optional[Dict[str, dict]] = None,
                      day0: str = DEFAULT_DAY0,
                      calendar: Optional[List[dict]] = None) -> List[DaySituation]:
    """``calendar`` (regime search): a list of stages, each
    ``{name, days: [a, b], day_kind: weekday|weekend, force_events: [{event, resident}],
    suppress_random_events: bool, roles: {resident: role}}``. ``day_kind`` overrides
    the weekend/weekday schedule and rates for the stage's days (the weekday name
    stays the calendar's); forced events are added as ordinary Cause objects;
    ``suppress_random_events`` drops the day's randomly rolled events (the rolls are
    still drawn, so the internal-state stream is the same with or without it);
    ``roles`` switches a resident's daily schedule template for the stage. With
    ``calendar=None`` every draw and every output is unchanged."""
    episodes = load_episodes() if episodes is None else episodes
    rng = random.Random(f"situation:{seed}")
    ep_rng = random.Random(f"episodes:{seed}")
    days: List[DaySituation] = []
    prev: Dict[str, Dict[str, float]] = {}
    residents = sorted(hh.residents.values(), key=lambda r: r.id)
    # phases already rolled for later days: day -> episodes
    scheduled: Dict[int, List[Episode]] = {}
    for d in range(n_days):
        weekday = weekday_of(d, day0)
        is_we = weekday in WEEKEND
        stage = stage_for(calendar, d)
        if stage and stage.get("day_kind") in ("weekday", "weekend"):
            is_we = stage["day_kind"] == "weekend"
        rate_key = "weekend" if is_we else "weekday"
        causes: List[Cause] = []

        # --- external events (iterate in sorted order for determinism) ----
        for ename in sorted(events):
            ev = events[ename]
            p = ev["p_day"][rate_key]
            if ev["scope"] == "household":
                if rng.random() < p:
                    causes.append(Cause(ename, "event", ename, None, ev["description"]))
            else:
                for r in residents:
                    if r.role in ev.get("roles", []) and rng.random() < p:
                        causes.append(Cause(f"{ename}:{r.id}", "event", ename, r.id,
                                            f"{r.name.capitalize()}: {ev['description']}"))
        if stage:
            if stage.get("suppress_random_events"):
                causes = []
            forced = forced_causes(stage, hh, events)
            have = {c.id for c in causes}
            causes += [c for c in forced if c.id not in have]
            causes.sort(key=lambda c: c.id)   # the random path lists events in sorted name order
        # a resident off sick is not also working late
        sick = {c.resident for c in causes if c.event == "sick_day"}
        causes = [c for c in causes if not (c.event == "late_work" and c.resident in sick)]

        # --- internal states with carryover ------------------------------
        states: Dict[str, Dict[str, float]] = {}
        flags: Dict[str, List[str]] = {}
        for r in residents:
            base = {"energy": r.base_energy, "hurriedness": r.base_hurry,
                    "distraction": r.base_distraction}
            st = {}
            for k in sorted(base):
                yesterday = prev.get(r.id, base)[k]
                x = (STATE_CARRYOVER * yesterday + (1 - STATE_CARRYOVER) * base[k]
                     + rng.gauss(0, STATE_NOISE_SD))
                st[k] = x
            # event nudges (e.g. a sick day drains energy)
            for c in causes:
                if c.kind == "event" and (c.resident is None or c.resident == r.id):
                    for k, dv in sorted(events[c.event].get("internal", {}).items()):
                        st[k] += dv
            st = {k: round(_clamp(v), 3) for k, v in st.items()}
            states[r.id] = st
            fl = []
            nm = r.name.capitalize()
            if st["energy"] < THRESH_LOW_ENERGY:
                fl.append("low_energy")
                causes.append(Cause(f"low_energy:{r.id}", "internal", None, r.id,
                                    f"{nm} has low energy ({st['energy']:.2f}): leaves things where they were used instead of putting them back"))
            if st["hurriedness"] > THRESH_HURRIED:
                fl.append("running_late")
                causes.append(Cause(f"running_late:{r.id}", "internal", None, r.id,
                                    f"{nm} is running late all day ({st['hurriedness']:.2f}): rushed putdowns, things dumped at the door, more whim"))
            if st["distraction"] > THRESH_DISTRACTED:
                fl.append("distracted")
                causes.append(Cause(f"distracted:{r.id}", "internal", None, r.id,
                                    f"{nm} is distracted ({st['distraction']:.2f}): carries things into the next room absent-mindedly, more likely to forget pocket items"))
            flags[r.id] = fl
        prev = states

        # --- episodes: roll today's onsets, then collect every phase that
        #     lands today (rolled today or on an earlier day) ---------------
        today: List[Episode] = list(scheduled.pop(d, []))
        busy = {(e.resident, e.type) for e in today}
        for ename in sorted(episodes):
            spec = episodes[ename]
            p = spec["p_day"][rate_key]
            for r in residents:
                if not _eligible(spec, r) or (r.id, ename) in busy:
                    continue
                if ep_rng.random() >= p:
                    continue
                chain = f"ep:{ename}:{r.id}:d{d}"
                for ph in spec["phases"]:
                    pd = d + int(ph.get("day", 0))
                    if pd >= n_days:
                        continue
                    s, e = _window(ph, ep_rng)
                    pid = chain if len(spec["phases"]) == 1 else f"{chain}:{ph['id']}"
                    words = ph["words"].replace("{name}", r.name.capitalize())
                    ep = Episode(pid, ename, ph["id"], r.id, pd, s, e,
                                 ph.get("effects", {}), words, chain)
                    if pd == d:
                        today.append(ep)
                    else:
                        scheduled.setdefault(pd, []).append(ep)
        today.sort(key=lambda e: (e.start, e.resident, e.id))
        for e in today:
            causes.append(Cause(e.id, "episode", e.type, e.resident,
                                f"{e.words} ({hhmm(e.start)}–{hhmm(e.end)})"))
        sit = DaySituation(d, weekday, is_we, causes, states, flags, today)
        if stage:
            sit.stage = stage.get("name") or f"stage{calendar.index(stage)}"
            sit.roles = {}
            for who, role in (stage.get("roles") or {}).items():
                r = next((r for r in residents if r.id == who or r.name.lower() == str(who).lower()), None)
                if r is None:
                    raise ValueError(f"calendar: unknown resident {who!r} in roles")
                sit.roles[r.id] = role
        days.append(sit)
    return days

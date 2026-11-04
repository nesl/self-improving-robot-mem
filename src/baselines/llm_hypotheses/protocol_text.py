"""The ONE place the LLM prompts describe the sensing protocol.

Every LLM arm (hypothesis mixtures, treeLongLeaf, log reader, notebook
mixture and its dispatcher) tells the model how the robot observes the
home and what a look costs. Those mechanics differ by bank:

* the classic banks: a look opens ONE receptacle, costs one unit of a
  per-day look budget, and a resident can be looked at after being listed;
* the human inspector's protocol (``situation_eval`` banks, header
  ``protocol.room_level_looks``): a look is a whole ROOM, costs
  ``look_cost`` in the robot's current room and ``look_cost + travel_cost``
  anywhere else, budget resets at midnight with the robot back in its home
  base room, and pockets are never seen;
* the patrol protocol (``patrol`` banks, header ``protocol.free_look``): the
  robot patrols every room every ``patrol_hours`` hours, and before answering
  a question it may look at ONE room for free. No budget, no travel, no
  carry-over between questions.

The patrol banks also carry a resident intro card per resident
(``protocol.residents``) and, in the told arm, dated resident messages
(``protocol.hint_messages``); :func:`household_notes` renders both for any
prompt.

The bank header's ``protocol`` block (carried on ``EpisodeContext.protocol``)
selects the wording. With no block, every function below returns exactly
the sentences the prompts used before this module existed, so old banks
produce byte-identical prompts.

Edit the ROOM-LOOK wording here; nothing else in the prompt files states
costs or budgets.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from baselines.types import ON_PERSON, OUT_OF_HOUSE

# ---------------------------------------------------------------------------
# classic wording (unchanged from prompt.py / notebook_mixture.py)
# ---------------------------------------------------------------------------
CLASSIC_AWAY_SENTENCES = (
    "`ON_PERSON` means a resident who is in the house is carrying the "
    "object. `OUT_OF_HOUSE` means the object is not in the house. Neither "
    "can be chosen as the target of a look. A look at a receptacle also "
    "shows which residents are in that room. Looking at a resident shows "
    "what they are carrying, and is only possible after a look in the "
    "room they are in.")

CLASSIC_NOTEBOOK_LOOKS = (
    "it may spend a limited number of looks per day: a look opens one "
    "receptacle (it reveals everything inside and lists the residents in "
    "that room), or checks one resident the robot has just listed (it "
    "reveals everything they have on them)")

CLASSIC_DISPATCH_LOOKS = (
    "you decide whether the robot answers now or first spends one of "
    "today's looks (opening one receptacle, or checking one resident it "
    "has listed)")


_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_DAY0 = 0   # weekday index of day 0: Monday (the fleet convention) unless a bank says otherwise


def set_day0_weekday(protocol: Optional[Mapping[str, Any]]) -> None:
    """Read ``protocol.day0_weekday`` (e.g. "Wednesday") into the process-wide
    stamp offset used by every prompt's weekday names. One arm per process."""
    global _DAY0
    name = (protocol or {}).get("day0_weekday")
    _DAY0 = _WEEKDAYS.index(str(name)) if name in _WEEKDAYS else 0


def weekday_index(day: int) -> int:
    """Index into a Monday-first weekday tuple for simulated ``day``."""
    return (day + _DAY0) % 7


def day0_name() -> str:
    return _WEEKDAYS[_DAY0]


def is_room_look(protocol: Optional[Mapping[str, Any]]) -> bool:
    return bool(protocol and protocol.get("room_level_looks"))


def is_free_look(protocol: Optional[Mapping[str, Any]]) -> bool:
    """The patrol protocol: one free room look per question."""
    return bool(protocol and protocol.get("free_look"))


def is_patrol(protocol: Optional[Mapping[str, Any]]) -> bool:
    """A patrol bank: the free-look protocol, or the passive patrol stream
    (header ``protocol.patrol``, no looks at all)."""
    return bool(protocol and (protocol.get("free_look") or protocol.get("patrol")))


def patrol_sentence(protocol: Optional[Mapping[str, Any]]) -> str:
    """How the fixed patrol observes the home (patrol banks only)."""
    if not is_patrol(protocol):
        return ""
    h = protocol.get("patrol_hours")
    times = protocol.get("patrol_times") or []
    if times:
        every = "every day at " + ", ".join(times[:-1]) + f" and {times[-1]}" if len(times) > 1 else f"once a day at {times[0]}"
    else:
        every = f"every {h:g} hour{'s' if h != 1 else ''}" if h else "on a fixed schedule"
    fb = protocol.get("feedback_delay_min")
    tail = ("" if fb is None else
            (" At its nightly round the robot is told, for each of the day's questions, where the object "
             "turned out to be; those count as sightings at the time they were found." if fb < 0 else
             f" About {fb:g} minutes after each question the resident tells the robot where the object "
             f"turned out to be, and that counts as a sighting."))
    return (f"The robot patrols every room {every}, listing what is on every "
            f"receptacle and who is in the room; those listings arrive on their "
            f"own, whether or not a question is asked.{tail}")


QUESTIONS_ACTIVITY = (
    "WHEN THE ROBOT IS ASKED: the residents ask where something is when they are about to use it, a few "
    "minutes around the start of a meal, cooking, a work or study session, evening TV, a shower, bedtime, "
    "a hobby, and they ask about the things that activity uses (plates, glasses and bottles at meals; pans, "
    "knives and boards when cooking; laptop, mug, notebook and headphones at work; remote, blanket and "
    "snack bowl for TV; toiletries in the bathroom). After someone has left the house the robot is asked "
    "about chore things (vacuum cleaner, laundry basket, dog bowl, watering can). So what matters is where "
    "each object is WHILE IN USE and just before, at the time of day that use happens on a weekday and on "
    "a weekend, not only where it rests between uses.")


def questions_sentence(protocol: Optional[Mapping[str, Any]]) -> str:
    """How questions arise (activity-driven banks only)."""
    if protocol and protocol.get("question_moments") == "activity":
        return QUESTIONS_ACTIVITY
    return ""


def household_notes(protocol: Optional[Mapping[str, Any]],
                    day: Optional[int] = None,
                    names: Optional[Mapping[str, str]] = None) -> str:
    """The resident intro cards and, when the bank carries them (the told
    arm), the residents' dated messages up to ``day`` (all of them when
    ``day`` is None). Empty for banks without cards."""
    if not protocol:
        return ""
    cards = protocol.get("residents") or []
    lines = []
    if cards:
        lines.append("WHO LIVES HERE:")
        for c in cards:
            s = (f"  {c['name']} (in their {c['age_band']}) {c['occupation']}. "
                 f"{c['weekday']} {c['weekend']}")
            if c.get("hobbies"):
                s += f" Hobbies: {', '.join(c['hobbies'])}."
            if c.get("pet"):
                s += f" {c['name']} {c['pet']}."
            lines.append(s)
    msgs = [m for m in (protocol.get("hint_messages") or [])
            if day is None or int(m.get("day_index", 0)) <= day]
    if msgs:
        lines.append("MESSAGES FROM THE RESIDENTS (dated):")
        lines += [f"  {m['text']}" for m in msgs]
    return "\n".join(lines)


def _costs(protocol: Mapping[str, Any]):
    look = float(protocol.get("look_cost", 1))
    travel = float(protocol.get("travel_cost", 3))
    return look, look + travel


def _g(x: float) -> str:
    return f"{x:g}"


# ---------------------------------------------------------------------------
# room-look wording (the human inspector's protocol)
# ---------------------------------------------------------------------------
def away_sentences(protocol: Optional[Mapping[str, Any]] = None,
                   rmap: Optional[Mapping[str, str]] = None) -> str:
    """What the two away tokens mean and how people are seen. Goes into the
    vocabulary tables of every prompt."""
    r = rmap or {}
    on, out = r.get(ON_PERSON, ON_PERSON), r.get(OUT_OF_HOUSE, OUT_OF_HOUSE)
    if not is_room_look(protocol):
        return (CLASSIC_AWAY_SENTENCES.replace("`ON_PERSON`", f"`{on}`")
                .replace("`OUT_OF_HOUSE`", f"`{out}`"))
    people = ("A look also lists who is standing in that room — the residents "
              "by name, and visitors as a count of guests. What people carry "
              "is never visible: a resident cannot be looked at, so whether "
              "something is on a person has to be inferred (for example from "
              "their keys, shoes and jacket being in or out of the house).")
    return (f"`{on}` means a resident who is IN the house is carrying the object "
            f"(a phone in a pocket between activities). `{out}` means the object "
            f"has left the house — taken along by a resident on a trip out; it "
            f"comes back when they do. Neither can be chosen as the target of a "
            f"look. "
            f"A look is a whole ROOM: it reveals every receptacle in that room "
            f"and everything on each of them at that instant, including which "
            f"receptacles are empty. {people}")


def budget_sentences(protocol: Optional[Mapping[str, Any]],
                     budget_per_day: float,
                     home_base_room: Optional[str] = None) -> str:
    """The budget rule, for the household block of the notebook prompts and
    for any prompt that states the budget."""
    if not is_room_look(protocol):
        return f"LOOK BUDGET: {budget_per_day:g} looks per day, reset at midnight."
    if is_free_look(protocol):
        return (f"FREE LOOK: before answering each question the robot may look at "
                f"ONE room, at no cost. There is no daily budget and nothing carries "
                f"over between questions. {patrol_sentence(protocol)}")
    look, away = _costs(protocol)
    qpd = protocol.get("questions_per_day")
    base = (f" The robot starts every day in the {home_base_room}."
            if home_base_room else "")
    q = (f" There are about {qpd:g} questions a day, so the budget covers "
         f"roughly {int(budget_per_day // away)} looks that need travel or "
         f"{int(budget_per_day // look)} looks in the room the robot is "
         f"already in." if qpd else "")
    return (f"LOOK BUDGET: {budget_per_day:g} units per day, reset at midnight; "
            f"unspent units are lost. A look costs {_g(look)} when the room is "
            f"the one the robot is already in and {_g(away)} ({_g(look)} + "
            f"{_g(away - look)} travel) for any other room; after a look the robot "
            f"stays in that room. Looking again at a room already looked at "
            f"during the same question is free.{base}{q}")


def notebook_looks_clause(protocol: Optional[Mapping[str, Any]]) -> str:
    """The clause in the notebook agents' system prompt after 'the robot ...'."""
    if not is_room_look(protocol):
        return CLASSIC_NOTEBOOK_LOOKS
    if is_free_look(protocol):
        return ("it patrols every room on a fixed schedule (its listings arrive "
                "on their own) and, before answering each question, it may look "
                "at ONE room for free: a look reveals every receptacle in that "
                "room and everything on them, and lists who is in the room. "
                "Nobody's pockets can be looked into")
    look, away = _costs(protocol)
    return (f"it may spend a limited budget per day on looks: a look is a whole "
            f"room (it reveals every receptacle in that room and everything on "
            f"them, and lists who is in the room), costing {_g(look)} in the room "
            f"the robot is in and {_g(away)} anywhere else. Nobody's pockets can "
            f"be looked into")


def dispatch_looks_clause(protocol: Optional[Mapping[str, Any]]) -> str:
    """The clause in the dispatcher's system prompt."""
    if not is_room_look(protocol):
        return CLASSIC_DISPATCH_LOOKS
    if is_free_look(protocol):
        return ("you decide whether the robot answers now or first takes its one "
                "free look for this question. A look is a whole room: it reveals "
                "every receptacle in it and who is there, and costs nothing. "
                "Pockets cannot be looked into: whether an object is on a person "
                "is inferred, never seen")
    look, away = _costs(protocol)
    return (f"you decide whether the robot answers now or first spends part of "
            f"today's budget on a look. A look is a whole room: it reveals every "
            f"receptacle in it and who is there, and costs {_g(look)} if the room "
            f"is the one the robot is already in, {_g(away)} otherwise (the robot "
            f"then stays in that room). Pockets cannot be looked into: whether "
            f"an object is on a person is inferred, never seen")


def dispatch_rationing_clause(protocol: Optional[Mapping[str, Any]],
                              budget_per_day: float) -> str:
    """How the dispatcher is told to ration looks over the day."""
    if not is_room_look(protocol):
        return ("Unspent looks are lost at midnight, and a day has far more "
                "questions than looks, so looks are rationed across the day: "
                "what a gain is worth depends on what other gains this day is "
                "likely to offer.")
    if is_free_look(protocol):
        return ("The look is free and there is exactly one per question, so "
                "nothing is saved by answering without it: take the look whenever "
                "any room could change the answer or grade the panel; answer "
                "without looking only when no room could tell you anything.")
    look, away = _costs(protocol)
    qpd = protocol.get("questions_per_day")
    q = f" against about {qpd:g} questions" if qpd else ""
    return (f"Unspent budget is lost at midnight. Today's budget is "
            f"{budget_per_day:g} units{q}: that is {int(budget_per_day // away)} "
            f"looks if every one needs travel, more if the robot stays in one "
            f"room, so budget is rationed across the day and the robot's current "
            f"room matters — a question whose likely spots are in the room the "
            f"robot is already in is cheap to check. What a gain is worth depends "
            f"on what other gains this day is likely to offer.")


def asked_clause(protocol: Optional[Mapping[str, Any]]) -> str:
    """'many times a day' or the protocol's actual question count."""
    qpd = protocol.get("questions_per_day") if protocol else None
    return f"about {qpd:g} times a day" if qpd else "many times a day"


def dispatch_target_hint(protocol: Optional[Mapping[str, Any]]) -> str:
    """What the dispatcher's ``target`` field names."""
    if not is_room_look(protocol):
        return "<receptacle or resident id, for a look>"
    return ("<a receptacle id, for a look: the robot looks at the whole room "
            "that receptacle is in>")

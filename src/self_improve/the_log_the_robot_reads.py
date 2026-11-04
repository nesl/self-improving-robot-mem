"""The observation log, rendered for the model to read, instead of copied into its notes.

WHY THIS EXISTS. Counted over every claim-store run on disk on 2026-09-24: of 1,906
claims the model wrote, 91% name a place - "mug_felix is on the desk in the office" -
and six of them, 0.3%, say anything about a time, a condition or a change. 99% record
the condition they hold under as the single word "current". The memory is a table of
last-seen places.

That table already exists, for free, with dates attached and nothing garbled: it is the
robot's own observation record. Measured on the three homes where every arm has run the
same month, answering "where did I last see it" straight from that record - with no model
at all - gets the first room right for 72% of moved objects in the settled period and 70%
after the routine returns, against 44% to 52% for the arms that write notes. The notes
only draw level during the ten disrupted days.

So a memory whose content is places cannot be worth its tokens. This module gives the
model the record itself, which is strictly better than its copy of it, so that its notes
can hold the things a record cannot: when each person is where, what moves with what,
what changed and when, and where to look first.

Nothing here calls a model. It is rendering, and every line of it comes from the robot's
own looks - never from the simulator's ground truth.
"""
from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Sequence

HOW_MANY_SIGHTINGS_TO_SHOW = 40
# How many other things to name beside each sighting. A room holds up to nine spots and a
# look lists every one of them, so the whole contents of every room on every day would be
# most of the prompt. Six is enough to show a thing that always travels with another.
HOW_MANY_THINGS_BESIDE_IT = 6
# Above this many, the oldest are summarised rather than listed, so one object seen every
# day for a month does not push everything else out of the prompt.


def _clock(seconds_into_the_day: int) -> str:
    seconds_into_the_day %= 86400
    return f"{seconds_into_the_day // 3600:02d}:{(seconds_into_the_day % 3600) // 60:02d}"


def _what_else_was_there(eyes: Any, object_id: str, up_to_time: int
                         ) -> Dict[tuple, List[str]]:
    """What else the robot saw in the same room at the same moment.

    Added 2026-09-25: without it nothing in the record could ever suggest that two things
    move together, which is one of the four things the notes are asked to hold. Built from
    the robot's own looks only, and bounded by the same time as everything else here.
    """
    out: Dict[tuple, List[str]] = collections.defaultdict(list)
    for look in eyes.looks:
        if look.time > up_to_time:
            continue
        here = [s for s in look.sightings]
        if not any(s["object_id"] == object_id for s in here):
            continue
        for s in here:
            if s["object_id"] == object_id:
                continue
            out[(look.time, s["room"])].append(s["object_id"])
    return out


def what_you_have_seen_of(eyes: Any, object_id: str, up_to_time: int,
                          how_many: int = HOW_MANY_SIGHTINGS_TO_SHOW) -> List[str]:
    """Every sighting of one thing up to a moment, newest first, and where looking did
    not find it.

    `up_to_time` IS REQUIRED AND HAS NO DEFAULT, on purpose. Inside the live run the
    record only ever holds past looks, so a version without it happened to be safe there
    - and silently showed the whole month anywhere the record is loaded from a finished
    `looks.jsonl`, which is what the frozen-notes pass, a rebuilt snapshot and the
    re-answer pass all do. Measured on a real finished stream, one object with 52
    sightings over twelve days: a question on day 2 was shown 40 sightings of which 40
    were from its future, day 5 twenty-eight of forty, day 8 twenty-one of forty. The
    newest-first cap made it worse, because the ones it keeps are the most contaminated.
    A caller that has not thought about the time now cannot call this at all. Found by the
    research agent reviewing this module.
    """
    sightings = sorted([s for s in eyes.sightings_of(object_id) if s.time <= up_to_time],
                       key=lambda s: s.time, reverse=True)
    lines: List[str] = []
    if not sightings:
        lines.append(f"You have never seen {object_id}.")
    else:
        lines.append(f"Every time you have seen {object_id}, newest first "
                     f"({len(sightings)} in total). Each line also says what else you "
                     f"saw in that room at that moment, because things that move "
                     f"together are worth noticing:")
        beside = _what_else_was_there(eyes, object_id, up_to_time)
        for s in sightings[:how_many]:
            others = beside.get((s.time, s.room), [])
            with_it = ""
            if others:
                shown = others[:HOW_MANY_THINGS_BESIDE_IT]
                with_it = ("; also in the room: " + ", ".join(shown)
                           + (f", and {len(others) - len(shown)} other things"
                              if len(others) > len(shown) else ""))
            lines.append(f"- day {s.day} at {_clock(s.time)}: on the {s.place_id} "
                         f"in the {s.room}{with_it}")
        if len(sightings) > how_many:
            older = sightings[how_many:]
            where = collections.Counter(s.room for s in older)
            lines.append(f"- and {len(older)} older sightings, in "
                         + ", ".join(f"{room} ({n})" for room, n in where.most_common())
                         + f", the oldest on day {older[-1].day}")

    # Where looking did NOT find it, summarised by ROOM. One look at a nine-place kitchen
    # produces nine absence records and that is one piece of evidence, not nine.
    #
    # This counts looks by their target's name rather than through
    # `eyes.looks_that_did_not_find`, which expands each target into the places behind it.
    # At room granularity - the only one this study runs - the two agree, and this one
    # cannot print shelf identifiers under a heading that says rooms. The earlier version
    # of this function called the helper and then ignored it, which is the sort of dead
    # line that later gets "fixed" by using it.
    by_room: Dict[str, List[tuple]] = collections.defaultdict(list)
    ever_looked = set()
    for look in eyes.looks:
        if look.time > up_to_time:
            continue
        for target in look.targets:
            ever_looked.add(target["name"])
        if object_id in {s["object_id"] for s in look.sightings}:
            continue
        for target in look.targets:
            by_room[target["name"]].append((look.day, look.time))
    if by_room:
        lines.append("")
        lines.append("Rooms you have looked in without finding it:")
        for room, when in sorted(by_room.items(), key=lambda kv: max(kv[1]),
                                 reverse=True):
            newest = max(when)
            # THE CLOCK TIME, not only the day. Whether a failed look happened before or
            # after the moment being asked about is the whole of the evidence on a day when
            # something moved, and a day number cannot say which. Raised by the reasoner
            # running the ceiling probe, which had to assume it.
            lines.append(f"- {room}: {len(when)} look(s), most recently day "
                         f"{newest[0]} at {_clock(newest[1])}")
    # Rooms never entered at all are NOT the same as rooms entered and found empty, and
    # leaving them out of the list made the two indistinguishable. Also raised by the
    # reasoner: it could not tell whether a room was unvisited or merely never held the
    # thing, so it had no way to price it.
    never = sorted(set(eyes.household.rooms) - ever_looked)
    if never:
        lines.append("")
        lines.append("Rooms you have never looked in at all: " + ", ".join(never))
    return lines


def the_log_block(eyes: Any, object_id: str, up_to_time: int,
                  how_many: int = HOW_MANY_SIGHTINGS_TO_SHOW) -> str:
    """The same thing as one block of text, ready to drop into a prompt. `up_to_time` is
    required for the reason in `what_you_have_seen_of`."""
    return "\n".join([
        "YOUR RECORD. You do not have to remember where things are: everything you "
        "have ever seen is kept for you, and here is all of it for the thing you were "
        "asked about.",
        "",
        *what_you_have_seen_of(eyes, object_id, up_to_time, how_many),
    ])


def todays_sightings_block(eyes: Any, object_id: str, the_question_moment: int) -> str:
    """The asked object's sightings from earlier today, and the rooms looked in today without
    finding it. Lines as in `what_you_have_seen_of`; only looks from the start of the
    question's day to strictly before the question moment."""
    day_start = the_question_moment - the_question_moment % 86400
    in_window = lambda t: day_start <= t < the_question_moment  # noqa: E731
    sightings = sorted([s for s in eyes.sightings_of(object_id) if in_window(s.time)],
                       key=lambda s: s.time, reverse=True)
    lines: List[str] = ["TODAY'S RECORD. What you have seen earlier today of the thing you "
                        "were asked about.", ""]
    if not sightings:
        lines.append(f"You have not seen {object_id} earlier today.")
    else:
        lines.append(f"Every time you have seen {object_id} earlier today, newest first "
                     f"({len(sightings)} in total):")
        beside = _what_else_was_there(eyes, object_id, the_question_moment - 1)
        for s in sightings:
            others = beside.get((s.time, s.room), [])
            with_it = ""
            if others:
                shown = others[:HOW_MANY_THINGS_BESIDE_IT]
                with_it = ("; also in the room: " + ", ".join(shown)
                           + (f", and {len(others) - len(shown)} other things"
                              if len(others) > len(shown) else ""))
            lines.append(f"- day {s.day} at {_clock(s.time)}: on the {s.place_id} "
                         f"in the {s.room}{with_it}")
    by_room: Dict[str, List[tuple]] = collections.defaultdict(list)
    for look in eyes.looks:
        if not in_window(look.time):
            continue
        if object_id in {s["object_id"] for s in look.sightings}:
            continue
        for target in look.targets:
            by_room[target["name"]].append((look.day, look.time))
    if by_room:
        lines.append("")
        lines.append("Rooms you have looked in earlier today without finding it:")
        for room, when in sorted(by_room.items(), key=lambda kv: max(kv[1]), reverse=True):
            newest = max(when)
            lines.append(f"- {room}: {len(when)} look(s), most recently day "
                         f"{newest[0]} at {_clock(newest[1])}")
    return "\n".join(lines)


def the_log_block_for_a_replay(eyes: Any, object_id: str, the_question_moment: int,
                               how_many: int = HOW_MANY_SIGHTINGS_TO_SHOW) -> str:
    """The record as it stood STRICTLY BEFORE a question - the only safe form for a replay.

    `the_log_block`'s bound is inclusive, which is right in the live run: at the second room
    of a search the robot has just looked, and that look carries the question's own
    timestamp, so it must be shown. In a replay - a frozen pass, a re-answer pass, a rebuilt
    snapshot - the look that FOUND the object carries that same timestamp, so the inclusive
    bound hands over the answer. Measured on one question: at the question moment the block
    contains coffee_table_l1, the true place; one second earlier it does not.

    This exists as a separate name rather than as a convention about subtracting one, because
    a convention is something a caller has to remember and a name is something they have to
    choose. Found by the wave agent while checking an unrelated inconsistency.
    """
    return the_log_block(eyes, object_id, the_question_moment - 1, how_many)


WHAT_THE_PEOPLE_ARE_DOING = (
    "Think about what the people in this home are likely to be doing at this time of "
    "day, and where that puts the thing you are looking for. A thing is where somebody "
    "left it, and people are somewhere for a reason.")

"""Search-driven sensing: the robot's only information comes from its own searches.

WHAT CHANGED. Until 2026-09-24 the robot's only source of information was a
PATROL - one room a day at 13:00, plus a walkthrough on day 0 - and answering a
question taught it nothing. Here the patrol is gone. When the robot is asked
"where is X?" it goes and looks, and what it sees on that search is everything it
learns.

This closes the loop the study is about. Under a patrol, memory only helps at
answer time. Under search, a good memory sends the robot to the right room first,
so it finds the object AND collects a fresh observation, while a bad memory spends
its budget in the wrong rooms and learns less. Memory quality and sensing quality
stop being separable.

One question, in order:

  1. the robot is asked where object X is at time T;
  2. it opens up to k=3 rooms, one at a time, informed by its notes, learning what
     is in each and stopping early if it finds X;
  3. it answers - the true place if it found X, otherwise its best guess from its
     notes;
  4. everything it saw on that search enters its observation record, and the
     nightly note-writing runs on those observations instead of on a patrol.

TWO THINGS THIS FILE IS CAREFUL ABOUT.

  Every search step is described to the note-writer with
  `describe_look_for_the_model(record, only_these_objects=None)` - every sighting,
  every absence, every resident - for EVERY step and for ALL THREE sensing arms,
  not just the step that found the object and not just the quizzed objects. The
  patrol caller filtered to the quizzed objects, and that filtering is the fault
  being removed: measured, the look stream held 67 distinct objects in one
  household while the notes mentioned only the 13 quizzed ones and zero of the
  other 54. If only one arm were told everything, the arms would differ in what
  they are told as well as in where they go and nothing would be attributable.

  THE OBSERVATION CONFOUND IS LOGGED, NOT HIDDEN. A patrol gives one room-visit a
  day. Search at three rooms a question and eight questions a day gives up to
  twenty-four - more than twenty times the observation. So a search arm cannot be
  compared like-for-like with the patrol runs at
  results/self_improve/memory_factor_v1/, and it would beat them on coverage by
  construction. `observation_accounting` writes room-visits, distinct rooms,
  distinct objects and distinct asked-about objects per household per day, for the
  search arms and for the patrol run, so the ratio can be stated in one line. The
  three sensing arms all get the SAME budget, which is what makes their
  differences attributable to choosing rather than to looking more.

  python -m self_improve.search_driven --household hh_s0_t03 \
      --how "incremental edits" --sensing "memory-guided search"
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from baselines.patrol.llm import CONF_SCALE, CONF_SCHEMA, LLMClient, parse_conf
from baselines.types import DAY_SECONDS, ON_PERSON, OUT_OF_HOUSE
from self_improve.frozen_household import (BEYOND_REACH, FROZEN_BANKS, FrozenHousehold,
                                           LookTarget, period_of_day)
from self_improve.looking import TheHouseAsSeen, describe_look_for_the_model, hours_and_minutes
from self_improve.memory_notes import (A_PROFILE_AND_TOLD, A_PROFILE_OF_EACH_PERSON,
                                      CLAIM_STORE_PLUS_THE_LOG, CLAIM_STORE_PLUS_TODAY,
                                      TOLD_AND_ASKED_WHAT_CHANGES,
                                      THE_LOG_AND_THE_ROUTINE_DERIVED_ALLOWANCE,
                                      THE_LOG_AND_THE_ROUTINE_EIGHT,
                                      THE_LOG_AND_THE_ROUTINE_SIXTEEN,
                                      TOLD_THE_NIGHT_BEFORE, TOLD_ON_THE_FIRST_NIGHT,
                                      ACE_AS_PUBLISHED, MEMGPT_AS_PUBLISHED,
                                       A_WORKING_MEMORY_AND_AN_ARCHIVE, Notes,
                                       THE_LOG_AND_THE_ROUTINE, THE_LOG_ONLY,
                                       TOLD_IF_IT_WAS_RIGHT)
from self_improve.study_settings import LOCKED
from self_improve import write_the_notes as write_the_notes_module
from self_improve import the_log_the_robot_reads
from self_improve import what_the_robot_is_told as told
from self_improve.who_lives_here import (names_by_resident_id,
                                         who_lives_here_in_plain_words)
from self_improve.the_log_the_robot_reads import the_log_block
from self_improve.write_the_notes import write_the_notes
from self_improve.write_the_notes_about_the_routine import (
    write_the_notes_about_the_routine)
from self_improve.write_the_notes_memgpt import (what_it_can_see_of_its_memory,
                                                 write_the_notes_memgpt)
from self_improve.write_the_notes_memgpt_as_published import (
    what_it_can_see as what_the_block_shows,
    write_the_notes_memgpt_as_published)
from self_improve.write_the_notes_told_if_right import write_the_notes_told_if_right

NIGHT = 23 * 3600 + 30 * 60

MEMORY_GUIDED = "memory-guided search"
FIXED_ROTATION = "fixed rotation"
RANDOM = "random"
PRIOR_ONLY = "prior only, no notes"
LAST_SEEN = "last seen, no model"
# 2026-11-02. LAST_SEEN with today's sightings held back until night: searches and the no-find
# answer use the room list as it stood at the end of the previous night, like the nightly notes.
LAST_SEEN_NIGHTLY = "last seen, updated each night"
SENSING_ARMS = (MEMORY_GUIDED, FIXED_ROTATION, RANDOM, PRIOR_ONLY, LAST_SEEN, LAST_SEEN_NIGHTLY)
THE_NO_MODEL_ARMS = (LAST_SEEN, LAST_SEEN_NIGHTLY)

# WHAT CHANGED ON 2026-09-25, AND WHY THE OLD CELLS ARE UNDER `superseded_`. This arm used to
# keep ONE sighting per object, overwritten every time the object was seen. So its first guess
# was the room it was last seen in, and from the second guess on it had nothing left to say:
# that room was already opened and removed, so it fell through to the room where things of the
# same class were last seen, and then to a random room. Measured over 7,439 searches at 24
# questions a day in ten homes of 8 or 9 rooms, that second guess was right 178 times out of
# 1,332 - 13.4%, which is what a blind pick gets with seven or eight rooms left - and the
# third 172 of 1,154, 14.9%, again chance. So the arm named "last seen" was a first guess
# followed by a random walk, and every comparison against it after the first room flattered
# whatever it was compared with.
#
# It now keeps the whole trail: every room the object has been seen in, newest first, one
# entry per room. Guess k is the k-th most recent room that has not been opened yet. The
# class rule and the random draw are still there, but only once the trail runs out. This is
# what "go to where you last saw it" means when the search is allowed more than one room, and
# it is the fair opponent for a memory: the memory has to beat remembering, not beat
# remembering once and then guessing.
#
# It is still recency and nothing else. It does NOT prefer a sighting from this hour, or from
# a weekday, or from before somebody fell ill - a rule that did would be a model of the
# household's routine, which is the thing the memory arms are supposed to supply. Keeping it
# to pure recency is deliberate, not an oversight.
#
# LAST_SEEN is a few lines of code and NO MODEL CALLS AT ALL: walk back through the rooms
# where you have seen it, and answer with the shelf you last saw it on. It exists because
# scoring that rule on ANOTHER policy's look stream is not a clean counterfactual -
# the memory-guided arm's stream is generated by the model's own choices, so "where
# you last saw it" is a high-quality signal there partly BECAUSE the model chose well.
# Measured that way the rule reached 78.1% first-room against the model's 76.2% in the
# settled period and 73.5% against 63.2% on the movers, but a rule cannot be credited
# with a number it only achieves while riding on a better policy's observations. Run as
# its own arm it generates its own stream and the comparison is honest. It is free, so
# there is no reason not to have it.
#
# It writes no notes, so it is a REFERENCE and not a cell of the memory-format factor.
# Its `how_memory_is_written` is recorded as the format it is standing in for, purely so
# it lands beside the right cells.

# PRIOR_ONLY is the control that makes the memory-guided arm interpretable at all.
# Measured 2026-09-24: on day 1, with the notes EMPTY at the first question of the
# run, the memory-guided arm already found 92% of objects in 1.25 rooms, and
# household s0 then ran 7,8,7,6,8,7,8,7,7,8,8,7 of 8 from day 1 to day 12 - a flat
# line at ceiling, which is what a saturated prior looks like and not what learning
# looks like. So "memory-guided search finds 91% of objects in 1.3 rooms" cannot be
# attributed to the memory without pricing the prior separately. This arm is the
# identical chooser with the notes withheld at choice time: the gap between it and
# the memory-guided arm IS the memory contribution. Everything else - the budget, the
# stopping rule, the nightly note-writing, the answer step - is unchanged, so the
# notes still exist and are still used to answer; they are simply not consulted when
# deciding which room to walk into.

HOW_MEMORY_IS_WRITTEN = ("wholesale rewrite", "incremental edits",
                         CLAIM_STORE_PLUS_THE_LOG, CLAIM_STORE_PLUS_TODAY,
                         TOLD_IF_IT_WAS_RIGHT, THE_LOG_AND_THE_ROUTINE,
                         A_WORKING_MEMORY_AND_AN_ARCHIVE, ACE_AS_PUBLISHED,
                         MEMGPT_AS_PUBLISHED, THE_LOG_AND_THE_ROUTINE_DERIVED_ALLOWANCE,
                         TOLD_THE_NIGHT_BEFORE, TOLD_ON_THE_FIRST_NIGHT,
                         THE_LOG_AND_THE_ROUTINE_EIGHT,
                         THE_LOG_AND_THE_ROUTINE_SIXTEEN,
                         A_PROFILE_OF_EACH_PERSON, A_PROFILE_AND_TOLD,
                         TOLD_AND_ASKED_WHAT_CHANGES)

# The arms built on `the log and notes about the routine`: the same prompt, one thing changed.
THE_LOG_AND_THE_ROUTINE_FAMILY = (THE_LOG_ONLY,
                                  THE_LOG_AND_THE_ROUTINE,
                                  THE_LOG_AND_THE_ROUTINE_DERIVED_ALLOWANCE,
                                  THE_LOG_AND_THE_ROUTINE_EIGHT,
                                  THE_LOG_AND_THE_ROUTINE_SIXTEEN,
                                  A_PROFILE_OF_EACH_PERSON, A_PROFILE_AND_TOLD,
                                  TOLD_AND_ASKED_WHAT_CHANGES,
                                  TOLD_THE_NIGHT_BEFORE, TOLD_ON_THE_FIRST_NIGHT)
# How many notes a night each member of that family may write. The flat 16 is the module
# default; None means counted from what the night saw, the same rule the control uses.
NOTES_A_NIGHT = {THE_LOG_AND_THE_ROUTINE_EIGHT: 8,
                 THE_LOG_AND_THE_ROUTINE_SIXTEEN: 16,
                 THE_LOG_AND_THE_ROUTINE_DERIVED_ALLOWANCE: None}
# Which night the one sentence arrives on, as an offset from the day the routine changes and the
# day it changes back. 0 is the first changed day itself; -1 is the night before, when the robot
# has seen nothing of the change yet.
WHEN_IT_IS_TOLD = {TOLD_THE_NIGHT_BEFORE: -1, TOLD_ON_THE_FIRST_NIGHT: 0,
                   # The combined arm hears it on the same night as TOLD_ON_THE_FIRST_NIGHT,
                   # so the pair differ only in whether there is a profile to put it in.
                   A_PROFILE_AND_TOLD: 0,
                   # THE NIGHT BEFORE, on purpose: the whole point is a prediction made with no
                   # evidence at all, so it must be told before the first changed day, not on it.
                   TOLD_AND_ASKED_WHAT_CHANGES: -1}

# The three-way split of a wrong answer, computed from the STRUCTURED sighting
# records - how many times this arm had already seen this object at the place it
# actually was - never from a text matcher.
NEVER_SAW_IT = "never saw the object there"
SAW_IT_ONCE = "saw it there once and did not record it"
SAW_IT_OFTEN = "saw it there repeatedly and did not record it"

CHOOSE_SYSTEM = ("You help a home robot decide which room to walk into to find "
                 "something. Nobody tells the robot where anything is; looking is "
                 "the only way it learns. Answer with JSON only.")

ANSWER_SYSTEM = ("You help a home robot keep track of where household things are. "
                 "Read the robot's notes and answer with JSON only.")


# ------------------------------------------------------------- the questions --


def questions_spread_across_the_day(questions: Sequence[dict], how_many: int) -> List[dict]:
    """`how_many` of one day's questions, spread evenly across the day.

    NOT `questions[:n]`. That slice silently selected only the first hours (and, at
    the window level, only the first two days of a ten-day window) and biased every
    number that used it. The bank asks 24 questions a day in time order, so we take
    evenly spaced positions through the day's list.

    The same list for every arm, because it depends on nothing but the bank.
    """
    ordered = sorted(questions, key=lambda q: (q["t_query"], q["question_id"]))
    if how_many >= len(ordered) or how_many <= 0:
        return ordered
    step = len(ordered) / how_many
    taken = [ordered[min(len(ordered) - 1, int(i * step))] for i in range(how_many)]
    # int(i*step) can repeat when the day is short; keep it a set of distinct rows
    seen: Set[str] = set()
    out = []
    for q in taken:
        if q["question_id"] not in seen:
            seen.add(q["question_id"])
            out.append(q)
    return out


def answerable_questions_on_day(household: FrozenHousehold, day: int) -> List[dict]:
    """The day's questions whose answer a look could actually reach. Dropping the
    rest is the same rule `frozen_memory_test.questions_in_the_window` uses."""
    out = []
    for question in household.questions_on_day(day):
        if household.true_place_for_question(question) is None:
            continue
        if household.room_of_object(question["object_id"],
                                    question["t_query"]) == BEYOND_REACH:
            continue
        out.append(question)
    return out


DAYTIME_HOURS = tuple(range(8, 23))
SETTLED_DAYS = tuple(range(0, 14))
DISRUPTED_DAYS = tuple(range(14, 24))


def _commonest_place_over(household: FrozenHousehold, object_id: str,
                          days: Sequence[int]) -> Optional[str]:
    counts: collections.Counter = collections.Counter()
    for day in days:
        for hour in DAYTIME_HOURS:
            place = household.place_of_object(object_id, day * DAY_SECONDS + hour * 3600)
            if place and place not in (OUT_OF_HOUSE, ON_PERSON):
                counts[place] += 1
    return counts.most_common(1)[0][0] if counts else None


def the_movers(household: FrozenHousehold) -> Set[str]:
    """Asked-about objects whose usual daytime place changes when the resident
    falls ill. Measured over daytime hours, which is the locked exclusion rule
    (study_settings.exclusion_rule_measured)."""
    movers = set()
    for object_id in household.asked_objects:
        settled = _commonest_place_over(household, object_id, SETTLED_DAYS)
        disrupted = _commonest_place_over(household, object_id, DISRUPTED_DAYS)
        if settled and disrupted and settled != disrupted:
            movers.add(object_id)
    return movers


# --------------------------------------------------------- choosing the room --


def choice_schema(rooms: Sequence[str]) -> Dict[str, Any]:
    """REASONING FIRST, THEN THE ROOM, and the order is the whole point.

    Until 2026-09-25 this declared `room` before `why`. The server enforces the schema while it
    generates, and every call sets `enable_thinking: False`, so there was no thinking channel
    either: the model named a room after zero tokens of deliberation and then wrote a
    justification for a decision already made. Measured in the response cache rather than
    inferred from the schema - **2,206 of 2,206 cached room-choice completions emitted `room`
    first**. Our own answering schema had it the right way round the whole time, and nobody
    compared the two.

    It matters beyond reading the traces. Every arm that chooses a room from its notes got one
    forward pass with no room to think, while the rule with no model needs none - which is
    exactly the flaw that squashes the differences between memory designs toward the cheapest
    thing in the comparison.

    The cap is 1,200 characters, not 240. A reasoning field is not a memory field: 240 was the
    per-note limit that bounds the store, copied onto a field with no store to bound, and it cut
    **14,308 of 46,537 explanations, 30.7%**, nearly all mid-word. Everything run before this
    change stands as the ablation: does letting a memory reason before it acts change which
    memory wins?
    """
    return {
        "type": "object",
        "properties": {
            "why": {"type": "string", "maxLength": 1200},
            "room": {"type": "string", "enum": list(rooms)},
        },
        "required": ["why", "room"], "additionalProperties": False}


def choice_prompt(household: FrozenHousehold, notes: Notes, question: dict,
                  rooms_left: Sequence[str], already_opened: Sequence[str],
                  what_the_search_has_found_so_far: Sequence[str],
                  days_since_each_room_was_looked_in: Dict[str, Optional[int]],
                  step: int, budget: int,
                  show_it_the_notes: bool = True,
                  the_record_to_read: Optional[str] = None,
                  ask_what_the_people_are_doing: bool = False,
                  plainly: Optional[bool] = None) -> List[dict]:
    """THE INSTRUCTION COMES FIRST AND THE EXPLANATION AFTER.

    Not a stylistic choice. Measured on the note-writing prompt on 2026-09-24: with
    the explanation first this model answered a night of six fresh sightings with an
    empty edit list, on every variant tried. Any edit to this prompt must be
    re-smoked for the same failure - check that day 1 produces a real room and a
    real reason, not a default.

    At SEARCH time the robot sees all of its notes. The 8-line read budget is an
    ANSWER-time window, not amnesia; that is the same convention
    choose_where_to_look uses, so the two choosers are comparable.
    """
    if notes.how_memory_is_written == "wholesale rewrite":
        current = notes.newest_summary() or "(nothing written yet)"
    elif notes.how_memory_is_written == MEMGPT_AS_PUBLISHED:
        current = what_the_block_shows(notes, question["object_id"])
    elif notes.how_memory_is_written == A_WORKING_MEMORY_AND_AN_ARCHIVE:
        # Its working memory in full, plus its archive searched for the thing it was asked
        # about. That search IS this arm's retrieval, and it is the only way the archive
        # ever reaches a prompt.
        current = what_it_can_see_of_its_memory(notes, question["object_id"])
    else:
        current = ("\n".join(c.as_plain_words() for c in notes.claims_in_reading_order(
            question["object_id"])) if notes.claims else "(nothing written yet)")

    # `plainly` (residents, the three-things block, the length line, the other system prompt)
    # follows the record unless the caller sets it; the claim-store-plus arms set it False.
    if plainly is None:
        plainly = the_record_to_read is not None
    lines = (
        (told.the_people_who_live_here(household.asked_objects, household.resident_ids)
         + [""] + told.the_three_things_you_have(budget) + [""]) if plainly else []
    ) + [
        f"You have been asked where {question['object_id']} is, right now, on day "
        f"{question['day_index']} at {hours_and_minutes(question['t_query'])}.",
        "",
        f"Choose ONE room to walk into and search. This is search {step} of at most "
        f"{budget} you may make before you have to answer.",
        "",
        ("Pick the room where you are most likely to find it. When you walk into a room "
         "you see everything in it, so a room you have not been in for a while may have "
         "changed without your record showing it." if plainly else
         "Pick the room where you are most likely to find it. You see everything in "
         "a room you walk into, so a room you have not been in for a long time may "
         "have changed without your notes showing it."),
        "",
        "The rooms of this home and the spots in each:",
        *[f"- {room}: {', '.join(household.places_in_room[room])}"
          for room in household.rooms],
        "",
        "How long since you last looked in each room:",
        *[f"- {room}: " + ("never looked" if since is None
                           else "today" if since == 0
                           else "yesterday" if since == 1
                           else f"{since} days ago")
          for room, since in sorted(days_since_each_room_was_looked_in.items())],
        "",
    ]
    if already_opened:
        lines += [
            "On this search you have already looked in: " + ", ".join(already_opened),
            *(what_the_search_has_found_so_far or
              ["and did not find what you are looking for in any of them."]),
            "",
        ]
    if the_record_to_read:
        # The arm that reads its own record instead of a copy of it. The record goes
        # BEFORE the notes, because it is the better source for where things have been
        # and the notes are there to say what it cannot.
        lines += [the_record_to_read, ""]
    if ask_what_the_people_are_doing:
        # One line, not a paragraph. The earlier version restated what the three-tools
        # block already says and used "point at different rooms", which means nothing.
        lines += [the_log_the_robot_reads.WHAT_THE_PEOPLE_ARE_DOING, ""]
    if show_it_the_notes:
        lines += ["Your notes:", "", current, ""]
    else:
        # The prior-only control. The notes are withheld HERE and nowhere else: they
        # are still written every night and still used to answer. Nothing is said in
        # their place, because a line like "you have no notes" would be false and
        # would itself be an instruction.
        pass
    if plainly:
        lines += [told.how_much_you_may_write(1200), ""]
    lines += ["Rooms you may still choose: " + ", ".join(rooms_left)]
    return [{"role": "system",
             "content": told.CHOOSING_SYSTEM if plainly else CHOOSE_SYSTEM},
            {"role": "user", "content": "\n".join(lines)}]


@dataclass
class OneSearch:
    """One question, its search, and its answer. Every field structured."""
    question_id: str
    object_id: str
    object_class: str
    day: int
    time: int
    period: str
    is_a_mover: bool
    true_place: Optional[str]
    true_room: Optional[str]
    rooms_opened: List[str] = field(default_factory=list)
    why_each_room: List[str] = field(default_factory=list)
    n_rooms_opened: int = 0
    found_it: bool = False
    found_at_step: Optional[int] = None
    n_choice_calls_that_failed: int = 0
    answered_from: str = ""          # "the search found it" or "the notes"
    answer_place: Optional[str] = None
    answer_room: Optional[str] = None
    correct_place: Optional[bool] = None
    correct_room: Optional[bool] = None
    confidence: Optional[float] = None
    parse_status: str = ""
    n_lines_of_notes_available: int = 0
    the_read_budget_bit: bool = False
    # the structured three-way error split: how many times this arm had already seen
    # this object at the place it actually is, before this question was asked
    times_seen_there_before: int = 0
    error_kind: Optional[str] = None
    the_answer_was_told_what_the_search_ruled_out: bool = False
    the_answer_named_a_room_the_search_had_ruled_out: bool = False


CLAIM_STORE_PLUS = (CLAIM_STORE_PLUS_THE_LOG, CLAIM_STORE_PLUS_TODAY)


def _the_claim_store_plus_block(how: str, eyes: TheHouseAsSeen, object_id: str,
                                at_time: int) -> str:
    """The one block the claim-store-plus arms add: the whole log (as the log-and-notes arm
    sees it) or today's sightings before the question."""
    if how == CLAIM_STORE_PLUS_THE_LOG:
        return the_log_block(eyes, object_id, at_time)
    return the_log_the_robot_reads.todays_sightings_block(eyes, object_id, at_time)


def _room_of(household: FrozenHousehold, place: Optional[str]) -> Optional[str]:
    if not place:
        return None
    room = household.place_room.get(place)
    return room if room and room != "person_check" else None


def run_one_search(eyes: TheHouseAsSeen, household: FrozenHousehold, notes: Notes,
                   question: dict, sensing_arm: str, client: LLMClient,
                   budget: int, rotation: List[str], rotation_cursor: List[int],
                   rng: random.Random, movers: Set[str],
                   seen_before: Dict[Tuple[str, str], int],
                   allowed_places: Set[str],
                   tell_it_what_it_ruled_out: bool = True,
                   where_it_has_been: Optional[
                       Dict[str, List[Tuple[int, str, str]]]] = None,
                   the_trail_to_record_into: Optional[
                       Dict[str, List[Tuple[int, str, str]]]] = None
                   ) -> OneSearch:
    """One question end to end: choose rooms, look, then answer.

    The three sensing arms differ ONLY in how `room` is chosen. The budget, the
    stopping rule, what the look records, what the note-writer is later shown and
    the answering prompt are identical, which is what makes a difference between
    them attributable to choosing.
    """
    day = question["day_index"]
    at_time = question["t_query"]
    true_place = household.true_place_for_question(question)
    record = OneSearch(
        question_id=question["question_id"], object_id=question["object_id"],
        object_class=question.get("object_class", ""), day=day, time=at_time,
        period=period_of_day(day), is_a_mover=question["object_id"] in movers,
        true_place=true_place, true_room=_room_of(household, true_place),
        times_seen_there_before=seen_before.get((question["object_id"], true_place or ""), 0))

    where_it_has_been = {} if where_it_has_been is None else where_it_has_been
    # Sightings go into the trail the choice reads, unless a separate one is given (the nightly arm).
    record_into = where_it_has_been if the_trail_to_record_into is None else the_trail_to_record_into
    if sensing_arm == LAST_SEEN_NIGHTLY:
        _nothing_from_today(where_it_has_been, day)
    rooms_left = list(household.rooms)
    found_here: Optional[str] = None
    notes_on_the_search: List[str] = []
    time_of_day = at_time % DAY_SECONDS

    for step in range(1, budget + 1):
        if not rooms_left:
            break
        if sensing_arm in (MEMORY_GUIDED, PRIOR_ONLY):
            # MEMBERSHIP, NOT EQUALITY, AND IT WAS EQUALITY UNTIL 2026-09-26. The log block
            # and the question about what the people are doing are part of what this method IS,
            # but the test named ONE arm, so the five variants built on it - eight a night,
            # sixteen a night, the counted allowance, and both told arms - searched and answered
            # without the log while the arm they vary searched with it. Measured paired on three
            # homes, first room right: the log is worth +17.3 points in the settled window,
            # +26.0 during the illness and +20.8 on the day the routine goes back. So every
            # number from those five was a comparison against a different method, not an
            # ablation of this one. Anything named in THE_LOG_AND_THE_ROUTINE_FAMILY reads the
            # record, and a new variant gets it by being added to that tuple and nothing else.
            reads_its_record = (
                notes.how_memory_is_written in THE_LOG_AND_THE_ROUTINE_FAMILY)
            if notes.how_memory_is_written in CLAIM_STORE_PLUS:
                messages = choice_prompt(
                    household, notes, question, rooms_left, record.rooms_opened,
                    notes_on_the_search, eyes.days_since_each_room_was_looked_in(day),
                    step, budget, show_it_the_notes=(sensing_arm == MEMORY_GUIDED),
                    the_record_to_read=_the_claim_store_plus_block(
                        notes.how_memory_is_written, eyes, question["object_id"], at_time),
                    ask_what_the_people_are_doing=False, plainly=False)
            else:
                messages = choice_prompt(
                    household, notes, question, rooms_left, record.rooms_opened,
                    notes_on_the_search, eyes.days_since_each_room_was_looked_in(day),
                    step, budget, show_it_the_notes=(sensing_arm == MEMORY_GUIDED),
                    the_record_to_read=(
                        the_log_block(eyes, question["object_id"], at_time)
                        if reads_its_record else None),
                    ask_what_the_people_are_doing=reads_its_record)
            # 250 UNTIL 2026-09-25 AND IT WOULD HAVE RUINED THE NEXT WAVE. Raising the `why`
            # cap from 240 to 1,200 characters made this the binding limit: 1,200 characters is
            # about 385 tokens, and 250 cut it off. A cut-off reply is not valid JSON, and the
            # code just below hands the room to the FIXED ROTATION when the reply will not
            # parse - so a memory-guided step would quietly have become the cheapest baseline
            # in the study, which is the exact fault the reordering was meant to remove. Two
            # limits, and the one nobody looked at binds. 900 clears 385 with room for the
            # room name and the JSON around it.
            text, _ = client.complete(messages, choice_schema(rooms_left), max_tokens=900)
            room, why = None, ""
            if text:
                try:
                    parsed = json.loads(text)
                    if parsed.get("room") in set(rooms_left):
                        room, why = parsed["room"], (parsed.get("why") or "")
                except ValueError:
                    room = None
            if room is None:
                # It was asked and did not answer. Fall back to the rotation so the
                # search still happens, and record that it fell back rather than
                # pretending it chose.
                record.n_choice_calls_that_failed += 1
                room = _next_from_rotation(rotation, rotation_cursor, rooms_left)
                why = "the model was asked but did not answer, so the rotation chose"
            chosen_by = ("the model, from its notes" if sensing_arm == MEMORY_GUIDED
                         else "the model, with no notes: its prior alone")
        elif sensing_arm == FIXED_ROTATION:
            room = _next_from_rotation(rotation, rotation_cursor, rooms_left)
            why = "the fixed rotation, which does not look at the question"
            chosen_by = "the fixed rotation"
        elif sensing_arm == RANDOM:
            room = rng.choice(rooms_left)
            why = "uniformly at random"
            chosen_by = "random"
        elif sensing_arm in THE_NO_MODEL_ARMS:
            # THE REASON COMES BACK FROM THE CHOOSER. It used to be this one string on every
            # room, so the record said "the room it was last seen in" for 1,332 guesses that
            # the last sighting had no part in - a trace that states a rule the code did not
            # use is worse than no trace.
            room, why = _the_next_room_it_was_seen_in(
                question["object_id"], household, where_it_has_been, rooms_left, rng)
            chosen_by = "a few lines of code: walk back through where it has been seen"
        else:
            raise ValueError(f"unknown sensing arm {sensing_arm!r}")

        rooms_left.remove(room)
        record.rooms_opened.append(room)
        record.why_each_room.append(why)
        look = eyes.look([LookTarget(room, "room")], day, time_of_day, chosen_by)
        for sighting in look.sightings:
            seen_before[(sighting["object_id"], sighting["place_id"])] = \
                seen_before.get((sighting["object_id"], sighting["place_id"]), 0) + 1
            _remember_where_it_was_seen(record_into, sighting["object_id"],
                                        sighting["time"], sighting["place_id"],
                                        sighting["room"])
        here = [s for s in look.sightings if s["object_id"] == question["object_id"]]
        if here:
            found_here = here[0]["place_id"]
            record.found_it = True
            record.found_at_step = step
            break
        notes_on_the_search.append(
            f"- {room}: {question['object_id']} was not there. You saw: "
            + (", ".join(sorted({s["object_id"] for s in look.sightings})) or "nothing"))

    record.n_rooms_opened = len(record.rooms_opened)

    if record.found_it:
        record.answered_from = "the search found it"
        record.answer_place = found_here
        record.answer_room = _room_of(household, found_here)
        record.confidence = 1.0
        record.parse_status = "found"
    elif sensing_arm in THE_NO_MODEL_ARMS:
        if sensing_arm == LAST_SEEN_NIGHTLY:
            _nothing_from_today(where_it_has_been, day)
        # No model: the shelf you last saw it on, which is the newest entry on its trail. The
        # trail was updated by this search's own looks, so a failed search that saw the object
        # nowhere leaves the previous sighting standing, which is the honest behaviour of the
        # rule. Only the ANSWER uses the newest entry; the SEARCH walks the whole trail.
        trail = where_it_has_been.get(question["object_id"]) or ()
        mine = trail[0] if trail else None
        record.answered_from = "no model: the shelf it was last seen on"
        record.answer_place = mine[1] if mine else None
        record.answer_room = mine[2] if mine else None
        record.confidence = None
        record.parse_status = "mechanical" if mine else "never seen it at all"
    else:
        record.answered_from = "the notes"
        was_read = notes.what_the_robot_can_read(LOCKED.read_budget_lines,
                                                 about_object=question["object_id"])
        record.n_lines_of_notes_available = was_read.n_lines_available
        record.the_read_budget_bit = was_read.budget_bit
        record.the_answer_was_told_what_the_search_ruled_out = tell_it_what_it_ruled_out
        if notes.how_memory_is_written in CLAIM_STORE_PLUS:
            messages = answer_prompt(
                household, question, was_read.text,
                record.rooms_opened if tell_it_what_it_ruled_out else (),
                what_it_saw_while_searching=(notes_on_the_search
                                             if tell_it_what_it_ruled_out else ()),
                the_record_to_read=_the_claim_store_plus_block(
                    notes.how_memory_is_written, eyes, question["object_id"], at_time),
                plainly=False)
        else:
            messages = answer_prompt(
                household, question, was_read.text,
                record.rooms_opened if tell_it_what_it_ruled_out else (),
                what_it_saw_while_searching=(notes_on_the_search
                                             if tell_it_what_it_ruled_out else ()),
                the_record_to_read=(
                    the_log_block(eyes, question["object_id"], at_time)
                    # The same fault as at the choice step, and the same fix: the whole family
                    # reads the record, not the one arm whose name the test spelled out.
                    if notes.how_memory_is_written in THE_LOG_AND_THE_ROUTINE_FAMILY else None))
        # 400 until 2026-09-25, for the same reason: CONF_SCHEMA's `reasoning` now allows 1,200
        # characters, about 385 tokens, and the location and confidence follow it.
        text, _ = client.complete(messages, CONF_SCHEMA, max_tokens=900)
        place, confidence, _why, status = parse_conf(text, allowed_places)
        record.answer_place = place
        record.answer_room = _room_of(household, place)
        record.confidence = confidence
        record.parse_status = status
        # Whether the answer contradicts the search it just made. A robot that names
        # a room it has this minute searched and found empty is not using the
        # evidence it was handed, and no accuracy number would say so.
        record.the_answer_named_a_room_the_search_had_ruled_out = (
            record.answer_room in set(record.rooms_opened))

    if record.answer_place is not None and true_place is not None:
        record.correct_place = record.answer_place == true_place
        record.correct_room = record.answer_room == record.true_room
        if not record.correct_place:
            times = record.times_seen_there_before
            record.error_kind = (NEVER_SAW_IT if times == 0
                                 else SAW_IT_ONCE if times == 1 else SAW_IT_OFTEN)
    return record


def _in_words(n: int) -> str:
    """second, third, fourth ... for the reason a room was chosen, so a person reading a
    trace is not reading "the 2th most recent room"."""
    return {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh",
            8: "eighth", 9: "ninth", 10: "tenth"}.get(n, f"{n}th")


def _remember_where_it_was_seen(where_it_has_been: Dict[str, List[Tuple[int, str, str]]],
                               object_id: str, at_time: int, place_id: str,
                               room: str) -> None:
    """Put this sighting at the front of the object's trail, one entry per room.

    Seeing a thing again in a room it has been seen in before does not add an entry; it moves
    that room back to the front. So the trail is the rooms it has been seen in, most recently
    first, and its length is the number of DIFFERENT rooms rather than the number of sightings.

    Sightings arrive in time order, because the searches run forward through the month, and
    the whole trail depends on that. So it is checked rather than assumed: out of order, the
    front of the trail would not be the newest sighting and the answer would be wrong while
    everything still looked fine.
    """
    trail = where_it_has_been.setdefault(object_id, [])
    if trail and at_time < trail[0][0]:
        raise ValueError(
            f"sightings of {object_id} arrived out of order: {at_time} after {trail[0][0]}. "
            f"The trail is built on the assumption that they do not.")
    for i, (_t, _p, seen_in) in enumerate(trail):
        if seen_in == room:
            trail.pop(i)
            break
    trail.insert(0, (at_time, place_id, room))


def _nothing_from_today(trail_by_object: Dict[str, List[Tuple[int, str, str]]], day: int) -> None:
    """The nightly arm's guard: no sighting dated today may be in the list a choice reads."""
    today_starts = day * DAY_SECONDS
    for object_id, trail in trail_by_object.items():
        for t, _p, _r in trail:
            assert t < today_starts, (
                f"{object_id}: a sighting at {t} (day {t // DAY_SECONDS}) is in the room list "
                f"read on day {day}")


def _the_next_room_it_was_seen_in(object_id: str, household: FrozenHousehold,
                                  where_it_has_been: Dict[str, List[Tuple[int, str, str]]],
                                  rooms_left: Sequence[str],
                                  rng: random.Random) -> Tuple[str, str]:
    """The mechanical chooser, and the reason it chose. No model call.

    Walk back through the rooms this object has been seen in, newest first, and take the first
    one not opened yet on this search. Only when the trail runs out does it fall back: to the
    room where things of its class were last seen, which is the mechanical stand-in for the
    commonsense prior, and then to a seeded random room. The fallbacks are named in the reason
    rather than hidden, because on day 1 they are the whole policy.
    """
    allowed = set(rooms_left)
    for how_far_back, (_t, _p, room) in enumerate(where_it_has_been.get(object_id, ()), 1):
        if room in allowed:
            return room, (f"the room it was last seen in"
                          if how_far_back == 1 else
                          f"the {_in_words(how_far_back)} most recent room it has been "
                          f"seen in")
    klass = household.object_class.get(object_id, "")
    if klass:
        counts: collections.Counter = collections.Counter()
        for other, trail in where_it_has_been.items():
            if other == object_id or not trail:
                continue
            if household.object_class.get(other, "") == klass and trail[0][2] in allowed:
                counts[trail[0][2]] += 1
        if counts:
            # Ties broken by room name, so the choice does not depend on dictionary order.
            room = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            return room, (f"it has not been seen in any room left to open, so: the room where "
                          f"other {klass}s were last seen")
    return rng.choice(list(rooms_left)), (
        "at random: it has never been seen, and nothing of its class either")


def _next_from_rotation(rotation: List[str], cursor: List[int],
                        rooms_left: Sequence[str]) -> str:
    """The next room on the fair fixed rotation that has not been opened on this
    search. The cursor runs on across questions and days, so the rotation covers
    the house evenly and never depends on the question."""
    allowed = set(rooms_left)
    for _ in range(len(rotation)):
        room = rotation[cursor[0] % len(rotation)]
        cursor[0] += 1
        if room in allowed:
            return room
    return list(rooms_left)[0]


def answer_prompt(household: FrozenHousehold, question: dict, notes_text: str,
                  what_the_search_ruled_out: Sequence[str] = (),
                  the_record_to_read: Optional[str] = None,
                  what_it_saw_while_searching: Sequence[str] = (),
                  plainly: Optional[bool] = None) -> List[dict]:
    """Identical across arms apart from the notes and what this search ruled out.
    Deliberately the same shape as frozen_memory_test.question_prompt.

    THE ROOMS THE FAILED SEARCH JUST RULED OUT ARE IN THIS PROMPT. A robot that has
    opened three rooms and not found the towel knows the towel is not in those three
    rooms; making it answer as though it had not looked is artificial. It does not
    favour an arm, which is the only thing that would justify withholding it: all
    three arms rule out the SAME NUMBER of rooms, k = 3, they just rule out
    different ones - and that the memory-guided arm rules out better-chosen rooms is
    the hypothesis, not a confound. Withholding it would specifically suppress the
    advantage being measured.

    `ablation_the_notes_alone` runs the same thing with this list empty, to price
    what the immediate search evidence adds over the notes on their own.
    """
    if plainly is None:
        plainly = bool(the_record_to_read)
    lines = (
        (told.the_people_who_live_here(household.asked_objects, household.resident_ids)
         + [""]) if plainly else []
    ) + [
        f"It is day {question['day_index']} at {hours_and_minutes(question['t_query'])}.",
        "",
        "The rooms of this home and the spots in each:",
        *[f"- {room}: {', '.join(household.places_in_room[room])}"
          for room in household.rooms],
        "",
        "The robot has not been told where anything is. Everything it knows comes",
        "from looks it took.",
        "",
    ]
    if the_record_to_read:
        lines += [the_record_to_read, ""]
    lines += ["These are its notes:", "", notes_text, ""]
    if what_the_search_ruled_out:
        lines += [
            "You have just looked, and you did NOT find it in: "
            + ", ".join(what_the_search_ruled_out) + ".",
            "You see everything in a room you walk into, so it is not in any of "
            "those rooms. Do not name a spot in one of them.",
            "",
        ]
        # What it saw in those rooms, which the choice step was given and this step used
        # to throw away. A person in the kitchen, or the mug beside where the bottle
        # usually is, bears on the answer as much as the absence does.
        if what_it_saw_while_searching:
            lines += ["What you did see while looking just now:",
                      *what_it_saw_while_searching, ""]
    lines += [
        f"Question: where is {question['object_id']} right now?",
        "",
        "Name exactly one spot from the list above.",
        # This arm gets a plain scale. The shared CONF_SCALE ties 0.9 to "you have seen it
        # there repeatedly at this time of day", which is a different question from how
        # sure it is, and it lives in code other people's runs depend on.
        ("Also give a number from 0 to 1 for how sure you are that the spot is right."
         if plainly else CONF_SCALE),
        "",
        # Said to EVERY arm now, not only the record-reading one. It used to appear for that
        # arm alone, so one arm was verbally capped on a shared field and the others were not -
        # and it showed: that arm hit the old 240 on 8.8% of choices against 28.6% elsewhere.
        told.how_much_you_may_write(1200),
    ]
    return [{"role": "system",
             "content": (told.ANSWERING_SYSTEM if plainly
                         else ANSWER_SYSTEM)},
            {"role": "user", "content": "\n".join(lines)}]


# ------------------------------------------------------ the observation ledger --


def observation_accounting_for_a_day(looks_today: Sequence[Any], household: FrozenHousehold
                                     ) -> Dict[str, Any]:
    """How much the robot actually observed today. The confound, in numbers.

    A patrol gives one room-visit a day; search at three rooms a question gives up
    to twenty-four. This is logged per household per day so the ratio against
    results/self_improve/memory_factor_v1/ can be stated in one line rather than
    discovered late.
    """
    rooms: List[str] = []
    objects: Set[str] = set()
    residents: Set[str] = set()
    for look in looks_today:
        for target in look.targets:
            rooms.append(target["name"])
        for sighting in look.sightings:
            objects.add(sighting["object_id"])
        residents.update(look.residents_seen)
    asked = set(household.asked_objects)
    return {
        "room_visits": len(rooms),
        "distinct_rooms": len(set(rooms)),
        "n_rooms_in_the_house": len(household.rooms),
        "distinct_objects_observed": len(objects),
        "distinct_asked_about_objects_observed": len(objects & asked),
        "n_asked_about_objects": len(asked),
        "distinct_residents_seen": len(residents),
    }


AN_ABSENCE_LINE = " was not anywhere in the "


def the_day_in_words(looks_today: Sequence[Any], say_each_absence_once: bool = True) -> str:
    """Every search step described in full, by `describe_look_for_the_model(look,
    None)` - every sighting, every resident, every absence, no filtering by object.
    That last part is the fault being removed: the patrol caller filtered the
    description to the quizzed objects, so the notes mentioned 12 of 13 asked-about
    objects and zero of the other 54 the look stream held.

    `say_each_absence_once` is the ONE compaction, and it is lossless. Absence is
    emitted per known object per place looked at, so a day of 24 room-visits repeats
    the identical fact - "towel_nora was not anywhere in the kitchen" - up to
    twenty-four times. Measured on four households, 2026-09-24: rendering every look
    verbatim gives a nightly description of 124,000 to 161,000 characters (31,000 to
    40,000 tokens); stating each absence fact once the first time it is observed
    gives 45,000 to 72,000 characters (11,000 to 18,000 tokens), 0.36 to 0.45 of the
    size, with no fact removed - only verbatim restatements of a fact already in the
    same prompt. Every sighting is kept in full, with its own observation id, because
    a second sighting of the same object in the same room IS new information.

    A de-duplicated absence CARRIES ITS COUNT: "towel_nora was not anywhere in the
    kitchen (checked 3 times today)". Bare de-duplication would silently discard how
    thoroughly the robot looked, and thoroughness is evidence - one failed check of
    the kitchen and three failed checks are different grounds for concluding the
    towel has moved. So the absence is keyed on (object, room), stated once where it
    is first observed, and told how many times the day confirmed it. A fact is kept;
    only the repetition goes.

    False renders every look verbatim, for reproducing the uncompacted size.
    """
    if not looks_today:
        return "The robot did not look anywhere today."
    if not say_each_absence_once:
        return "\n\n".join(describe_look_for_the_model(look, None) for look in looks_today)

    # First pass: how many of today's looks confirmed each (object, room) absence.
    # Counted per look, not per absence record, because one look at a nine-place
    # kitchen emits nine absence records for one object and that is one check.
    how_many_times: Dict[Tuple[str, str], int] = collections.Counter()
    for look in looks_today:
        for pair in {(a["object_id"], a["room"]) for a in look.absences}:
            how_many_times[pair] += 1

    already_said: Set[str] = set()
    blocks: List[str] = []
    for look in looks_today:
        kept: List[str] = []
        for line in describe_look_for_the_model(look, None).splitlines():
            if AN_ABSENCE_LINE in line:
                fact = line.split("  [")[0].strip()
                if fact in already_said:
                    continue
                already_said.add(fact)
                object_id = fact.split(AN_ABSENCE_LINE)[0].strip().lstrip("- ").strip()
                room = fact.split(AN_ABSENCE_LINE)[-1].strip()
                times = how_many_times.get((object_id, room), 1)
                if times > 1:
                    line = f"{line} (checked {times} times today)"
            kept.append(line)
        while kept and kept[-1].strip() == "It looked and did NOT find:":
            kept.pop()
        blocks.append("\n".join(kept))
    return "\n\n".join(blocks)


# ------------------------------- how the day reaches the note-writing prompt --

# `write_the_notes` builds the day's description through the module-global
# `_what_happened_today`. We replace that global IN THIS PROCESS ONLY, rather than
# editing the shared module, because two other jobs are writing to
# src/self_improve/write_the_notes.py tonight and an edit there would collide. The
# replacement calls `the_day_in_words`, which is
# `describe_look_for_the_model(look, None)` for every look with duplicate absence
# facts said once.
#
# A monkey-patch that silently does not land would be exactly the failure this
# project keeps hitting - a run that looks complete because everything it did do
# succeeded - so the replacement counts its calls and `run_one_cell` asserts the
# count went up on every single night.
if not hasattr(write_the_notes_module, "_what_happened_today"):
    raise ImportError(
        "self_improve.write_the_notes no longer has _what_happened_today; the "
        "search-driven day renderer was wired to it and must be re-wired before "
        "any number from this module is believed")

_HOW_MANY_TIMES_THE_DAY_WAS_RENDERED = [0]
_THE_LAST_DAY_RENDERED = [""]


def _the_search_day_for_the_note_writer(looks_today, asked_objects,
                                        tell_the_model_everything_it_saw=False) -> str:
    """Stands in for write_the_notes._what_happened_today. `asked_objects` and
    `tell_the_model_everything_it_saw` are accepted and deliberately ignored: under
    search there is no filtering to the quizzed objects at all, for any arm."""
    _HOW_MANY_TIMES_THE_DAY_WAS_RENDERED[0] += 1
    _THE_LAST_DAY_RENDERED[0] = the_day_in_words(looks_today)
    return _THE_LAST_DAY_RENDERED[0]


def use_the_search_day_renderer() -> None:
    write_the_notes_module._what_happened_today = _the_search_day_for_the_note_writer


def the_sentence_for_tonight(how_memory_is_written: str, household: FrozenHousehold,
                             day: int) -> Optional[str]:
    """The one sentence a told arm hears tonight, or None on the other thirty nights.

    Returns None for every arm that is not a told arm, so nothing about any other arm's prompt
    changes by a single character - checked by running one household of the untold arm against
    the existing response cache and confirming every night was a hit.

    WHO is ill and WHEN comes from the household's own `day_causes`, and the function it uses
    raises rather than guessing, because a sentence delivered on the wrong night is an arm that
    looks compliant and is not. The person is named, because every look this arm has ever read
    says "Tomas": a sentence saying `resident_1 is unwell` would put the only inconsistency in
    the file into the one sentence we ever volunteer.
    """
    offset = WHEN_IT_IS_TOLD.get(how_memory_is_written)
    if offset is None:
        return None
    # EVERY spell, not the first. An episode with two illnesses gets told before each of them and
    # before each recovery, because an arm told only about the first would be a different arm
    # after day 32 and nothing would say so.
    from self_improve.three_prompts import the_unwell_spells
    names = names_by_resident_id(str(household.bank_path))
    for who, first_disrupted, first_back in the_unwell_spells(household):
        said = write_the_notes_module.the_message_for_tonight(
            day, who, first_disrupted + offset, first_back + offset, called=names[who])
        if said:
            return said
    return None


def extra_note_writing_settings(name_the_objects_it_will_be_quizzed_on: bool = True,
                                household: Optional[FrozenHousehold] = None
                                ) -> Dict[str, Any]:
    """Settings of `write_the_notes` that this arm PINS rather than inherits.

    Another job added `name_the_objects_it_will_be_quizzed_on` to write_the_notes and
    to study_settings while this wave was in flight (module mtime 13:51 against a
    13:40 launch, so the running processes were unaffected - Python had already
    imported the old code). It is passed EXPLICITLY here rather than inherited,
    because a later change to its default would otherwise silently alter what this
    arm's note-writer was told and the change would appear nowhere in this module.
    Passed only if the parameter exists, so this module does not break when someone
    else's signature moves again - and if it does NOT exist while False was asked
    for, that is raised rather than ignored, because silently running the primary
    configuration as the comparison one is precisely the failure this project keeps
    hitting.

    False is the PRIMARY configuration: the nightly prompt does not enumerate the
    objects the robot will be quizzed on, so the notes have to describe the house
    rather than cache answers to anticipated questions. Measured elsewhere on the
    patrol design: with the list, ZERO non-quizzed objects were named across 580
    nights; without it, 6.7, at unchanged note length - but coverage of the quizzed
    objects fell from 12.4 of 16.7 to 2.0. Under a fixed read budget that should cost
    note-derived answer accuracy. This arm is the one where it may not matter,
    because the robot goes and looks: a chooser needs the notes to point at a room,
    not to name a shelf.
    """
    import inspect
    extra: Dict[str, Any] = {}
    parameters = inspect.signature(write_the_notes).parameters
    # The nightly edit allowance is NOT pinned here. It depends on what tonight's looks
    # saw and on what the notes already cover, so it is computed per night at the call
    # site in run_one_cell. All this does is refuse to run against a write_the_notes that
    # could not accept it, because the alternative is silently keeping the
    # eight-edits-a-night limit that was measured to bind on four nights in five.
    if household is not None and "max_edits" not in parameters:
        raise RuntimeError(
            "this self_improve.write_the_notes has no max_edits parameter, so the "
            "claim store would silently keep the eight-edits-a-night limit. Refusing.")
    if "name_the_objects_it_will_be_quizzed_on" in parameters:
        extra["name_the_objects_it_will_be_quizzed_on"] = \
            name_the_objects_it_will_be_quizzed_on
    elif not name_the_objects_it_will_be_quizzed_on:
        raise RuntimeError(
            "the no-asked-list configuration was asked for, but this "
            "self_improve.write_the_notes has no name_the_objects_it_will_be_quizzed_on "
            "parameter, so the quiz list WOULD still be in the prompt. Refusing to run "
            "the primary configuration as the comparison one by accident.")
    return extra


# --------------------------------------------------------------- one cell --


def run_one_cell(household: FrozenHousehold, how_memory_is_written: str,
                 sensing_arm: str, client: LLMClient, out_dir: pathlib.Path,
                 last_day: int = 31, questions_per_day: int = 8,
                 budget: int = 3, seed: int = 0,
                 tell_it_what_it_ruled_out: bool = True,
                 name_the_objects_it_will_be_quizzed_on: bool = True) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pinned = extra_note_writing_settings(name_the_objects_it_will_be_quizzed_on,
                                        household=household)
    use_the_search_day_renderer()
    arm = f"{sensing_arm} with {how_memory_is_written}"
    # The people are recorded by name, not by resident_1. The mapping is exact, rebuilt
    # from the household's own seed and checked against the names on the objects, and it
    # raises rather than guessing: see who_lives_here.
    eyes = TheHouseAsSeen(household, out_dir / "looks.jsonl", "room",
                          names=names_by_resident_id(str(household.bank_path)))
    notes = Notes(out_dir / "notes.json", household.name, arm, how_memory_is_written)
    movers = the_movers(household)
    allowed_places = set(household.places)

    rotation = list(household.rooms)
    random.Random(f"{household.name}/rotation/{seed}").shuffle(rotation)
    rotation_cursor = [0]
    rng = random.Random(f"{household.name}/{sensing_arm}/{how_memory_is_written}/{seed}")
    seen_before: Dict[Tuple[str, str], int] = {}
    # Every room each object has been seen in, newest first, one entry per room. Created once
    # per household so it spans the whole month, and passed to every search.
    where_it_has_been: Dict[str, List[Tuple[int, str, str]]] = {}
    # The mechanical reference calls no model at all, so it writes no notes. Running the
    # nightly writer for it would spend GPU on an artifact nothing reads.
    it_writes_notes = sensing_arm not in THE_NO_MODEL_ARMS
    # The nightly arm reads `as_of_last_night` and records today's sightings into
    # `where_it_has_been`; the copy is refreshed after each day's questions.
    as_of_last_night: Dict[str, List[Tuple[int, str, str]]] = {}

    searches: List[Dict[str, Any]] = []
    per_day: List[Dict[str, Any]] = []
    nightly: List[Dict[str, Any]] = []
    started = time.time()
    # Written and flushed per question, so a cell is analysable while it is still
    # running. cell.json only appears when all 32 days are done, and the first wave
    # tonight was two thirds of the way through before any of its answers could be
    # read - search cost could be rebuilt from looks.jsonl but accuracy could not.
    searches_log = (out_dir / "searches.jsonl").open("w")
    searches_log.write(json.dumps({
        "kind": "header", "household": household.name, "sensing_arm": sensing_arm,
        "how_memory_is_written": how_memory_is_written,
        "the_answer_was_told_what_the_search_ruled_out": tell_it_what_it_ruled_out,
        "name_the_objects_it_will_be_quizzed_on": name_the_objects_it_will_be_quizzed_on,
        "budget_rooms_per_question": budget, "questions_per_day": questions_per_day,
        "movers": sorted(movers)}) + "\n")
    searches_log.flush()

    for day in range(last_day + 1):
        day_questions = questions_spread_across_the_day(
            answerable_questions_on_day(household, day), questions_per_day)
        looks_before = len(eyes.looks)
        day_rows: List[OneSearch] = []
        for question in day_questions:
            if sensing_arm == LAST_SEEN_NIGHTLY:
                row = run_one_search(eyes, household, notes, question, sensing_arm, client,
                                     budget, rotation, rotation_cursor, rng, movers,
                                     seen_before, allowed_places, tell_it_what_it_ruled_out,
                                     as_of_last_night, the_trail_to_record_into=where_it_has_been)
            else:
                row = run_one_search(eyes, household, notes, question, sensing_arm, client,
                                     budget, rotation, rotation_cursor, rng, movers,
                                     seen_before, allowed_places, tell_it_what_it_ruled_out,
                                     where_it_has_been)
            day_rows.append(row)
            searches.append(asdict(row))
            searches_log.write(json.dumps({"kind": "search", **asdict(row)}) + "\n")
            searches_log.flush()
        looks_today = eyes.looks[looks_before:]
        ledger = observation_accounting_for_a_day(looks_today, household)
        if sensing_arm == LAST_SEEN_NIGHTLY:
            # tonight's merge: the buffer of today's sightings joins the room list
            as_of_last_night = {o: list(t) for o, t in where_it_has_been.items()}

        if it_writes_notes:
            renders_before = _HOW_MANY_TIMES_THE_DAY_WAS_RENDERED[0]
            # One edit for each thing tonight saw that the notes do not mention yet,
            # plus room to revise what they do. Recomputed every night because that is
            # what it depends on: see write_the_notes.how_many_edits_a_night.
            edits_tonight = write_the_notes_module.how_many_edits_a_night(
                household, notes, looks_today)
            if how_memory_is_written == MEMGPT_AS_PUBLISHED:
                report = write_the_notes_memgpt_as_published(
                    notes, household, day, day * DAY_SECONDS + NIGHT, looks_today, client,
                    tell_the_model_everything_it_saw=True,
                    name_the_objects_it_will_be_quizzed_on=pinned.get(
                        "name_the_objects_it_will_be_quizzed_on", True),
                    a_message_tonight=pinned.get("a_message_tonight"))
            elif how_memory_is_written == A_WORKING_MEMORY_AND_AN_ARCHIVE:
                report = write_the_notes_memgpt(
                    notes, household, day, day * DAY_SECONDS + NIGHT, looks_today, client,
                    tell_the_model_everything_it_saw=True,
                    name_the_objects_it_will_be_quizzed_on=pinned.get(
                        "name_the_objects_it_will_be_quizzed_on", True),
                    a_message_tonight=pinned.get("a_message_tonight"))
            elif how_memory_is_written == THE_LOG_ONLY:
                # NO NIGHT AT ALL. This arm never writes, so there is nothing to call a model
                # about; it is in the log-reading family only so that its search-time prompt is
                # the control's, log and all, with an empty notes block.
                report = {"day": day, "model_call_failed": False, "n_model_calls_tonight": 0,
                          "applied": [], "rejected": [], "n_claims_now": 0,
                          "this_arm_never_writes": True}
            elif how_memory_is_written in THE_LOG_AND_THE_ROUTINE_FAMILY:
                # This arm's notes may not say where anything is, so its allowance is a flat
                # number rather than derived from the objects it saw: see EDITS_A_NIGHT in
                # that module. That flat 16 BINDS - on 27% of nights at 24 questions a day -
                # and the control's derived allowance does not, so the second name here runs
                # the same arm on the control's allowance and nothing else, to measure what
                # the ceiling cost. The two must never be pooled.
                report = write_the_notes_about_the_routine(
                    notes, household, day, day * DAY_SECONDS + NIGHT, looks_today,
                    client, tell_the_model_everything_it_saw=True,
                    name_the_objects_it_will_be_quizzed_on=pinned.get(
                        "name_the_objects_it_will_be_quizzed_on", True),
                    # PER NIGHT, not pinned for the cell. `pinned` is built once before the
                    # day loop, so a message put there would be repeated on all 32 nights and
                    # the arm would be "told every day", which is a different experiment.
                    a_message_tonight=(the_sentence_for_tonight(how_memory_is_written,
                                                               household, day)
                                       or pinned.get("a_message_tonight")),
                    # Only on a night a sentence arrives, so the other thirty nights are the
                    # parent arm's prompt exactly and replay from the cache.
                    **({"ask_it_what_will_change": True}
                       if (how_memory_is_written == TOLD_AND_ASKED_WHAT_CHANGES
                           and the_sentence_for_tonight(how_memory_is_written, household, day))
                       else {}),
                    **({"keeps_a_profile_of_each_person": True,
                        "who_lives_here": who_lives_here_in_plain_words(
                            str(household.bank_path))}
                       if how_memory_is_written in (A_PROFILE_OF_EACH_PERSON,
                                                   A_PROFILE_AND_TOLD) else {}),
                    **({"max_edits": (edits_tonight
                                      if NOTES_A_NIGHT[how_memory_is_written] is None
                                      else NOTES_A_NIGHT[how_memory_is_written])}
                       if how_memory_is_written in NOTES_A_NIGHT else {}))
            elif how_memory_is_written in (TOLD_IF_IT_WAS_RIGHT, ACE_AS_PUBLISHED):
                # The third arm is the only one that sees how the day went, so it is
                # the only one handed the day's answers. Everything else about the
                # night - the same day renderer, the same pinned prompt settings - is
                # the second arm's, which is what makes the pair comparable.
                report = write_the_notes_told_if_right(
                    notes, household, day, day * DAY_SECONDS + NIGHT, looks_today,
                    day_rows, client,
                    as_published=(how_memory_is_written == ACE_AS_PUBLISHED),
                    max_edits=edits_tonight,
                    tell_the_model_everything_it_saw=True,
                    name_the_objects_it_will_be_quizzed_on=pinned.get(
                        "name_the_objects_it_will_be_quizzed_on", True),
                    a_message_tonight=pinned.get("a_message_tonight"))
            else:
                report = write_the_notes(notes, household, day, day * DAY_SECONDS + NIGHT,
                                         looks_today, client, LOCKED.read_budget_lines,
                                         tell_the_model_everything_it_saw=True,
                                         max_edits=edits_tonight, **pinned)
            # WAS `!= renders_before + 1`, which killed a cell mid-run. What this check
            # exists to catch is a writer that builds the day text ITSELF and so shows the
            # model something other than what this module renders: that shows up as the
            # counter NOT moving. An arm that calls the model twice in a night renders
            # twice, through this same renderer, and is not the fault being guarded against.
            # `a small working memory and an archive` does exactly that when a write is
            # refused for want of room and it is given a second go, and
            # `a_small_working_memory_and_an_archive/hh_s19_t03` died at day 3 on its first
            # such night. The guard is unchanged in strength: only a render through
            # `the_day_this_arm_sees` moves this counter at all.
            rendered_tonight = _HOW_MANY_TIMES_THE_DAY_WAS_RENDERED[0] - renders_before
            # The log-only arm has no night, so it renders nothing and the guard does not apply.
            # It is exempted by name rather than by relaxing the test, which stays exactly as
            # strong for every arm that does write.
            if how_memory_is_written != THE_LOG_ONLY and rendered_tonight < 1:
                raise AssertionError(
                    f"{household.name}/{arm} day {day}: the note-writing prompt did NOT "
                    f"go through the search-driven day renderer, so what the model was "
                    f"shown is not what this module claims. Refusing to continue.")
        else:
            report = {"day": day, "no_notes_were_written": True,
                      "why": "this arm calls no model at all"}
        if it_writes_notes:
            report["n_times_the_day_was_rendered_for_the_writer"] = rendered_tonight
        report["period"] = period_of_day(day)
        report["n_characters_of_look_description"] = len(_THE_LAST_DAY_RENDERED[0]) \
            if it_writes_notes else 0
        report["n_characters_if_every_look_were_restated"] = len(
            the_day_in_words(looks_today, say_each_absence_once=False)) \
            if it_writes_notes else 0
        report["n_look_records_described"] = len(looks_today)
        nightly.append(report)
        if it_writes_notes:
            notes.save()

        scored_place = [r for r in day_rows if r.correct_place is not None]
        per_day.append({
            "household": household.name, "day": day, "period": period_of_day(day),
            "sensing_arm": sensing_arm, "how_memory_is_written": how_memory_is_written,
            "n_questions": len(day_rows),
            "n_scored": len(scored_place),
            "n_correct_place": sum(1 for r in scored_place if r.correct_place),
            "n_correct_room": sum(1 for r in scored_place if r.correct_room),
            "n_found": sum(1 for r in day_rows if r.found_it),
            "rooms_opened": sum(r.n_rooms_opened for r in day_rows),
            **ledger,
        })
        print(f"  {household.name} {sensing_arm[:14]:14s} {how_memory_is_written[:11]:11s} "
              f"day {day:2d} | {len(day_rows)} q | "
              f"found {sum(1 for r in day_rows if r.found_it)} | "
              f"rooms {sum(r.n_rooms_opened for r in day_rows)} | "
              f"place {sum(1 for r in scored_place if r.correct_place)}/{len(scored_place)} | "
              f"{time.time() - started:.0f}s", flush=True)

    eyes.close()
    searches_log.close()
    result = {
        "household": household.name,
        "arm": arm,
        "sensing_arm": sensing_arm,
        "how_memory_is_written": how_memory_is_written,
        "budget_rooms_per_question": budget,
        "questions_per_day": questions_per_day,
        "last_day": last_day,
        "seed": seed,
        "the_note_writer_was_shown_everything_it_saw": True,
        "there_was_no_patrol_and_no_warm_start": True,
        "the_answer_was_told_what_the_search_ruled_out": tell_it_what_it_ruled_out,
        "name_the_objects_it_will_be_quizzed_on": name_the_objects_it_will_be_quizzed_on,
        "the_note_writing_settings_this_arm_pinned": pinned,
        "how_this_differs_from_the_patrol_runs": (
            "The search runs differ from the patrol runs at "
            "results/self_improve/memory_factor_v1/ in TWO ways at once - no priors "
            "and search-driven looking - so no one may compare the two sets naively. "
            "The day-0 whole-house walkthrough is OFF here, so the settled fortnight "
            "is learned from scratch rather than given; and every observation comes "
            "from a search the robot made to answer a question, at up to 24 "
            "room-visits a day against the patrol's 1.26."),
        "movers": sorted(movers),
        "n_asked_objects": len(household.asked_objects),
        "rotation_in_the_order_it_is_walked": rotation,
        "searches": searches,
        "per_day": per_day,
        "nightly": nightly,
        "n_looks": len(eyes.looks),
        "n_sightings": len(eyes.sightings_so_far()),
        "n_absences_recorded": len(eyes.absences_so_far()),
        "n_distinct_objects_ever_observed": len({s.object_id for s in eyes.sightings_so_far()}),
        "n_distinct_asked_about_objects_ever_observed": len(
            {s.object_id for s in eyes.sightings_so_far()} & set(household.asked_objects)),
    }
    result["summary"] = summarise_a_cell(result)
    (out_dir / "cell.json").write_text(json.dumps(result, indent=1))
    # the tidy per-day rows, as their own file, so nothing has to parse cell.json
    with (out_dir / "per_day.jsonl").open("w") as fh:
        for row in per_day:
            fh.write(json.dumps(row) + "\n")
    return result


def summarise_a_cell(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = result["searches"]
    scored = [r for r in rows if r["correct_place"] is not None]
    n = len(scored)
    movers = [r for r in scored if r["is_a_mover"]]
    non_movers = [r for r in scored if not r["is_a_mover"]]
    errors = [r for r in scored if r["correct_place"] is False]
    split = collections.Counter(r["error_kind"] for r in errors)
    by_period: Dict[str, Dict[str, Any]] = {}
    for period in ("settled", "disrupted", "back to normal"):
        here = [r for r in scored if r["period"] == period]
        if here:
            by_period[period] = {
                "n": len(here),
                "share_correct_place": sum(1 for r in here if r["correct_place"]) / len(here),
                "share_correct_room": sum(1 for r in here if r["correct_room"]) / len(here),
                "share_found_within_budget": sum(1 for r in here if r["found_it"]) / len(here),
                "mean_rooms_opened": statistics.mean(r["n_rooms_opened"] for r in here),
            }
    ledger = result["per_day"]
    return {
        "n_questions_asked": len(rows),
        "n_questions_scored": n,
        "n_unparsed": len(rows) - n,
        "share_correct_place": (sum(1 for r in scored if r["correct_place"]) / n) if n else None,
        "share_correct_room": (sum(1 for r in scored if r["correct_room"]) / n) if n else None,
        "share_found_within_budget": (sum(1 for r in rows if r["found_it"]) / len(rows))
                                     if rows else None,
        "mean_rooms_opened_per_question": statistics.mean(r["n_rooms_opened"] for r in rows)
                                          if rows else None,
        "share_correct_place_movers": (sum(1 for r in movers if r["correct_place"]) / len(movers))
                                      if movers else None,
        "share_correct_place_non_movers": (sum(1 for r in non_movers if r["correct_place"])
                                           / len(non_movers)) if non_movers else None,
        "n_choice_calls_that_failed": sum(r["n_choice_calls_that_failed"] for r in rows),
        "share_of_answers_naming_a_room_the_search_had_ruled_out": (
            sum(1 for r in rows if r["answered_from"] == "the notes"
                and r["the_answer_named_a_room_the_search_had_ruled_out"])
            / max(1, sum(1 for r in rows if r["answered_from"] == "the notes"))),
        "the_three_way_error_split": {
            "n_errors": len(errors),
            NEVER_SAW_IT: split.get(NEVER_SAW_IT, 0),
            SAW_IT_ONCE: split.get(SAW_IT_ONCE, 0),
            SAW_IT_OFTEN: split.get(SAW_IT_OFTEN, 0),
            "share_never_saw_it": (split.get(NEVER_SAW_IT, 0) / len(errors)) if errors else None,
        },
        "observation": {
            "total_room_visits": sum(r["room_visits"] for r in ledger),
            "mean_room_visits_per_day": statistics.mean(r["room_visits"] for r in ledger)
                                        if ledger else None,
            "mean_distinct_rooms_per_day": statistics.mean(r["distinct_rooms"] for r in ledger)
                                           if ledger else None,
            "mean_distinct_objects_per_day":
                statistics.mean(r["distinct_objects_observed"] for r in ledger) if ledger else None,
            "mean_distinct_asked_about_objects_per_day":
                statistics.mean(r["distinct_asked_about_objects_observed"] for r in ledger)
                if ledger else None,
            "n_distinct_objects_ever_observed": result.get("n_distinct_objects_ever_observed"),
            "n_distinct_asked_about_objects_ever_observed":
                result.get("n_distinct_asked_about_objects_ever_observed"),
        },
    }


def sanity_assay_on_searches(result: Dict[str, Any]) -> Dict[str, Any]:
    """The first-minute assay, on the answers this cell produced.

    Three ways an arm is broken that accuracy will not show: the answers pile onto
    one spot regardless of the question, consecutive answers are identical, or the
    model would not answer at all. Run this before trusting any accuracy figure.

    Answers the search FOUND are excluded: they are not the model's answers, they
    are the world's, and including them would mask a degenerate note-reader.
    """
    from_notes = [r for r in result["searches"] if r["answered_from"] == "the notes"]
    places = [r["answer_place"] for r in from_notes if r["answer_place"]]
    counts: Dict[str, int] = {}
    for p in places:
        counts[p] = counts.get(p, 0) + 1
    commonest = max(counts.values()) / len(places) if places else None
    repeats = sum(1 for i in range(1, len(places)) if places[i] == places[i - 1])
    report = {
        "n_answers_from_the_notes": len(from_notes),
        "n_distinct_spots_named": len(counts),
        "share_on_the_commonest_spot": commonest,
        "share_identical_to_the_previous_answer":
            repeats / (len(places) - 1) if len(places) > 1 else None,
        "share_unparsed": (sum(1 for r in from_notes if r["answer_place"] is None)
                           / len(from_notes)) if from_notes else None,
        "concerns": [],
    }
    if commonest is not None and commonest > 0.6:
        report["concerns"].append(
            f"{commonest:.0%} of answers name the same spot: the answer distribution "
            f"is degenerate")
    if len(counts) <= 2 and len(places) > 10:
        report["concerns"].append(
            f"only {len(counts)} distinct spots ever named across {len(places)} answers")
    if report["share_identical_to_the_previous_answer"] is not None and \
            report["share_identical_to_the_previous_answer"] > 0.8:
        report["concerns"].append("consecutive answers are nearly always identical")
    if report["share_unparsed"] and report["share_unparsed"] > 0.1:
        report["concerns"].append(f"{report['share_unparsed']:.0%} of answers did not parse")
    if result["summary"]["n_choice_calls_that_failed"] > 0.05 * max(
            1, result["summary"]["n_questions_asked"]):
        report["concerns"].append(
            f"{result['summary']['n_choice_calls_that_failed']} room choices fell back "
            f"to the rotation because the model did not answer")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household", default="hh_s0_t03")
    parser.add_argument("--how", default="incremental edits", choices=HOW_MEMORY_IS_WRITTEN)
    parser.add_argument("--sensing", default=MEMORY_GUIDED, choices=list(SENSING_ARMS))
    parser.add_argument("--banks", type=pathlib.Path, default=FROZEN_BANKS)
    parser.add_argument("--last-day", type=int, default=31)
    parser.add_argument("--questions-per-day", type=int, default=8)
    parser.add_argument("--budget", type=int, default=LOCKED.search_looks_when_answering)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--the-notes-alone", action="store_true",
                        help="ABLATION: do not tell the answer step which rooms the "
                             "failed search just ruled out. Prices what the immediate "
                             "search evidence adds over the notes on their own. The "
                             "primary condition tells it.")
    parser.add_argument("--no-asked-list", action="store_true",
                        help="PRIMARY: do not enumerate the objects the robot will be "
                             "quizzed on in the nightly note-writing prompt, so the "
                             "notes must describe the house rather than cache answers "
                             "to anticipated questions. Refuses to run if this "
                             "write_the_notes has no such parameter.")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("results/self_improve/search_driven"))
    parser.add_argument("--cache", type=pathlib.Path,
                        default=pathlib.Path("llm_prior_cache/self_improve"))
    args = parser.parse_args(argv)

    household = FrozenHousehold(args.banks / f"{args.household}.jsonl")
    client = LLMClient(args.cache)
    out_dir = (args.out / household.name / args.sensing.replace(" ", "_")
               / args.how.replace(" ", "_"))
    result = run_one_cell(household, args.how, args.sensing, client, out_dir,
                          args.last_day, args.questions_per_day, args.budget, args.seed,
                          tell_it_what_it_ruled_out=not args.the_notes_alone,
                          name_the_objects_it_will_be_quizzed_on=not args.no_asked_list)
    assay = sanity_assay_on_searches(result)
    (out_dir / "sanity_assay.json").write_text(json.dumps(assay, indent=1))
    print()
    print(json.dumps(result["summary"], indent=1))
    print()
    print("sanity assay:", json.dumps(assay, indent=1))
    print(f"model calls {client.stats['calls']}, cache hits {client.stats['cached']}, "
          f"{client.stats['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

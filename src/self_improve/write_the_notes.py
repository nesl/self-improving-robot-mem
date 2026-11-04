"""How the robot writes its notes each night. This is factor one of the study.

Two ways, and the prompts are identical apart from the way itself:

  wholesale rewrite   the model is shown today's looks and last night's summary,
                      and writes a new summary of the household from scratch.
  incremental edits   the model is shown today's looks and its current claims,
                      and emits individual edits: add a claim, revise one, or
                      attach today's evidence to one.

Both are shown today's looks through the same function,
`looking.describe_look_for_the_model`, so both receive sightings and absences in
exactly the same words. If only one format could represent "looked and it was not
there", the two factors would be confounded and no measured difference would be
attributable to either.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from baselines.patrol.llm import LLMClient
from self_improve import what_the_robot_is_told as told
from self_improve.frozen_household import FrozenHousehold, period_of_day
from self_improve.frozen_household import plain_place_name
from self_improve.looking import LookRecord, describe_look_for_the_model
from self_improve.memory_notes import (ESTABLISHED, PROVISIONAL, SET_ASIDE,
                                      STILL_STANDING, Notes, summary_lines)

def bare_place_words(place_id: str) -> str:
    """'bathroom_shelf_ba1' -> 'bathroom shelf'. No article.

    `plain_place_name` prefixes "the ", which is right for prose the model reads and
    wrong for matching prose the model wrote: the summaries say "Kitchen cupboard
    holds glass_nora", never "the cupboard". Matching on the article-prefixed form
    recovered pairs on only 8 of 29 nights, which is why this exists.
    """
    parts = place_id.split("_")
    if len(parts) > 1 and any(ch.isdigit() for ch in parts[-1]):
        parts = parts[:-1]
    return " ".join(parts).replace("_", " ").lower()


def ways_an_object_may_be_written(object_id: str) -> List[str]:
    """The forms a summary might use for `glass_aisha`.

    Object ids are `<class>_<person>`, but the rewrite arm writes prose and names
    things as people's possessions: "Aisha glass", "Felix's book", "Hana, Ines and Yuki
    mugs". Matching only the identifier recovered **nothing at all** from three of the
    ten households' summaries - and those three then scored as 100% vacuous, which is a
    pure measurement artefact and not a property of the memory.

    A class word alone is deliberately NOT a match: "the mug" cannot be attributed to a
    person, and crediting it to one would invent a fact.
    """
    forms = [object_id.lower()]
    parts = object_id.split("_")
    if len(parts) >= 2:
        person = parts[-1].lower()
        thing = " ".join(parts[:-1]).lower()
        forms += [f"{person} {thing}", f"{person}'s {thing}", f"{person}s {thing}",
                  f"{thing} {person}", f"{thing} of {person}"]
    return forms


def where_an_object_is_named(text: str, object_id: str) -> int:
    """The first position this object is named at, in any of its forms, or -1.

    IT NORMALISES THE TEXT ITSELF, and it did not. The forms are lowercase and the match
    was case-sensitive against the raw text, so "Marco's water bottle is in the dining
    room" scored as not naming `water_bottle_marco` - and so did every prose mention that
    began a sentence or used a capitalised name. It only ever appeared to work because the
    other caller, `facts_a_statement_asserts`, lowercases before calling, and because until
    the people had real names the model wrote identifiers like `book_dana` in lowercase.
    The night the notes started saying "Marco's water bottle", a hand count over one
    household went from six movers named to zero, and I believed the zero.

    Found by two measurements of the same thing disagreeing - 0 of 6 against 6 of 6 on the
    same household and night - which is the only reason it surfaced at all. Normalising
    inside the function means no caller can hold it wrong.
    """
    hay = _normalise(text)
    for form in ways_an_object_may_be_written(object_id):
        at = _find_whole_words(hay, _normalise(form))
        if at >= 0:
            return at
    return -1


def _find_whole_words(text: str, needle: str) -> int:
    """Where `needle` appears as whole words, or -1.

    Plain substring matching credited "Aisha glasses" to BOTH glasses_aisha and
    glass_aisha, because "aisha glass" sits inside "aisha glasses". Found by hand-
    checking ten nights per arm, and it only ever invents facts.
    """
    if not needle:
        return -1
    for match in re.finditer(re.escape(needle), text):
        before = match.start() - 1
        after = match.end()
        if (before < 0 or not (text[before].isalnum() or text[before] == "_")) and \
           (after >= len(text) or not (text[after].isalnum() or text[after] == "_")):
            return match.start()
    return -1


def _normalise(text: str) -> str:
    """Underscores to spaces, so a place written KITCHEN_TABLE or BATHROOM_TOWEL_RACK
    matches the same place written "kitchen table".

    The rewrite arm writes place names in both styles and the claim store writes only
    one, so failing on the underscore style was a fault that penalised one arm alone -
    it silently lost every fact on some nights. Also found by hand-check.
    """
    return (text or "").lower().replace("_", " ")


def facts_a_statement_asserts(statement: str, object_ids: Sequence[str],
                              place_ids: Sequence[str],
                              place_room: Optional[Dict[str, str]] = None) -> set:
    """The (thing, place) pairs one line names.

    A place counts as named when the line carries its identifier, or its bare words
    - and where those bare words are ambiguous across rooms ("desk" belongs to a
    bedroom and an office) the line must also name the right room, so "Bedroom_1 desk
    holds charger_yuki" is not credited to the office desk.

    Used only to tell a real edit from a re-wording. It never reaches a prompt.
    """
    text = (statement or "").lower()
    named_at = {o: where_an_object_is_named(text, o) for o in object_ids}
    things = [o for o, at in named_at.items() if at >= 0]
    if not things:
        return set()

    how_many_places_share_these_words: Dict[str, int] = {}
    for place in place_ids:
        words = bare_place_words(place)
        how_many_places_share_these_words[words] = \
            how_many_places_share_these_words.get(words, 0) + 1

    # Where each place is mentioned, so an object can be paired with the nearest one.
    # Pairing every object with every place would emit a cross-product: a line like
    # "bedroom nightstands hold books for residents 1-3" would assert nine pairs, most
    # of them false, and a slight rewording would shuffle them and look like new
    # information. Nearest-mention pairing keeps one pair per object.
    flat = _normalise(text)
    where: List[Tuple[int, str]] = []
    for place in place_ids:
        at = _find_whole_words(text, place.lower())
        if at < 0:
            words = bare_place_words(place)
            if not words:
                continue
            ambiguous = how_many_places_share_these_words[words] > 1
            if ambiguous:
                room = (place_room or {}).get(place, "")
                if not room or _find_whole_words(flat, room.lower().replace("_", " ")) < 0:
                    continue
            # searched in the underscore-flattened text so KITCHEN_TABLE matches too
            at = _find_whole_words(flat, words)
        if at >= 0:
            where.append((at, place))
    if not where:
        return set()

    pairs = set()
    for thing in things:
        at = named_at[thing]
        nearest = min(where, key=lambda spot: abs(spot[0] - at))
        pairs.add((thing, nearest[1]))
    return pairs


def were_the_edits_vacuous(edits: Sequence[dict], claims_before: Sequence[str],
                           household: FrozenHousehold) -> Dict[str, Any]:
    """Did a required edit carry information the notes did not already have?

    The forced-edit rule (see write_the_notes_incrementally) guarantees the arm
    says SOMETHING on a night it saw something. It does not guarantee the something
    is new. An edit that re-asserts an existing claim in different words satisfies
    the schema and teaches the notes nothing, and it would look identical to real
    revision in any accuracy number - so we count it.
    """
    already = set()
    for statement in claims_before:
        already |= facts_a_statement_asserts(statement, household.asked_objects,
                                             household.places, household.place_room)
    verdicts = []
    for edit in edits:
        asserted = facts_a_statement_asserts(edit.get("statement") or "",
                                             household.asked_objects, household.places,
                                             household.place_room)
        new_pairs = asserted - already
        verdicts.append({
            "action": edit.get("action"),
            "named_a_thing_and_a_place": bool(asserted),
            "carried_something_new": bool(new_pairs)
                                     or edit.get("action") == "record evidence"
                                     or bool(edit.get("contradicting_observation_ids")),
            "new_pairs": sorted(f"{t} on {p}" for t, p in new_pairs),
        })
        already |= asserted
    n = len(verdicts)
    return {"n_edits": n,
            "n_that_carried_something_new": sum(1 for v in verdicts if v["carried_something_new"]),
            "share_that_carried_something_new":
                (sum(1 for v in verdicts if v["carried_something_new"]) / n) if n else None,
            "n_that_named_no_thing_and_place":
                sum(1 for v in verdicts if not v["named_a_thing_and_a_place"]),
            "verdicts": verdicts}


SYSTEM = ("You help a home robot keep notes about where a household's things are "
          "kept. Nobody tells the robot the right answer; everything it knows "
          "comes from looking. Answer with JSON only.")

EDITS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array", "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    # SETTING A NOTE ASIDE IS ITS OWN ACTION. It used to be a field on a
                    # revision, and the model filled it in whenever it touched a note: on
                    # 2026-09-25, 23 of 24 set-asides in one cell happened in the same edit
                    # as a change of wording, with stated reasons like "no new evidence to
                    # change status" - a reason for KEEPING a note. 85% of notes then read
                    # "set aside for now" whatever the routine, which destroys the one
                    # measure the study reads off that field. Same shape as the display
                    # teaching the format: a field offered is a field filled.
                    # WHY FIRST. It used to come last, so the model chose an action, named
                    # the note and wrote the new wording before saying why - a decision made
                    # with no tokens spent on it, then justified. Same fault as the room
                    # chooser, found the same way. 600 characters, not 240: this field is
                    # reasoning, not memory, and the 240 was the per-note cap borrowed.
                    "why": {"type": "string", "maxLength": 600},
                    "action": {"type": "string",
                               "enum": ["add", "revise", "record evidence",
                                        "set a note aside", "bring a note back"]},
                    "claim_id": {"type": ["string", "null"]},
                    "statement": {"type": ["string", "null"], "maxLength": 240},
                    "holds_under": {"type": ["string", "null"], "maxLength": 120},
                    "status": {"type": ["string", "null"],
                               "enum": [PROVISIONAL, ESTABLISHED, None]},
                    "supporting_observation_ids": {"type": "array", "items": {"type": "string"},
                                                   "maxItems": 6},
                    "contradicting_observation_ids": {"type": "array", "items": {"type": "string"},
                                                      "maxItems": 6},
                },
                "required": ["why", "action", "claim_id", "statement", "holds_under", "status",
                             "supporting_observation_ids",
                             "contradicting_observation_ids"],
                "additionalProperties": False},
        },
    },
    "required": ["edits"], "additionalProperties": False}

def edits_schema(max_edits: int) -> Dict[str, Any]:
    """EDITS_SCHEMA with the number of edits a night moved. A fresh dict every time,
    because the module constant is imported elsewhere and mutating it would move other
    people's numbers.

    WHY THIS EXISTS. The constant allows eight edits a night. Measured on 2026-09-24
    across all five running claim-store arms, about four nights in five made exactly
    eight dated changes, and on the first night of a fifteen-object household every arm
    wrote exactly eight claims and then needed three or four more nights to reach one
    claim per object. So eight was not a ceiling the model chose, it was deciding what
    the memory was allowed to record.

    It is also not even-handed between the prompt variants. KEEP_RIVAL_BELIEFS asks for
    TWO edits per moved object - add the new claim, set the old one aside - where the
    control spends one, so under a shared cap that arm covers half as many objects on
    the night a routine changes. That is the night the study is about.

    A survey of eight published memory systems (see results/self_improve/
    HOW_OTHER_MEMORIES_BOUND_THEIR_GROWTH.md) found exactly one that deliberately
    limits how much may be written in a step, and it ties the limit to an actual
    overflow rather than to a constant. So the right number here is one that cannot
    bind: two per object the robot is asked about, which is what the most expensive
    variant needs to cover every object in one night. Whether it bound anyway is
    recorded per night as `the_edits_cap_bit`, so a limit that never binds costs
    nothing and can be shown to have never bound."""
    schema = json.loads(json.dumps(EDITS_SCHEMA))
    schema["properties"]["edits"]["maxItems"] = max_edits
    return schema


# Spare edits on top of the two counts below, so a night is never one edit short of
# what it wanted. Four, and it is the only unmeasured number left in this function.
SPARE_EDITS_A_NIGHT = 4


def how_many_edits_a_night(household: FrozenHousehold,
                           notes: Optional["Notes"] = None,
                           looks_today: Sequence[LookRecord] = (),
                           two_edits_per_change: bool = False) -> int:
    """How many edits tonight may make: one for each thing it saw that its notes do not
    mention, one for each thing it saw somewhere other than where its notes say, and
    four spare.

    BOTH COUNTS ARE MEASURED FROM TONIGHT'S LOOKS, and each replaced a borrowed eight.

    The first version of this was two per object the robot is asked about. It bound on
    night 1, because with the quiz list gone the writer describes the house: one household
    saw 47 distinct objects against an allowance of 22.

    The second version fixed that and gave every night eight slots to revise with. It
    bound on nights 4 and 5 of the first six-day run - 9 of 9 and 10 of 10 edits offered.
    Measured on that run, the number of things a night saw somewhere other than where its
    notes said was 12, 6, 13 and 5. Eight was below half of those nights' needs. So the
    revision count is now also counted rather than assumed, with the audited extractor
    `facts_a_statement_asserts`, which is the same one every compliance number uses.

    `two_edits_per_change` doubles the revision part for the prompts that ask for a new
    claim AND a set-aside for one change, so every arm can represent the same number of
    CHANGES out of the same purse. Giving them the same number of edits instead is what
    made the flat cap uneven between arms.

    Passing no notes and no looks falls back to the old count, so a caller that has not
    been updated keeps a working allowance rather than a broken one.
    """
    if notes is None or not looks_today:
        return max(8, 2 * len(household.asked_objects))
    seen_at: Dict[str, str] = {}
    for look in looks_today:
        for sighting in look.sightings:
            seen_at[sighting["object_id"]] = sighting["place_id"]
    live = [c for c in notes.claims if getattr(c, "folded_into", None) is None]
    places = list(household.places)
    uncovered = disagreeing = 0
    for thing, place in seen_at.items():
        about_it = [c for c in live
                    if where_an_object_is_named(c.statement, thing) >= 0]
        if not about_it:
            uncovered += 1
            continue
        said = set()
        for claim in about_it:
            said |= {p for _, p in facts_a_statement_asserts(
                claim.statement, [thing], places, household.place_room)}
        if said and place not in said:
            disagreeing += 1
    return max(8, uncovered
               + disagreeing * (2 if two_edits_per_change else 1)
               + SPARE_EDITS_A_NIGHT)


AS_IT_STANDS = "as it stands"

# ------------------------------------------------ the three note-writing prompts
# added 2026-09-24 for results/self_improve/three_prompts/. EVERY ONE IS OFF BY
# DEFAULT and nothing above or below changes, so every number measured before today
# reproduces byte-for-byte - verifiable by running a household against the existing
# completion cache and seeing every night hit.
#
# Each is a pure ADDITION of lines to a prompt. None removes or rewords an existing
# line, because a delta that also rewords cannot be attributed.

KEEP_RIVAL_BELIEFS: List[str] = [
    # The mechanism already exists in memory_notes: add_claim plus a revise that sets
    # `standing` to SET_ASIDE keeps both wordings readable. The prompt mentions
    # SET_ASIDE but frames it as something to do TO a claim the evidence went against,
    # and revising a claim's statement is offered first - so in 81 of 118 traced
    # losses the model OVERWROTE the old claim instead of keeping both. This says
    # plainly which of the two to do, and why keeping the loser is worth the space.
    "",
    "When something appears to have MOVED, do not rewrite the claim that said where "
    "it used to be. Instead do two things in the same set of edits: ADD a new claim "
    "for where it is now, and revise the OLD claim only to set its standing to "
    f'"{SET_ASIDE}", leaving its wording exactly as it is. Both claims then stay in '
    "your notes and both stay readable.",
    "",
    "Keeping the belief you have set aside is worth the space. Routines come back. "
    "When this household returns to how it used to be, the claim you set aside is "
    "the right answer again, and you will not have to see it happen a second time "
    "to know it.",
]

DESCRIBE_THE_PERSON: List[str] = [
    # Nothing in this project has ever asked the robot to describe the people. The
    # prompt asks for object places and nudges toward saying WHEN a place holds
    # something, but never for a routine, a habit or a regime - while the study's
    # whole question is whether it notices a person's routine change. The line budget
    # is deliberately NOT raised: a routine line has to displace an object line, and
    # that trade is part of what is being measured.
    "",
    "Write about the people in this home as well as about where things are. Record "
    "what you can work out about their routine: when each of them is at home and "
    "when they are out, what they seem to be doing and in which rooms, at what times "
    "of day - and whether anything about that pattern has changed lately.",
    "",
    "Your notes get no more lines for this. A line spent on the household's routine "
    "is a line not spent on one object's spot, so decide which is worth more.",
]


def the_message_for_tonight(day: int, ill_resident: str = "resident_1",
                            first_disrupted_day: int = 14,
                            first_day_back: int = 24, *,
                            called: Optional[str] = None) -> Optional[str]:
    # The day numbers are checked because passing a name where a day belongs returns None on
    # every night, which turns the told-it arm into a silent copy of the control with
    # nothing anywhere recording that its one sentence went missing. That nearly happened
    # tonight when a parameter was inserted mid-signature. A wrong type must raise.
    for name, value in (("first_disrupted_day", first_disrupted_day),
                        ("first_day_back", first_day_back), ("day", day)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be a whole number of days, got {value!r}. If you meant to "
                f"pass the person's name, it is the keyword-only argument `called`.")
    """The ONE sentence the told-it arm adds, on exactly two nights and no others.

    Nothing in any arm of this study has ever told the robot that anything changed.
    This is the cheapest possible version of telling it: one sentence, on the first
    disrupted night and the first night back, and silence on the other thirty.

    `called` is the person's real name, and it is what the sentence says. It must be
    given now that looks record people by name: a message saying "resident_1 is unwell"
    while every look says "Tomas" is an inconsistency the robot cannot resolve, and it
    would be the told-it arm's own message that confused it. `ill_resident` stays the id,
    because that is what the bank's day_causes are keyed on. It is derived from the bank's day_causes
    (`unwell_spell:resident_N`), never hard-coded per home.

    Returns None on every other day, so the caller can log exactly which nights
    carried a message and assert the set is {first_disrupted_day, first_day_back}.
    """
    if day == first_disrupted_day:
        return (f"Something you have been told, which you did not see for yourself: "
                f"{called or ill_resident} is unwell and is staying at home instead of "
                f"going out.")
    if day == first_day_back:
        return (f"Something you have been told, which you did not see for yourself: "
                f"{called or ill_resident} is better and is back to their usual "
                f"routine.")
    return None


REWRITE_WORDINGS: Dict[str, List[str]] = {
    # ------------------------------------------------------------------ the default
    # EXACTLY the lines that were inline in rewrite_the_notes_wholesale before
    # 2026-09-24, character for character. Every number measured with the wholesale
    # arm up to that date was measured with this text, so it must never be edited:
    # add a key instead. Verified by running one household against the pre-existing
    # completion cache and confirming every night was a cache HIT, which can only
    # happen if the prompt is byte-identical.
    AS_IT_STANDS: [
        "Write the notes again from scratch. Produce one summary of where this",
        "household keeps the things above, as you now believe it to be. It",
        "replaces what is there: whatever you do not write down is gone.",
        "Say which routine or condition each part of it holds under. A household",
        "can be in more than one routine over a month.",
    ],
    # ------------------------------------------------- one-change deltas of the above
    # Why these exist: the wholesale arm reproduces its previous night's summary
    # byte-for-byte on about a quarter of the nights that saw something worth
    # recording. If that is because "Write the notes AGAIN" reads to the model as
    # "reproduce them", the finding is about our phrasing and not about wholesale
    # summarising. There are two candidate cues in the default text and they are
    # different hypotheses, so each variant changes ONE thing:
    #
    #   "written afresh"                the sentence carrying "again" is replaced by
    #                                   one with no word that can imply reproduction.
    #                                   The loss warning is kept.
    #   "the loss warning removed"      "whatever you do not write down is gone" is
    #                                   dropped and nothing else changes. Under a read
    #                                   budget that warning rewards carrying the whole
    #                                   of last night's text over, which produces a
    #                                   byte-identical summary without the model ever
    #                                   reading "again" as "copy". NOTE: this withholds
    #                                   a fact that is TRUE of the method - omitted
    #                                   text really is lost - so it is an ablation of a
    #                                   prompt cue, not a better prompt.
    #   "unchanged need not be repeated"  the default plus an explicit licence to leave
    #                                   out what has not changed. Should drive
    #                                   byte-identity to zero; the question is what it
    #                                   does to the semantic measure and to accuracy.
    #   "restating is invited"          the opposite pole, the positive control. If
    #                                   repetition does not rise here, the measure is
    #                                   not responding to wording at all and no null
    #                                   from the other arms means anything.
    "written afresh": [
        "Write down where this household keeps the things above, as you now believe",
        "it to be. One summary, put together from what you know tonight. It",
        "replaces what is there: whatever you do not write down is gone.",
        "Say which routine or condition each part of it holds under. A household",
        "can be in more than one routine over a month.",
    ],
    "the loss warning removed": [
        "Write the notes again from scratch. Produce one summary of where this",
        "household keeps the things above, as you now believe it to be.",
        "Say which routine or condition each part of it holds under. A household",
        "can be in more than one routine over a month.",
    ],
    "unchanged need not be repeated": [
        "Write the notes again from scratch. Produce one summary of where this",
        "household keeps the things above, as you now believe it to be. It",
        "replaces what is there: whatever you do not write down is gone.",
        "Anything that has not changed since last night does not need writing out",
        "again in the same words; spend the lines on what is new or now different.",
        "Say which routine or condition each part of it holds under. A household",
        "can be in more than one routine over a month.",
    ],
    "restating is invited": [
        "Write the notes again from scratch. Produce one summary of where this",
        "household keeps the things above, as you now believe it to be. It",
        "replaces what is there: whatever you do not write down is gone.",
        "Where nothing has changed since last night, writing last night's notes out",
        "again word for word is exactly right.",
        "Say which routine or condition each part of it holds under. A household",
        "can be in more than one routine over a month.",
    ],
}


SUMMARY_MAX_CHARACTERS = 2400
"""The wholesale summary's hard ceiling, in the SCHEMA rather than in the prompt.

FOUND 2026-09-24 while auditing what the two formats were actually told. This number is
a cap on the wholesale arm's ENTIRE MEMORY, for the whole month, and the claim store has
no equivalent: the claim store accumulates, so with up to 8 edits a night at 240
characters a statement it can reach tens of thousands of characters over 32 nights while
the summary can never exceed 2400. Removing the budget SENTENCE from the prompt does not
remove this, so a "no length limit" wholesale arm that still carries this schema is not
one. It is raised by `summary_max_characters`; the default is this number so no earlier
number moves."""

SUMMARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string", "maxLength": SUMMARY_MAX_CHARACTERS}},
    "required": ["summary"], "additionalProperties": False}


def summary_schema(max_characters: int = SUMMARY_MAX_CHARACTERS) -> Dict[str, Any]:
    """SUMMARY_SCHEMA with the ceiling moved. A fresh dict every time, because the module
    constant is imported elsewhere and mutating it would move other people's numbers."""
    return {"type": "object",
            "properties": {"summary": {"type": "string", "maxLength": max_characters}},
            "required": ["summary"], "additionalProperties": False}


def _what_happened_today(looks_today: Sequence[LookRecord],
                         asked_objects: Sequence[str],
                         tell_the_model_everything_it_saw: bool = False) -> str:
    """The looks in words.

    `only_these_objects` decides how much of the look reaches the model. Passing the
    asked-about objects filters the description down to them, which is what every
    number before 2026-09-24 was measured with; passing None renders every sighting
    and every absence the look recorded. See
    study_settings.Settings.tell_the_model_everything_it_saw. The renderer is the
    same either way, so both note formats are affected identically.
    """
    if not looks_today:
        return "The robot did not look anywhere today."
    only_these = None if tell_the_model_everything_it_saw else asked_objects
    return "\n\n".join(describe_look_for_the_model(look, only_these)
                       for look in looks_today)


def _shared_preamble(household: FrozenHousehold, day: int,
                     looks_today: Sequence[LookRecord],
                     tell_the_model_everything_it_saw: bool = False,
                     name_the_objects_it_will_be_quizzed_on: bool = True,
                     a_message_tonight: Optional[str] = None) -> List[str]:
    """Every word here is the same for both ways of writing notes.

    `name_the_objects_it_will_be_quizzed_on` decides whether the quiz set is
    ENUMERATED here. Naming it is the default and is what every number before
    2026-09-24 was measured with. The alternative exists because the enumeration
    confounds the rich-looks arm: see
    study_settings.Settings.name_the_objects_it_will_be_quizzed_on. The substitute
    sentence still says the robot will be asked where a thing is, so the task is
    unchanged; only what the writer is told to attend to moves.
    """
    what_it_is_asked_about = (
        "The things the robot is asked about: " + ", ".join(household.asked_objects)
        if name_the_objects_it_will_be_quizzed_on else
        "Later, someone will ask the robot where a thing in this home is. Nobody has "
        "said which things, so anything in this home may be asked about.")
    # The told-it arm's one sentence, if tonight is one of its two nights. It sits
    # here, with what the robot knows about today, rather than among the
    # instructions: it is a fact about the household and not a thing to do.
    told = [a_message_tonight, ""] if a_message_tonight else []
    return [
        f"It is the end of day {day}.",
        "",
        *told,
        "The rooms of this home and the spots in each:",
        *[f"- {room}: {', '.join(household.places_in_room[room])}"
          for room in household.rooms],
        "",
        what_it_is_asked_about,
        "",
        "Nobody tells the robot where anything is. What it saw today, and what it",
        "looked for and did not find, is all it has:",
        "",
        _what_happened_today(looks_today, household.asked_objects,
                             tell_the_model_everything_it_saw),
        "",
    ]


def rewrite_the_notes_wholesale(notes: Notes, household: FrozenHousehold, day: int,
                                time: int, looks_today: Sequence[LookRecord],
                                client: LLMClient,
                                read_budget_lines: int = 8,
                                tell_the_model_everything_it_saw: bool = False,
                                wording: str = AS_IT_STANDS,
                                name_the_objects_it_will_be_quizzed_on: bool = True,
                                describe_the_person: bool = False,
                                a_message_tonight: Optional[str] = None,
                                say_the_notes_must_fit_a_budget: bool = True,
                                summary_max_characters: int = SUMMARY_MAX_CHARACTERS,
                                max_tokens: int = 900
                                ) -> Dict[str, Any]:
    if wording not in REWRITE_WORDINGS:
        raise ValueError(f"unknown rewrite wording {wording!r}; "
                         f"known: {sorted(REWRITE_WORDINGS)}")
    previous = notes.newest_summary()
    # Every word this arm is told now comes from what_the_robot_is_told: the shared part
    # from `the_nightly_prompt` and this arm's own part from `HOW_YOUR_MEMORY_WORKS`. It
    # used to be built here, and that is how the two formats came to be told opposite
    # things about loss without anyone noticing. `wording` and the old budget block are
    # gone: the budget because there is no length limit any more, and the wordings because
    # a per-arm wording experiment is exactly the confound this structure exists to stop.
    lines = told.the_nightly_prompt(
        "wholesale rewrite", household, day,
        _what_happened_today(looks_today, household.asked_objects,
                             tell_the_model_everything_it_saw),
        previous or "(nothing written yet)", summary_max_characters,
        name_the_things_it_is_asked_about=name_the_objects_it_will_be_quizzed_on,
        a_message_tonight=a_message_tonight,
        extra_before_the_instruction=(DESCRIBE_THE_PERSON if describe_the_person else ()))
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n".join(lines)}]
    text, _ = client.complete(messages, summary_schema(summary_max_characters),
                              max_tokens=max_tokens)
    summary = None
    if text:
        try:
            summary = json.loads(text).get("summary")
        except ValueError:
            summary = None
    if summary is None:
        summary = previous or ""
        failed = True
    else:
        failed = False
    notes.write_nightly_summary(day, time, summary)
    return {"day": day, "model_call_failed": failed,
            "wording": wording,
            "the_prompt_asked_it_to_describe_the_person": describe_the_person,
            "the_message_it_was_told_tonight": a_message_tonight,
            "the_prompt_told_it_to_fit_a_line_budget": say_the_notes_must_fit_a_budget,
            "the_schema_ceiling_on_the_whole_summary": summary_max_characters,
            "the_summary_is_at_the_schema_ceiling":
                len(summary) >= summary_max_characters - 5,
            "n_lines_of_summary": len(summary_lines(summary)),
            "n_characters": len(summary),
            "identical_to_last_night": bool(previous) and summary.strip() == previous.strip()}


def write_the_notes_incrementally(notes: Notes, household: FrozenHousehold, day: int,
                                  time: int, looks_today: Sequence[LookRecord],
                                  client: LLMClient,
                                  tell_the_model_everything_it_saw: bool = False,
                                  name_the_objects_it_will_be_quizzed_on: bool = True,
                                  keep_rival_beliefs: bool = False,
                                  describe_the_person: bool = False,
                                  a_message_tonight: Optional[str] = None,
                                  max_edits: Optional[int] = None,
                                  edit_max_tokens: Optional[int] = None
                                  ) -> Dict[str, Any]:
    # At WRITE time the claim store sees all of its claims - the read budget is an
    # answer-time window, not an amnesia. What it may SHOW at answer time is capped
    # in memory_notes.what_the_robot_can_read.
    current = ("\n".join(c.as_plain_words() for c in notes.claims_in_reading_order())
               if notes.claims else "(nothing written yet)")
    # As above: the shared words come from `the_nightly_prompt` and this arm's own words
    # from `HOW_YOUR_MEMORY_WORKS["incremental edits"]`. The instruction-first ordering
    # that this arm needs is a property of that entry, and the empty-edit-list failure it
    # guards against must be re-smoked after any change to it.
    lines = told.the_nightly_prompt(
        "incremental edits", household, day,
        _what_happened_today(looks_today, household.asked_objects,
                             tell_the_model_everything_it_saw),
        current, 240,
        name_the_things_it_is_asked_about=name_the_objects_it_will_be_quizzed_on,
        a_message_tonight=a_message_tonight,
        extra_before_the_instruction=(
            list(KEEP_RIVAL_BELIEFS if keep_rival_beliefs else [])
            + list(DESCRIBE_THE_PERSON if describe_the_person else [])))
    # On a night whose look found something the robot is asked about, the schema
    # REQUIRES at least one edit. Measured on 2026-09-24, one household, 29
    # nights: without it the model replied {"edits": []} on 17 of the 19 nights
    # that saw an asked-about object, ending the month with 6 claims for a
    # 15-object household while the rewrite arm wrote 1,885 characters. Wording
    # the instruction three different ways did not fix it; the schema does. A
    # night that found nothing is still allowed to make no edit, so the model is
    # never forced to invent.
    saw_something = any(s["object_id"] in set(household.asked_objects)
                        for look in looks_today for s in look.sightings)
    # `max_edits=None` keeps the old constant of eight, so every number measured
    # before 2026-09-24 reproduces exactly; run_one_cell passes the household's own
    # allowance. The token budget has to move with it or the cap simply reappears as a
    # truncated reply: 1,400 tokens does not hold thirty edits.
    cap = EDITS_SCHEMA["properties"]["edits"]["maxItems"] if max_edits is None else max_edits
    schema = EDITS_SCHEMA if max_edits is None else edits_schema(cap)
    if saw_something:
        schema = json.loads(json.dumps(schema))
        schema["properties"]["edits"]["minItems"] = 1
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n".join(lines)}]
    # HOW MANY TOKENS THE EDITS MAY TAKE, and why this is a parameter.
    #
    # `200 + 120 * cap` was measured to be TOO SMALL and it fails SILENTLY. On 2026-09-24,
    # with cap 22, the model produced 8,855 characters and was cut off mid-array by the
    # 2,840-token ceiling; the text then does not parse, `edits` comes back empty, and the
    # night writes NOTHING - while `model_call_failed` stays False because text was
    # returned. Measured on that completion: 3.12 characters per token, about 420
    # characters an edit, so about 135 tokens an edit against the 120 the formula allows.
    # The worst case a schema permits - statement 240, why 240, holds_under 120, plus six
    # observation ids - is nearer 700 characters, or 225 tokens.
    #
    # So the default is left EXACTLY as it was, because other work is running under it and
    # every number measured with it must reproduce, and callers who need a budget that
    # cannot truncate pass one. A caller that hits this ceiling loses the whole night, so
    # the only safe value is one with margin over the worst case the schema allows.
    # Corrected 2026-09-24, same night: the too-small formula was left as the default for
    # every caller that sets an allowance, which meant a caller could lose whole nights to
    # a silent truncation it never asked for. `max_edits` is one night old and NOTHING was
    # ever measured under it, so there is no number to preserve on that path - only the
    # `max_edits is None` path has to reproduce, and it does, exactly. The safe formula has
    # margin over the worst case the schema allows.
    if edit_max_tokens is not None:
        budget = edit_max_tokens
    elif max_edits is None:
        budget = 1400
    else:
        # 260 a slot was sized when `why` was capped at 240 characters. It is 600 now, so a
        # single edit can carry 240 + 120 + 600 characters of text, about 310 tokens before the
        # JSON around it. 420 keeps the margin the old number had.
        budget = 300 + 420 * cap
    text, _ = client.complete(messages, schema, max_tokens=budget)
    edits: List[dict] = []
    # WHETHER IT PARSED IS NOT THE SAME AS WHETHER IT WAS EMPTY. My first version of
    # `the_completion_did_not_parse` was `bool(text) and not edits`, which is True for a
    # night that legitimately returned `{"edits": []}` - and night 0 does exactly that,
    # because day 0 has no questions, so no looks, so nothing to write. That made every
    # cell fail a guard on night 0. The flag has to come from the parse itself.
    did_not_parse = False
    if text:
        try:
            edits = json.loads(text).get("edits") or []
        except ValueError:
            edits = []
            did_not_parse = True
    claims_before = [c.statement for c in notes.claims]
    applied = {"add": 0, "revise": 0, "record evidence": 0}
    rejected: List[str] = []
    for edit in edits:
        action = edit.get("action")
        try:
            if action == "add":
                if not edit.get("statement"):
                    rejected.append("add with no statement")
                    continue
                notes.add_claim(edit["statement"], edit.get("holds_under") or "not said",
                                day, time,
                                edit.get("supporting_observation_ids") or (),
                                edit.get("status") or PROVISIONAL)
                applied["add"] += 1
            elif action == "revise":
                notes.revise_claim(edit["claim_id"], day, time,
                                   new_statement=edit.get("statement"),
                                   new_holds_under=edit.get("holds_under"),
                                   new_status=edit.get("status"),
                                   why=edit.get("why") or "")
                applied["revise"] += 1
            elif action == "set a note aside":
                notes.revise_claim(edit["claim_id"], day, time, new_standing=SET_ASIDE,
                                   why=edit.get("why") or "")
                applied["set a note aside"] = applied.get("set a note aside", 0) + 1
            elif action == "bring a note back":
                notes.revise_claim(edit["claim_id"], day, time,
                                   new_standing=STILL_STANDING,
                                   why=edit.get("why") or "")
                applied["bring a note back"] = applied.get("bring a note back", 0) + 1
            elif action == "record evidence":
                for ids, supports in ((edit.get("supporting_observation_ids") or (), True),
                                      (edit.get("contradicting_observation_ids") or (), False)):
                    if ids:
                        notes.record_evidence(edit["claim_id"], ids, day, time, supports)
                applied["record evidence"] += 1
            else:
                rejected.append(f"unknown action {action!r}")
        except (KeyError, ValueError) as problem:
            rejected.append(f"{action}: {problem}")
    notes.written_up_to_day = day
    return {"day": day, "model_call_failed": not text,
            "the_prompt_asked_it_to_keep_rival_beliefs": keep_rival_beliefs,
            "the_prompt_asked_it_to_describe_the_person": describe_the_person,
            "the_message_it_was_told_tonight": a_message_tonight,
            "the_token_budget_for_the_edits": budget,
            # The silent failure above, made loud: text came back but did not parse, so
            # the night wrote nothing and nothing else in this report would say so.
            "the_completion_did_not_parse": did_not_parse,
            "the_look_saw_something_it_is_asked_about": saw_something,
            "at_least_one_edit_was_required": saw_something,
            "n_edits_offered": len(edits),
            # Whether tonight ran into the write-time edits-per-night cap
            # (EDITS_SCHEMA's maxItems), not the read-time line budget. Found
            # 2026-09-24: an instruction that costs more than one edit per fact
            # (e.g. keep_rival_beliefs, which asks for an add AND a revise per
            # moved object) can hit this cap far more often than the control
            # condition covering the identical set of facts, which silently
            # bounds how much the memory may change in one night rather than
            # how big it may ever get. Report this beside any count of claims
            # written or any coverage/accuracy number, or the difference may be
            # a budget artefact rather than a content one.
            "how_many_edits_it_was_allowed": cap,
            "the_edits_cap_bit": len(edits) >= cap,
            "applied": applied,
            "rejected": rejected, "n_claims_now": len(notes.claims),
            "were_the_edits_vacuous": were_the_edits_vacuous(edits, claims_before,
                                                            household)}


def write_the_notes(notes: Notes, household: FrozenHousehold, day: int, time: int,
                    looks_today: Sequence[LookRecord], client: LLMClient,
                    read_budget_lines: int = 8,
                    tell_the_model_everything_it_saw: bool = False,
                    wording: str = AS_IT_STANDS,
                    name_the_objects_it_will_be_quizzed_on: bool = True,
                    keep_rival_beliefs: bool = False,
                    describe_the_person: bool = False,
                    a_message_tonight: Optional[str] = None,
                    say_the_notes_must_fit_a_budget: bool = True,
                    summary_max_characters: int = SUMMARY_MAX_CHARACTERS,
                    wholesale_max_tokens: int = 900,
                    max_edits: Optional[int] = None,
                    edit_max_tokens: Optional[int] = None
                    ) -> Dict[str, Any]:
    """Both prompt settings default to what was measured before 2026-09-24 - the look
    description filtered to the asked-about objects, and the quiz list enumerated in
    the prompt - so every earlier number reproduces unchanged. The new arms set them
    from study_settings.Settings."""
    if notes.how_memory_is_written == "wholesale rewrite":
        if keep_rival_beliefs:
            raise ValueError(
                "keep_rival_beliefs is a claim-store instruction: the wholesale "
                "rewrite has no claim to set aside, so switching it on there would "
                "silently be a no-op arm. Refusing.")
        return rewrite_the_notes_wholesale(
            notes, household, day, time, looks_today, client, read_budget_lines,
            tell_the_model_everything_it_saw, wording,
            name_the_objects_it_will_be_quizzed_on=name_the_objects_it_will_be_quizzed_on,
            describe_the_person=describe_the_person,
            a_message_tonight=a_message_tonight,
            say_the_notes_must_fit_a_budget=say_the_notes_must_fit_a_budget,
            summary_max_characters=summary_max_characters,
            max_tokens=wholesale_max_tokens)
    return write_the_notes_incrementally(
        notes, household, day, time, looks_today, client,
        tell_the_model_everything_it_saw,
        name_the_objects_it_will_be_quizzed_on=name_the_objects_it_will_be_quizzed_on,
        keep_rival_beliefs=keep_rival_beliefs,
        describe_the_person=describe_the_person,
        a_message_tonight=a_message_tonight,
        max_edits=max_edits, edit_max_tokens=edit_max_tokens)
    # NB the claim-store writing prompt has never mentioned a line budget - only the
    # wholesale one does - so `say_the_notes_must_fit_a_budget` is not forwarded here
    # and has no effect on this format. For the claim store, removing the read-time cap
    # IS the whole of "no length limit".

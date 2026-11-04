"""Notes that may not say where anything is, because the robot can read its own record.

The arm this writes for pairs with `the_log_the_robot_reads`: when the robot is asked
where something is, it is shown every sighting of it with the day and the time, and every
room it has looked in without finding it. So a note saying "the mug is on the desk" is
worse than useless - the record says the same thing with dates, and more exactly.

What a record CANNOT say at a glance is what this arm's notes are for:
  - when each person is at home, and where, at what times of day;
  - which things move together, or move when somebody does something;
  - what has changed lately, and when it changed;
  - where to look first for a thing, and why.

Measured before this arm existed, over 1,906 claims: 91% named a place and 0.3% said
anything about a time, a condition or a change. Those two numbers are the gate on this
arm, in `did_it_stop_copying_the_record`, and neither can be passed by relabelling.

The claim store underneath is unchanged - add, revise keeping the old wording, set aside -
so the question the study turns on, whether a belief written for the ordinary routine
survives the disrupted one, is still asked of this arm in exactly the same way.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from baselines.patrol.llm import LLMClient
from self_improve import what_the_robot_is_told as told
from self_improve import write_the_notes as writing
from self_improve.frozen_household import FrozenHousehold
from self_improve.looking import LookRecord, describe_look_for_the_model
from self_improve.memory_notes import (ESTABLISHED, PROVISIONAL, SET_ASIDE,
                                      STILL_STANDING, THE_LOG_AND_THE_ROUTINE, Notes)

# How many notes a night may write or change. NOT derived from the number of objects,
# because this arm does not write one note per object - that is the whole point of it. It
# is deliberately generous: the hypothesis is that useful notes about a household are few,
# so an allowance it never approaches is evidence for the hypothesis, and a night that
# reaches it is a finding and a reason to raise it. Whether it bound is recorded every
# night as `the_edits_cap_bit`.
EDITS_A_NIGHT = 16

# Words that make a line a statement about a place, and words that make it a statement
# about time, a condition or a change. Used only to score compliance; never in a prompt.
# `are` as well as `is`: without it, "glasses_hana ARE on the nightstand" scored as not
# naming a place. 18 of 2,279 claims on disk, all plural-noun classes - coasters, glasses,
# headphones - and the bias runs the wrong way for a gate, because a missed place-claim
# makes this arm look more compliant than it is. Found by the research agent.
A_PLACE = re.compile(r"\b(is|are) (on|in|at|inside) the\b|\b(kept|live|lives) "
                     r"(on|in|at)\b", re.I)
A_TIME_OR_A_CHANGE = re.compile(
    r"\b(morning|afternoon|evening|night|midday|noon|breakfast|lunch|dinner|"
    r"weekday|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{1,2}:\d{2}|every day|each day|usually|often|sometimes|when |while |after |"
    r"before |during |routine|habit|pattern|tends to|used to|no longer|since day|"
    r"has changed|changed on|stopped|started|now that|instead of|at home|out of the "
    r"house|first look|look first)\b", re.I)

_THE_OLD_INSTRUCTION_NOW_IN_what_the_robot_is_told: List[str] = [
    # The instruction first and the reason after it. Measured on this model: with the
    # reason first it answered a night of six fresh sightings with an empty list of edits,
    # on every wording tried.
    #
    # NO EXAMPLES OF WHAT TO WRITE. An earlier draft listed five kinds of note to write -
    # where people are at different times, what travels with what, what changed and when -
    # and that list is five hypotheses put into the model's head instead of found by it.
    # One of them described this study's own disruption. The list is gone. What is left
    # says only that copying the record is not worth a line, and asks for its own thinking.
    "WHAT TO DO NOW. It is the end of the day. Write down anything you have worked out "
    "about this home. Do not just copy your record of where things are.",
    "",
    "Your record already holds where everything has been, with the days and the times, "
    "and you are shown all of it every time you are asked. Repeating it in a note adds "
    "nothing.",
    "",
    "It may help to think about what was surprising or important in what you saw today, "
    "and why it happened. Write down what you think is going on in this home.",
    "",
    "It may help to say when each note is true, if it is only true at some times. That is "
    "what lets you choose between two notes that disagree.",
    "",
    "It may help to say what you would expect to see if a note were wrong, and to record "
    "it against that note when you do see it.",
    "",
    "When something you believed stops being true, make your best judgement on whether to "
    "keep it, revise it, or archive it. It may help to record which sightings informed "
    "your choice.",
    "",
    "You can point at any sighting using the number in square brackets beside it.",
    "",
    "Each note is cut off after 240 characters, so keep one idea to a note.",
]


def the_day_this_arm_sees(looks_today: Sequence[LookRecord],
                          asked_objects: Sequence[str],
                          tell_it_everything: bool = True) -> str:
    """Today's looks, exactly as every other arm sees them.

    IT USED TO LEAVE OUT THE ABSENCE LIST, and that was wrong. Dropping "I looked in the
    kitchen and the mug was not there" saved about 15% of the characters and cost this arm
    evidence that its own control was given - so a difference in results between the two
    could have been a difference in what they were shown. Absence is evidence. Found by the
    wave agent diffing the two renderers on looks that actually had absences; its first
    test used a look with none and the two came out identical.

    It also calls `write_the_notes._what_happened_today`, which is what the docstring used
    to claim while in fact calling the renderer directly - so the check in search_driven
    that the day was rendered exactly once a night now really does hold for this arm.
    """
    return writing._what_happened_today(looks_today, asked_objects, tell_it_everything)


def did_it_stop_copying_the_record(notes: Notes) -> Dict[str, Any]:
    """The gate: are these notes about the household, or still a table of places?

    THE BASELINE IS A MEASUREMENT, NOT A CONSTANT, and it has to come from the same
    estimator on a named population or the gate compares two different things. The number
    quoted when this arm was designed - 91% naming a place, 0.3% naming a time - came from
    a regex almost but not exactly this one, over the 1,906 claim-store claims on disk at
    22:30 on 2026-09-24, and the population has grown since. So the threshold is not
    hard-coded here: `results/self_improve/what_the_old_notes_were_about.py` recomputes it
    with THIS regex over whatever claim-store cells exist, and its output is what this
    gate is read against. A structural extractor cannot be used instead: it only knows the
    objects the robot is asked about, and with the quiz list off most claims are about the
    kettle and the dog bowl, so it scored 45.8% where the regex scored 97.2% on the same
    claims.
    """
    live = [c for c in notes.claims if c.folded_into is None]
    if not live:
        return {"n_claims": 0, "share_that_name_a_place": None,
                "share_that_name_a_time_or_a_change": None,
                "share_whose_condition_is_just_current": None,
                "the_baseline_is_recomputed_by":
                    "results/self_improve/what_the_old_notes_were_about.py"}
    places = sum(1 for c in live if A_PLACE.search(c.statement))
    times = sum(1 for c in live if A_TIME_OR_A_CHANGE.search(c.statement))
    bare = sum(1 for c in live if c.holds_under.strip().lower() in ("current", "", "not said"))
    return {"n_claims": len(live),
            "share_that_name_a_place": places / len(live),
            "share_that_name_a_time_or_a_change": times / len(live),
            "share_whose_condition_is_just_current": bare / len(live),
            "the_baseline_is_recomputed_by":
                "results/self_improve/what_the_old_notes_were_about.py"}


def write_the_notes_about_the_routine(notes: Notes, household: FrozenHousehold, day: int,
                                     time: int, looks_today: Sequence[LookRecord],
                                     client: LLMClient,
                                     tell_the_model_everything_it_saw: bool = True,
                                     name_the_objects_it_will_be_quizzed_on: bool = False,
                                     a_message_tonight: Optional[str] = None,
                                     max_edits: int = EDITS_A_NIGHT,
                                     keeps_a_profile_of_each_person: bool = False,
                                     who_lives_here: str = "",
                                     ask_it_what_will_change: bool = False) -> Dict[str, Any]:
    """One night. The same claim store and the same edit actions, a different job.

    `keeps_a_profile_of_each_person` adds ONE thing and changes nothing else: the model is shown
    who lives here and its own profile of each person, and writes each profile again before it
    writes a single claim edit. It is the same function rather than a copy of it, because the six
    accidental differences that confounded the first comparison all came from a copy.
    """
    live = [c for c in notes.claims if c.folded_into is None]
    current = ("\n".join(c.as_plain_words() for c in notes.claims_in_reading_order())
               if live else "(nothing written yet)")
    # Every word is in what_the_robot_is_told: the shared part in `the_nightly_prompt`
    # and this arm's own part in `HOW_YOUR_MEMORY_WORKS`. Nothing about this arm's prompt
    # lives in this file any more, which is what makes it comparable with the others.
    lines = told.the_nightly_prompt(
        THE_LOG_AND_THE_ROUTINE, household, day,
        the_day_this_arm_sees(looks_today, household.asked_objects,
                              tell_the_model_everything_it_saw), current, 240,
        name_the_things_it_is_asked_about=name_the_objects_it_will_be_quizzed_on,
        a_message_tonight=a_message_tonight,
        how_many_notes=max_edits,
        extra_before_the_instruction=(
            (told.the_profiles(notes.profiles, who_lives_here)
             if keeps_a_profile_of_each_person else [])
            + (told.what_to_write_before_you_have_seen_it()
               if ask_it_what_will_change else [])))
    schema = writing.edits_schema(max_edits)
    if keeps_a_profile_of_each_person:
        # THE PROFILES ARE DECLARED FIRST, and the order is the whole point: the server enforces
        # the schema as it generates and thinking is switched off, so a field declared after the
        # edits would be written after every claim had already been decided. That is the same
        # fault that had every room chosen before any reasoning.
        schema = json.loads(json.dumps(schema))
        schema["properties"] = {
            "people": {"type": "array", "maxItems": 6,
                       "items": {"type": "object", "properties": {
                           "name": {"type": "string", "maxLength": 40},
                           "profile": {"type": "string", "maxLength": 900}},
                           "required": ["name", "profile"], "additionalProperties": False}},
            **schema["properties"]}
        schema["required"] = ["people"] + [r for r in schema.get("required", [])]
    # No minimum is imposed. The other arms require an edit on a night that saw an
    # asked-about object, because for them every sighting is a fact to record. Here most
    # nights genuinely have nothing new to say about how the household works, and forcing
    # an edit would make the model invent a routine. Nights with no edit are counted
    # instead, as `n_nights_with_nothing_to_add`, by the caller.
    text, _ = client.complete(
        [{"role": "system", "content": told.WRITING_SYSTEM},
         {"role": "user", "content": "\n".join(lines)}],
        # 260 a slot was sized when `why` was capped at 240 characters; it is 600 now, so
        # sixteen edits can want about 4,900 tokens against the 4,460 this granted. A cut-off
        # completion does not parse and the night writes nothing, so the arm would have lost
        # nights silently. `write_the_notes` was already corrected; this copy was not.
        schema, max_tokens=300 + 420 * max_edits
        # Six profiles of 900 characters is about 1,400 tokens, and a reply cut off in the
        # middle is not JSON, so the night would write nothing at all.
        + (1600 if keeps_a_profile_of_each_person else 0))
    edits: List[dict] = []
    did_not_parse = False
    if text:
        try:
            edits = json.loads(text).get("edits") or []
        except ValueError:
            edits, did_not_parse = [], True
    applied = {"add": 0, "revise": 0, "record evidence": 0}
    rejected: List[str] = []
    n_profiles = 0
    if keeps_a_profile_of_each_person and text:
        try:
            for person in (json.loads(text).get("people") or []):
                name, profile = (person.get("name") or "").strip(), (person.get("profile") or "").strip()
                if not name or not profile:
                    rejected.append("a profile with no name or no words")
                    continue
                notes.profiles[name] = profile
                n_profiles += 1
        except ValueError:
            pass
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
            "the_completion_did_not_parse": did_not_parse,
            "how_many_edits_it_was_allowed": max_edits,
            "it_was_asked_what_will_change": ask_it_what_will_change,
            "n_profiles_written": n_profiles,
            "n_characters_of_profile": sum(len(v) for v in notes.profiles.values()),
            "n_edits_offered": len(edits),
            "the_edits_cap_bit": len(edits) >= max_edits,
            "applied": applied, "rejected": rejected,
            "it_had_nothing_to_add": not edits,
            "n_claims_now": len([c for c in notes.claims if c.folded_into is None]),
            "did_it_stop_copying_the_record": did_it_stop_copying_the_record(notes)}

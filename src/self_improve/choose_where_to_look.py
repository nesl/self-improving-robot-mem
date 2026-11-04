"""The robot chooses the room, instead of following the schedule. Factor two.

The hypothesis is that a robot remembers a home better when it looks where its
own guesses disagree. So the choice is not "pick a likely room": it is a
structured action that has to name the disagreement it is going to settle.

  the room it will look in;
  two competing claims from its own notes;
  for each claim, which of the things it is asked about it expects to find in
      that room if that claim holds - and an empty list is a real prediction,
      because a claim that puts the mug on the desk predicts an empty living room;
  for each outcome, what it would change in its notes.

Every field is logged. `look_diagnosticity_check` then asks, mechanically, whether
the two expectations differ at all and whether the look settled which claim held.
A look whose two claims predict the same thing cannot teach the robot anything and
is the degenerate case of this study - it would never show up in accuracy.

The same number of looks, at the same times, as the fixed schedule gets. Only
where it looks differs.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Sequence

from baselines.patrol.llm import LLMClient
from self_improve.frozen_household import FrozenHousehold, LookTarget, plain_place_name
from self_improve.looking import ChosenLook, FixedLookSchedule, LookRecord, TheHouseAsSeen
from self_improve.memory_notes import Notes

SYSTEM = ("You help a home robot decide where to look in a house. Nobody tells the "
          "robot where anything is; looking is the only way it learns. Answer with "
          "JSON only.")


def how_many_claims_mention_each_room(household: FrozenHousehold, notes: Notes
                                      ) -> Dict[str, int]:
    """How many lines of the notes say anything about each room."""
    if notes.how_memory_is_written == "wholesale rewrite":
        from self_improve.memory_notes import summary_lines
        statements = [line.lower() for line in summary_lines(notes.newest_summary() or "")]
    else:
        statements = [c.statement.lower() for c in notes.claims]
    counts = {}
    for room in household.rooms:
        places = household.places_in_room[room]
        counts[room] = sum(
            1 for statement in statements
            if room in statement or any(place in statement
                                        or plain_place_name(place) in statement
                                        for place in places))
    return counts


def what_the_notes_predict_for_each_room(household: FrozenHousehold, notes: Notes
                                         ) -> Dict[str, List[str]]:
    """For every room in the house, which asked-about things the notes place there.

    This is the fix for the collapse. Generating candidates by scanning the notes
    for disagreements means a room the notes never mention cannot enter the
    candidate list - and that is exactly the room a new disruption has moved things
    into. Measured before the fix: all four chooser variants visited ONE room ten
    times out of ten and none ever reached the room the change had moved into.

    Enumerating every room keeps faith with the hypothesis rather than weakening
    it: finding an object in room Y refutes a claim placing it in room Z, so every
    room is a possible site of disagreement. The old version could only see
    disagreements it had already written down.
    """
    if notes.how_memory_is_written == "wholesale rewrite":
        text = (notes.newest_summary() or "").lower()
        statements = [text]
    else:
        statements = [c.statement.lower() for c in notes.claims]

    predicted: Dict[str, List[str]] = {room: [] for room in household.rooms}
    for room in household.rooms:
        places = household.places_in_room[room]
        for statement in statements:
            mentions_room = room in statement or any(
                place in statement or plain_place_name(place) in statement
                for place in places)
            if not mentions_room:
                continue
            for object_id in household.asked_objects:
                if object_id.lower() in statement and object_id not in predicted[room]:
                    predicted[room].append(object_id)
    return predicted


def choice_schema(rooms: Sequence[str], asked_objects: Sequence[str],
                  require_the_two_claims_to_be_about_the_same_object: bool
                  ) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "room": {"type": "string", "enum": list(rooms)},
            "first_claim_id": {"type": ["string", "null"]},
            "first_claim": {"type": "string", "maxLength": 240},
            "second_claim_id": {"type": ["string", "null"]},
            "second_claim": {"type": "string", "maxLength": 240},
            "objects_expected_if_the_first_claim_holds": {
                "type": "array", "items": {"type": "string"}, "maxItems": 12},
            "objects_expected_if_the_second_claim_holds": {
                "type": "array", "items": {"type": "string"}, "maxItems": 12},
            "what_i_would_change_if_the_first_claim_holds": {"type": "string", "maxLength": 240},
            "what_i_would_change_if_the_second_claim_holds": {"type": "string", "maxLength": 240},
            "reasoning": {"type": "string", "maxLength": 400},
        },
        "required": ["room", "first_claim_id", "first_claim", "second_claim_id",
                     "second_claim",
                     "objects_expected_if_the_first_claim_holds",
                     "objects_expected_if_the_second_claim_holds",
                     "what_i_would_change_if_the_first_claim_holds",
                     "what_i_would_change_if_the_second_claim_holds", "reasoning"],
        "additionalProperties": False}
    if require_the_two_claims_to_be_about_the_same_object:
        # The correction to the specification: the two claims must disagree about
        # ONE thing's whereabouts. Making the object a required field of the action
        # is what makes that checkable rather than hoped for.
        schema["properties"]["the_thing_the_two_claims_disagree_about"] = {
            "type": "string", "enum": list(asked_objects)}
        schema["required"].append("the_thing_the_two_claims_disagree_about")
    return schema


def candidate_order(household: FrozenHousehold, day: int,
                    shuffle_with_seed: Optional[int] = None) -> List[str]:
    """The order the rooms are PRESENTED in.

    By default this is the household's own sorted order, which is the same every
    night - and that is a problem we have to be able to rule out. `household.rooms`
    sorts alphabetically, so in all three households measured so far the bathroom sat
    at position 2, directly after the balcony, and the naive chooser picked the
    bathroom every single night. Two explanations fit: the bathroom genuinely wins
    because it holds several same-class objects belonging to different people, or the
    model is taking the first plausible item in a fixed list.

    Passing a seed shuffles the order per night, which settles it: if the choice
    follows the position the order is an artefact and the arm is a bug; if it stays on
    the bathroom while its position moves, it is reasoning.
    """
    rooms = list(household.rooms)
    if shuffle_with_seed is not None:
        random.Random(f"{household.name}/{day}/{shuffle_with_seed}").shuffle(rooms)
    return rooms


def choice_prompt(household: FrozenHousehold, notes: Notes, day: int,
                  at_clock: str,
                  days_since_each_room_was_looked_in: Optional[Dict[str, Any]] = None,
                  require_the_two_claims_to_be_about_the_same_object: bool = False,
                  enumerate_every_room: bool = True,
                  presented_order: Optional[Sequence[str]] = None
                  ) -> List[dict]:
    """The instruction comes first and the explanation after.

    Not a stylistic choice. Measured on 2026-09-24 on the note-writing prompt:
    with the explanation first, this model answered a night of six fresh sightings
    with an empty edit list, every variant tried. The same ordering rule is used
    here, and any edit to this prompt must be re-smoked for the same failure -
    check that the first day produces a filled-in action, not a default room with
    empty claims.
    """
    # At look time the robot sees all of its notes. The 8-line read budget is an
    # ANSWER-time window; it is not amnesia, and the fixed-schedule arm reads
    # nothing at look time because it makes no choice.
    predicted = (what_the_notes_predict_for_each_room(household, notes)
                 if enumerate_every_room else {})
    if notes.how_memory_is_written == "wholesale rewrite":
        current = notes.newest_summary() or "(nothing written yet)"
    else:
        current = ("\n".join(c.as_plain_words() for c in notes.claims_in_reading_order())
                   if notes.claims else "(nothing written yet)")

    if require_the_two_claims_to_be_about_the_same_object:
        what_to_do = (
            "Below is every room in this house and what your notes currently predict "
            "would be found in it. Choose the ONE room whose contents your notes are "
            "least able to predict - and then say what looking there would settle. "
            "Pick one thing your notes cannot place with confidence, name two claims "
            "that disagree about where THAT thing is, say which of the things you are "
            "asked about you would find in the room if the first claim holds, which if "
            "the second holds, and what you would change in your notes either way.")
    else:
        what_to_do = (
            "Below is every room in this house and what your notes currently predict "
            "would be found in it. Choose the ONE room whose contents your notes are "
            "least able to predict - and then say what looking there would settle. "
            "Name two claims that cannot both be right, say which of the things you "
            "are asked about you would find in that room if the first one holds, which "
            "you would find if the second holds, and what you would change in your "
            "notes either way.")

    lines = [
        f"It is day {day}. You may look in ONE room, at {at_clock}.",
        "",
        what_to_do,
        "",
        "A room your notes say nothing about is a room you cannot predict at all. "
        "Things move: if something is missing from where your notes put it, it is "
        "somewhere your notes are not looking.",
        "",
        "Expecting to find NOTHING is a real answer: if one claim puts the mug on "
        "the desk, then that claim predicts an empty living room, and an empty list "
        "is how you say so. What matters is that the two lists are not the same - "
        "if both claims predict the same things in the room you pick, looking there "
        "cannot tell you which is right and the look is wasted.",
        "",
        "The rooms of this home and the spots in each:",
        *[f"- {room}: {', '.join(household.places_in_room[room])}"
          for room in (presented_order or household.rooms)],
        "",
        "The things you are asked about: " + ", ".join(household.asked_objects),
        "",
        "What your notes predict for each room:",
        *[f"- {room}: "
          + (", ".join(predicted[room]) if predicted.get(room)
             else "your notes say nothing about this room")
          for room in (presented_order or household.rooms)],
        "",
    ]
    if days_since_each_room_was_looked_in is not None:
        # The robot's own history of where it has been - information it plainly has,
        # not a hint about the world. Without it the chooser re-tested one room for
        # nine days of ten and never reached the room the change had moved into,
        # because a room it holds no claim about can never be the site of a
        # disagreement.
        lines += [
            "How long since you last looked in each room:",
            *[f"- {room}: "
              + ("never looked" if since is None
                 else "today" if since == 0
                 else "yesterday" if since == 1
                 else f"{since} days ago")
              for room, since in sorted(days_since_each_room_was_looked_in.items())],
            "",
            "A room you have not looked in for a long time, or never, may have "
            "changed without your notes showing any disagreement at all - your notes "
            "cannot disagree about a room they say nothing about.",
            "",
        ]
    lines += [
        "Your notes:",
        "",
        current,
    ]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(lines)}]


def choose_and_look(eyes: TheHouseAsSeen, household: FrozenHousehold, notes: Notes,
                    day: int, at_clock: str, time_of_day_seconds: int,
                    client: LLMClient,
                    fall_back_to: Optional[FixedLookSchedule] = None,
                    visit_index: int = 0,
                    tell_it_days_since_each_room_was_looked_in: bool = False,
                    require_the_two_claims_to_be_about_the_same_object: bool = False,
                    enumerate_every_room: bool = True,
                    shuffle_the_candidate_order_with_seed: Optional[int] = None
                    ) -> LookRecord:
    """One chosen look. Falls back to the schedule's room if the model does not
    answer, and records that it fell back rather than pretending it chose.

    The two repairs are separate switches so each can be left out in turn. Both off
    is the naive chooser, which we keep as an arm: it is the honest consequence of
    the hypothesis taken literally and it is a real negative result.
    """
    history = (eyes.days_since_each_room_was_looked_in(day)
               if tell_it_days_since_each_room_was_looked_in else None)
    presented = candidate_order(household, day, shuffle_the_candidate_order_with_seed)
    messages = choice_prompt(household, notes, day, at_clock, history,
                             require_the_two_claims_to_be_about_the_same_object,
                             enumerate_every_room, presented)
    text, _ = client.complete(
        messages,
        choice_schema(household.rooms, household.asked_objects,
                      require_the_two_claims_to_be_about_the_same_object),
        max_tokens=700)

    choice = None
    if text:
        try:
            choice = json.loads(text)
        except ValueError:
            choice = None

    if not choice or choice.get("room") not in set(household.rooms):
        targets = (fall_back_to.targets_for(day, visit_index) if fall_back_to
                   else [LookTarget(household.rooms[0], "room")])
        return eyes.look(targets, day, time_of_day_seconds,
                         "the model was asked but did not answer, so the schedule chose",
                         model_call_failed=True)

    # Logged so the two explanations for a collapsed chooser can be told apart: the
    # candidate list in the order it was shown, which room was chosen, and where in
    # that list it sat. A choice that tracks position 1 is an artefact; a choice that
    # stays on one room while its position moves is reasoning.
    chosen_position = presented.index(choice["room"]) + 1

    known = set(household.asked_objects)
    action = ChosenLook(
        target=choice["room"], target_kind="room",
        first_claim_id=choice.get("first_claim_id") or "",
        first_claim=choice.get("first_claim") or "",
        second_claim_id=choice.get("second_claim_id") or "",
        second_claim=choice.get("second_claim") or "",
        # Only objects this household is actually asked about can be predicted; a
        # hallucinated object id would otherwise count as a discriminating
        # prediction that no look could ever confirm or refute.
        objects_expected_if_the_first_claim_holds=[
            o for o in (choice.get("objects_expected_if_the_first_claim_holds") or [])
            if o in known],
        objects_expected_if_the_second_claim_holds=[
            o for o in (choice.get("objects_expected_if_the_second_claim_holds") or [])
            if o in known],
        what_i_would_change_if_the_first_claim_holds=
            choice.get("what_i_would_change_if_the_first_claim_holds") or "",
        what_i_would_change_if_the_second_claim_holds=
            choice.get("what_i_would_change_if_the_second_claim_holds") or "",
        reasoning=choice.get("reasoning") or "",
        the_thing_the_two_claims_disagree_about=
            choice.get("the_thing_the_two_claims_disagree_about") or "",
        what_the_notes_predicted_for_each_room=
            what_the_notes_predict_for_each_room(household, notes),
        how_many_claims_the_notes_hold_about_each_room=
            how_many_claims_mention_each_room(household, notes),
        candidate_rooms_in_the_order_they_were_shown=list(presented),
        position_of_the_room_it_chose=chosen_position,
        the_candidate_order_was_shuffled=shuffle_the_candidate_order_with_seed is not None)
    which = ("the model" if not (tell_it_days_since_each_room_was_looked_in
                                 or require_the_two_claims_to_be_about_the_same_object)
             else "the model, with its own look history and one-thing disagreements")
    return eyes.look([LookTarget(choice["room"], "room")], day, time_of_day_seconds,
                     which, chosen_look=action)

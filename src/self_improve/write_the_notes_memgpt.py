"""MemGPT's shape: a small working memory always in front of the model, plus an archive.

From Packer et al., MemGPT / Letta. Of the eight published memory systems surveyed in
results/self_improve/HOW_OTHER_MEMORIES_BOUND_THEIR_GROWTH.md it is the only one that
bounds WRITING, and the only one where the model manages its own limited attention rather
than being handed whatever a retrieval rule picked. Two pieces:

  working memory   a fixed number of characters that goes into every prompt, always. A
                   write that would overflow it is REFUSED, and the refusal is handed back
                   to the model, which then has to decide what to move out. That refusal is
                   the mechanism of the paper, not an error to be avoided.
  the archive      unbounded, never shown by default, reached only by searching it. In the
                   paper the search is over embeddings; here it is word overlap, which is
                   the same crude stand-in this project uses elsewhere and is stated as a
                   departure rather than hidden.

Three departures from the paper, all to be stated in the write-up:
  - the paper's agent calls functions inside one long conversation and can retry until the
    write fits. Every call here is stateless, so the model gets ONE retry after a refusal,
    with the refusal in front of it. A model that cannot free space in one retry loses that
    night's write, and that is counted rather than hidden.
  - the paper pages whole conversation history; here the thing being paged is notes about a
    household, because that is this study's memory.
  - search is word overlap, not vectors.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from baselines.patrol.llm import LLMClient
from self_improve import what_the_robot_is_told as told
from self_improve import write_the_notes as writing
from self_improve.frozen_household import FrozenHousehold
from self_improve.looking import LookRecord
from self_improve.memory_notes import (A_WORKING_MEMORY_AND_AN_ARCHIVE, ESTABLISHED,
                                       PROVISIONAL, SET_ASIDE, STILL_STANDING,
                                       WORKING_MEMORY_CHARACTERS, MemoryIsFull, Notes)

EDITS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array", "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["write a new note", "revise a note",
                                        "move into working memory",
                                        "move out to the archive",
                                        "attach evidence",
                                        "set a note aside", "bring a note back"]},
                    "claim_id": {"type": ["string", "null"]},
                    "statement": {"type": ["string", "null"], "maxLength": 240},
                    "holds_under": {"type": ["string", "null"], "maxLength": 120},
                    "into_working_memory": {"type": ["boolean", "null"]},
                    "status": {"type": ["string", "null"],
                               "enum": [PROVISIONAL, ESTABLISHED, None]},
                    "supporting_observation_ids": {"type": "array", "maxItems": 6,
                                                   "items": {"type": "string"}},
                    "contradicting_observation_ids": {"type": "array", "maxItems": 6,
                                                      "items": {"type": "string"}},
                    "why": {"type": "string", "maxLength": 240},
                },
                "required": ["action", "claim_id", "statement", "holds_under",
                             "into_working_memory", "status",
                             "supporting_observation_ids",
                             "contradicting_observation_ids", "why"],
                "additionalProperties": False}}},
    "required": ["edits"], "additionalProperties": False}


def what_it_can_see_of_its_memory(notes: Notes, searched_for: str = "") -> str:
    """The working memory in full, and how full it is. The archive is only counted."""
    working = notes.working_memory()
    used, cap = notes.characters_in_working_memory(), WORKING_MEMORY_CHARACTERS
    archived = len([c for c in notes.claims
                    if not c.in_working_memory and c.folded_into is None])
    lines = [f"YOUR WORKING MEMORY. {used} of {cap} characters used, "
             f"{cap - used} free."]
    lines += ([f"[{c.claim_id}] {c.statement}" for c in working] or
              ["(nothing in it yet)"])
    lines += ["", f"YOUR ARCHIVE holds {archived} other note(s). They are not shown to you "
                  f"unless you search for them."]
    if searched_for:
        hits = notes.search_the_archive(searched_for)
        lines += ["", f"Notes found in your archive when it was searched for "
                      f"\"{searched_for}\":"]
        lines += ([f"[{c.claim_id}] {c.statement}" for c in hits] or
                  ["(the search found nothing)"])
    return "\n".join(lines)


def _apply(notes: Notes, edits: Sequence[dict], day: int, time: int
           ) -> Dict[str, Any]:
    applied = {"write a new note": 0, "revise a note": 0, "move into working memory": 0,
               "move out to the archive": 0, "attach evidence": 0,
               "set a note aside": 0, "bring a note back": 0}
    rejected: List[str] = []
    refused_for_space: List[str] = []
    for edit in edits:
        action = edit.get("action")
        try:
            if action == "write a new note":
                if not edit.get("statement"):
                    rejected.append("a new note with no words in it")
                    continue
                claim = notes.add_claim(
                    edit["statement"], edit.get("holds_under") or "not said", day, time,
                    edit.get("supporting_observation_ids") or (),
                    edit.get("status") or PROVISIONAL)
                applied["write a new note"] += 1
                if edit.get("into_working_memory"):
                    try:
                        notes.put_in_working_memory(claim.claim_id, day, time)
                        applied["move into working memory"] += 1
                    except MemoryIsFull as full:
                        refused_for_space.append(f"{claim.claim_id}: {full}")
            elif action == "revise a note":
                notes.revise_claim(edit["claim_id"], day, time,
                                   new_statement=edit.get("statement"),
                                   new_holds_under=edit.get("holds_under"),
                                   new_status=edit.get("status"),
                                   why=edit.get("why") or "")
                applied["revise a note"] += 1
            elif action == "set a note aside":
                notes.revise_claim(edit["claim_id"], day, time, new_standing=SET_ASIDE,
                                   why=edit.get("why") or "")
                applied["set a note aside"] = applied.get("set a note aside", 0) + 1
            elif action == "bring a note back":
                notes.revise_claim(edit["claim_id"], day, time,
                                   new_standing=STILL_STANDING,
                                   why=edit.get("why") or "")
                applied["bring a note back"] = applied.get("bring a note back", 0) + 1
            elif action == "move into working memory":
                notes.put_in_working_memory(edit["claim_id"], day, time)
                applied["move into working memory"] += 1
            elif action == "move out to the archive":
                notes.take_out_of_working_memory(edit["claim_id"], day, time)
                applied["move out to the archive"] += 1
            elif action == "attach evidence":
                for ids, supports in ((edit.get("supporting_observation_ids") or (), True),
                                      (edit.get("contradicting_observation_ids") or (),
                                       False)):
                    if ids:
                        notes.record_evidence(edit["claim_id"], ids, day, time, supports)
                applied["attach evidence"] += 1
            else:
                rejected.append(f"an action nobody offered: {action!r}")
        except MemoryIsFull as full:
            refused_for_space.append(f"{edit.get('claim_id')}: {full}")
        except (KeyError, ValueError) as problem:
            rejected.append(f"{action}: {problem}")
    return {"applied": applied, "rejected": rejected,
            "refused_because_working_memory_was_full": refused_for_space}


def write_the_notes_memgpt(notes: Notes, household: FrozenHousehold, day: int, time: int,
                           looks_today: Sequence[LookRecord], client: LLMClient,
                           tell_the_model_everything_it_saw: bool = True,
                           name_the_objects_it_will_be_quizzed_on: bool = False,
                           a_message_tonight: Optional[str] = None) -> Dict[str, Any]:
    """One night. One call, and one retry only if a write was refused for want of room."""
    def ask(extra: Sequence[str]) -> tuple:
        lines = told.the_nightly_prompt(
            A_WORKING_MEMORY_AND_AN_ARCHIVE, household, day,
            writing._what_happened_today(looks_today, household.asked_objects,
                                         tell_the_model_everything_it_saw),
            what_it_can_see_of_its_memory(notes), 240,
            name_the_things_it_is_asked_about=name_the_objects_it_will_be_quizzed_on,
            a_message_tonight=a_message_tonight,
            extra_before_the_instruction=extra)
        text, _ = client.complete(
            [{"role": "system", "content": told.WRITING_SYSTEM},
             {"role": "user", "content": "\n".join(lines)}],
            EDITS_SCHEMA, max_tokens=300 + 260 * 12)
        offered: List[dict] = []
        broke = False
        if text:
            try:
                offered = json.loads(text).get("edits") or []
            except ValueError:
                offered, broke = [], True
        return text, offered, broke

    text, edits, did_not_parse = ask(())
    outcome = _apply(notes, edits, day, time)
    retried = False
    if outcome["refused_because_working_memory_was_full"]:
        # The paper's agent sees the refusal and acts on it. Every call here is stateless,
        # so the refusal is put in front of it once and it gets one more go. A night that
        # still cannot free room loses those writes, and that is counted.
        retried = True
        again = ["YOUR LAST SET OF EDITS WAS PARTLY REFUSED, because your working memory "
                 "was full:"]
        again += [f"- {why}" for why in outcome["refused_because_working_memory_was_full"]]
        again += ["", "Move something out to the archive first, then write what you wanted "
                      "to write. Nothing you move out is lost: you can search for it.", ""]
        text2, edits2, broke2 = ask(again)
        second = _apply(notes, edits2, day, time)
        for key in outcome["applied"]:
            outcome["applied"][key] += second["applied"][key]
        outcome["rejected"] += second["rejected"]
        outcome["still_refused_after_the_retry"] = \
            second["refused_because_working_memory_was_full"]
        did_not_parse = did_not_parse or broke2
        edits = list(edits) + list(edits2)
    notes.written_up_to_day = day
    used = notes.characters_in_working_memory()
    return {"day": day, "model_call_failed": not text,
            "the_completion_did_not_parse": did_not_parse,
            "n_edits_offered": len(edits),
            "it_was_given_a_second_go_after_a_refusal": retried,
            "characters_in_working_memory": used,
            "share_of_working_memory_used": used / WORKING_MEMORY_CHARACTERS,
            "n_notes_in_working_memory": len(notes.working_memory()),
            "n_notes_in_the_archive": len([c for c in notes.claims
                                           if not c.in_working_memory
                                           and c.folded_into is None]),
            **outcome}

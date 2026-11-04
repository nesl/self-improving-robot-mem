"""MemGPT the way MemGPT is built: one block of free text, plus an archive it must search.

Our first MemGPT arm kept numbered notes in both tiers, which is a claim store with a size
limit rather than MemGPT. Their core memory is ONE BLOCK OF FREE TEXT, and the model edits it
by naming a piece of the text and what to put in its place; their archive holds separate
passages that are only ever reached by searching. That shape is what this module adds, and it
is the difference that made the earlier arm an imitation.

The four things their agent can actually do, and the four things here:

  core_memory_append      add a line to the end of the block
  core_memory_replace     name an exact piece of the block and what replaces it. Naming a
                          piece that is not there FAILS, as theirs does - it is a substring
                          match, not a fuzzy one, and the failure is reported to the model.
  archival_memory_insert  put a passage in the archive
  archival_memory_search  ask the archive for passages, a page at a time

Sizes are theirs: `letta/constants.py` puts the persona and human blocks at 20,000 characters
each. A write that would overflow is refused and the refusal is handed back, which is a
faithful reading of the paper's prose - the research agent could not reach an enforcement
point in the code, so that is stated as a reading rather than as a match.

Two departures that remain, both disclosed rather than fixed. Moving something from the block
to the archive is two calls for them (insert, then replace with nothing) and two here as well,
so that one is faithful. But their archive search is interactive across turns - page one, then
the model asks for page two - and a nightly write here is one call, so pages are offered by
asking for more in the same reply rather than in a later turn.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from baselines.patrol.llm import LLMClient
from self_improve import what_the_robot_is_told as told
from self_improve import write_the_notes as writing
from self_improve.frozen_household import FrozenHousehold
from self_improve.looking import LookRecord
from self_improve.memory_notes import (MEMGPT_AS_PUBLISHED, PROVISIONAL,
                                       WORKING_MEMORY_CHARACTERS, Notes)

HOW_MANY_PASSAGES_A_PAGE = 5

WHAT_IT_MAY_DO: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "calls": {
            "type": "array", "maxItems": 8,
            "items": {
                "type": "object",
                # THREE CORRECTIONS ON 2026-09-25, all of them caps of OURS on a method that
                # does not have them, and the second one was breaking the method.
                #
                # `why` FIRST, and 600 rather than 200. Every other schema in this project now
                # reasons before it decides; this one chose the call and wrote the whole of its
                # content before saying why.
                #
                # `the_piece_to_replace` 600 -> 20,000, THE SIZE OF THE BLOCK. This field has to
                # quote the block character for character or `replace_part_of_the_block` refuses
                # the call, and MemGPT puts no limit on it at all - the block's own 20,000 is the
                # only bound their design has. Ours cut the quote at 600, mid-word, and then the
                # exact-match test could not pass. Measured on the three landed cells: 19 of 48
                # replace attempts (40%) were refused because the quoted piece was not in the
                # block, against 29 applied; the audit measured 64 of 69 quotes (92.8%) sitting
                # exactly at 600. So the arm was being stopped from editing its own memory by a
                # number we chose, and every statement about how it maintains its block was
                # measured through that. What is NOT explained by this: the archive held 0
                # passages at every point in all three cells, and archive writes use `text`, not
                # this field.
                #
                # `text` 1200 -> 20,000 for the same reason: 44 of 135 (32.6%) sat exactly at
                # 1,200 and a truncated line entered the block truncated. MemGPT bounds the
                # block, not one write into it.
                "properties": {
                    "why": {"type": "string", "maxLength": 600},
                    "what": {"type": "string",
                             "enum": ["add to the block", "replace part of the block",
                                      "put a passage in the archive",
                                      "search the archive"]},
                    "text": {"type": ["string", "null"], "maxLength": 20000},
                    "the_piece_to_replace": {"type": ["string", "null"], "maxLength": 20000},
                },
                "required": ["why", "what", "text", "the_piece_to_replace"],
                "additionalProperties": False}}},
    "required": ["calls"], "additionalProperties": False}


class TheBlockIsFull(Exception):
    """The block has no room. Handed back to the model, as their overflow is."""


def add_to_the_block(notes: Notes, text: str) -> None:
    piece = (text or "").strip()
    if not piece:
        raise ValueError("nothing to add")
    room = WORKING_MEMORY_CHARACTERS - len(notes.the_block)
    if len(piece) + 1 > room:
        raise TheBlockIsFull(
            f"your block holds {len(notes.the_block)} of {WORKING_MEMORY_CHARACTERS} "
            f"characters, so there is no room for {len(piece)} more. Move something to the "
            f"archive first, or replace a piece of the block instead of adding to it.")
    notes.the_block = (notes.the_block + "\n" + piece).strip()


def replace_part_of_the_block(notes: Notes, old: str, new: str) -> None:
    """Exactly as theirs: the piece named must be in the block, character for character."""
    old = (old or "").strip()
    if not old:
        raise ValueError("no piece named to replace")
    if old not in notes.the_block:
        raise ValueError(
            "that piece is not in your block, character for character, so nothing was "
            "replaced. Quote it exactly as it appears.")
    after = notes.the_block.replace(old, (new or "").strip(), 1)
    if len(after) > WORKING_MEMORY_CHARACTERS:
        raise TheBlockIsFull(
            f"that replacement would take the block to {len(after)} characters, over the "
            f"{WORKING_MEMORY_CHARACTERS} it holds.")
    notes.the_block = after.strip()


def put_a_passage_in_the_archive(notes: Notes, text: str, day: int, time: int) -> None:
    notes.add_claim((text or "").strip() or "(empty)", "not said", day, time, (),
                    PROVISIONAL)


def search_the_archive(notes: Notes, words: str, page: int = 1) -> Dict[str, Any]:
    """A page of passages, and how many pages there are - theirs pages, so this does too."""
    hits = notes.search_the_archive(words or "", how_many=200)
    start = max(0, (page - 1)) * HOW_MANY_PASSAGES_A_PAGE
    shown = hits[start:start + HOW_MANY_PASSAGES_A_PAGE]
    pages = max(1, (len(hits) + HOW_MANY_PASSAGES_A_PAGE - 1) // HOW_MANY_PASSAGES_A_PAGE)
    return {"searched_for": words, "page": page, "of_pages": pages,
            "n_found": len(hits),
            "passages": [c.statement for c in shown]}


def what_it_can_see(notes: Notes, searched_for: str = "") -> str:
    used, cap = len(notes.the_block), WORKING_MEMORY_CHARACTERS
    archived = len([c for c in notes.claims if c.folded_into is None])
    lines = [f"YOUR BLOCK. {used} of {cap} characters used, {cap - used} free. Everything in "
             f"it is in front of you every time you are asked anything.",
             "",
             notes.the_block or "(it is empty)",
             "",
             f"YOUR ARCHIVE holds {archived} passage(s). None of them is shown to you unless "
             f"you search for it."]
    if searched_for:
        found = search_the_archive(notes, searched_for)
        lines += ["", f"You searched your archive for \"{searched_for}\" and it returned "
                      f"page {found['page']} of {found['of_pages']}, "
                      f"{found['n_found']} passage(s) in all:"]
        lines += [f"- {p}" for p in found["passages"]] or ["(nothing)"]
    return "\n".join(lines)


def write_the_notes_memgpt_as_published(notes: Notes, household: FrozenHousehold, day: int,
                                       time: int, looks_today: Sequence[LookRecord],
                                       client: LLMClient,
                                       tell_the_model_everything_it_saw: bool = True,
                                       name_the_objects_it_will_be_quizzed_on: bool = False,
                                       a_message_tonight: Optional[str] = None
                                       ) -> Dict[str, Any]:
    """One night: it edits its block and its archive with their four operations."""
    lines = told.the_nightly_prompt(
        MEMGPT_AS_PUBLISHED, household, day,
        writing._what_happened_today(looks_today, household.asked_objects,
                                     tell_the_model_everything_it_saw),
        what_it_can_see(notes), 1200,
        name_the_things_it_is_asked_about=name_the_objects_it_will_be_quizzed_on,
        a_message_tonight=a_message_tonight)
    text, _ = client.complete(
        [{"role": "system", "content": told.WRITING_SYSTEM},
         {"role": "user", "content": "\n".join(lines)}],
        # 2,600 was ample while one call could carry at most 1,200 + 600 + 200 characters. With
        # the caps corrected to MemGPT's own - the block's 20,000 - a single replace can quote a
        # long passage and supply its replacement, so the token ceiling would become the binding
        # limit and a cut-off reply does not parse, which loses the night. 12,000 tokens is about
        # 46,000 characters: more than one quote plus its replacement can be.
        WHAT_IT_MAY_DO, max_tokens=12000)
    calls: List[dict] = []
    did_not_parse = False
    if text:
        try:
            calls = json.loads(text).get("calls") or []
        except ValueError:
            calls, did_not_parse = [], True
    did = {"add to the block": 0, "replace part of the block": 0,
           "put a passage in the archive": 0, "search the archive": 0}
    refused_for_room, refused_for_wording, searches = [], [], []
    for c in calls:
        what = c.get("what")
        try:
            if what == "add to the block":
                add_to_the_block(notes, c.get("text") or "")
            elif what == "replace part of the block":
                replace_part_of_the_block(notes, c.get("the_piece_to_replace") or "",
                                          c.get("text") or "")
            elif what == "put a passage in the archive":
                put_a_passage_in_the_archive(notes, c.get("text") or "", day, time)
            elif what == "search the archive":
                searches.append(search_the_archive(notes, c.get("text") or ""))
            else:
                continue
            did[what] = did.get(what, 0) + 1
        except TheBlockIsFull as full:
            refused_for_room.append(str(full))
        except ValueError as wrong:
            refused_for_wording.append(str(wrong))
    notes.written_up_to_day = day
    return {"day": day, "model_call_failed": not text,
            "the_completion_did_not_parse": did_not_parse,
            "n_calls_offered": len(calls), "did": did,
            "characters_in_the_block": len(notes.the_block),
            "share_of_the_block_used": len(notes.the_block) / WORKING_MEMORY_CHARACTERS,
            "n_passages_in_the_archive": len([c for c in notes.claims
                                              if c.folded_into is None]),
            "refused_because_the_block_was_full": refused_for_room,
            "refused_because_the_piece_was_not_there": refused_for_wording,
            "searches_it_made": searches}

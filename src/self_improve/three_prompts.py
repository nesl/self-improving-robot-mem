"""The three note-writing prompts, each against an unchanged control, on the ten
varied_v2 pilot homes.

WHAT THE THREE ARE FIXING. Three things were established on 2026-09-24 and each one
says that a prompt the study has been running is not the prompt the study needed:

  1. THE CAREFUL-EDITING FORMAT DOES NOT EXIST AS A DISTINCT CONDITION. The claim
     store can add a claim and mark the old one "set aside for now", keeping both, and
     the model almost never does: in 81 of 118 traced losses it OVERWROTE the old
     claim's wording instead. So "edit single facts" has in practice been "rewrite one
     claim at a time", which is why the two memory formats have been indistinguishable.
     The prompt does name `set aside`, but it frames it as something to do to a claim
     the evidence went against and offers revising the statement first. Variant 1 says
     plainly which of the two to do - add, and set the old one aside without touching
     its wording - and why keeping the loser is worth the space.

  2. NOTHING HAS EVER ASKED THE ROBOT TO DESCRIBE THE PERSON. The prompt asks for
     object places and nudges toward saying WHEN a place holds something; it never asks
     for a routine, a habit or a regime. The study's question is whether the robot
     notices a change in a person's routine. Variant 2 asks for exactly that, at an
     UNCHANGED line budget, so a routine line has to displace an object line. That
     trade is part of what is being measured, not a side effect.

  3. THE ROBOT IS NEVER TOLD ANYTHING HAS CHANGED, in any arm, at any point. Variant 3
     adds one sentence on the first disrupted night and one on the first night back,
     and nothing on the other thirty nights.

WHAT IS HELD IDENTICAL. Everything except the nightly note-writing prompt: the homes,
the searches (3 rooms a question, 8 questions a day, memory-guided), the 8-line read
budget, the answer step and the question sets. Every arm uses `incremental edits`,
because variant 1 is a claim-store instruction and a wholesale summary has no claim to
set aside - `write_the_notes` raises rather than run it as a silent no-op.

HOW THIS MODULE AVOIDS COLLIDING WITH OTHER WORK. `search_driven.run_one_cell` is
shared and is being run by other jobs tonight. Rather than add parameters to it, this
module replaces `search_driven.write_the_notes` IN THIS PROCESS ONLY with a wrapper
that supplies the arm's settings - the same technique, and for the same stated reason,
that `search_driven` itself uses on `write_the_notes._what_happened_today`. The wrapper
counts its calls and records which nights carried a message, and `run_one_arm` asserts
both: that the wrapper was called once per night, and that the message nights are
EXACTLY the two intended ones. A patch that silently did not land, or an arm that was
asked for something and did not get it, is the dominant failure mode in this project
and neither would show up in any accuracy number.

    python -m self_improve.three_prompts --arm rival_beliefs --household hh_s2_t03
"""
from __future__ import annotations

import argparse
import datetime
import functools
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import replace

from baselines.patrol.llm import LLMClient
from self_improve import search_driven as search_driven_module
from self_improve.frozen_household import FrozenHousehold
from self_improve.search_driven import MEMORY_GUIDED, run_one_cell, sanity_assay_on_searches
from self_improve.study_settings import LOCKED
from self_improve.memory_notes import TOLD_IF_IT_WAS_RIGHT
from self_improve.who_lives_here import names_by_resident_id
from self_improve.write_the_notes import the_message_for_tonight, write_the_notes
from self_improve.write_the_notes_told_if_right import write_the_notes_told_if_right

# ---------------------------------------------------------- NO LENGTH LIMIT --
#
# Corrected 2026-09-24, after the first wave was launched and killed. The eight-line
# budget was applied in TWO places and changing one tests nothing:
#
#   at READ time   `memory_notes.what_the_robot_can_read` cut the notes down to eight
#                  lines before they reached a prompt;
#   at WRITE time  the wholesale prompt TOLD the model "your notes must fit in 8 lines"
#                  and, in the same sentence, how many objects it is asked about - so
#                  that arm's notes were composed under the constraint whatever the
#                  reader then did. The claim-store writing prompt never said it.
#
# Eight is fewer lines than there are asked-about objects in nine of the ten homes. It
# is a limit we imposed, cannot justify, and which shapes every number measured under
# it, so removing it is a CORRECTION and not an extra condition. There is deliberately
# no line-count sweep and no variable-budget arm: the question is not what the right
# budget is.
#
# `search_driven` reads the budget off its module-global `LOCKED` at call time, both for
# the answer window and for what it hands the note writer, so the setting is applied by
# replacing that global IN THIS PROCESS ONLY. The shared `study_settings.LOCKED` is not
# touched: other jobs import it and every earlier number must stay reproducible.
NO_LENGTH_LIMIT = replace(LOCKED, read_budget_lines=None)

# The wholesale arm's summary schema capped its ENTIRE memory at 2400 characters, for the
# whole month, with no equivalent cap on the claim store. That is a second, larger half of
# the same limit and it is invisible in the prompt, so it is lifted here together with the
# sentence. 24000 is ten times the old ceiling and ten times more than any summary has
# ever come near, so it is a removal in practice rather than a new budget; it is a number
# rather than None because the schema field requires one.
SUMMARY_CEILING_LIFTED = 24000
# AND the token budget has to clear the character ceiling, or the ceiling is not the
# binding limit and a number nobody measured is. 24,000 characters is about 7,700 tokens
# at the 3.12 characters-per-token measured on this model tonight, so 6,000 tokens - my
# first value - would have cut the summary off at about 18,700 characters, and a summary
# cut off mid-string does not parse, so the night silently falls back to the PREVIOUS
# summary. It has not bitten yet, because the longest summary so far is 1,733 characters -
# but unbounded growth is the HYPOTHESIS for this arm, so it is exactly the arm where it
# would. 9,000 clears 7,700 with margin, and a budget that is never reached costs nothing.
WHOLESALE_MAX_TOKENS = 9000

PILOT_BANKS = pathlib.Path(
    "results/self_improve/varied_homes/ten_homes/banks")
"""The ten varied_v2 pilot homes. NOT the default FROZEN_BANKS: `hh_s2_t03` exists in
both bank sets and they are different households, so the bank path is always explicit.
Older homes blurred the illness step and must not be used - see
results/self_improve/varied_homes/HANDOVER.md."""

PILOT_TEN = ("hh_s2_t03", "hh_s19_t03", "hh_s20_t03", "hh_s32_t03", "hh_s48_t03",
             "hh_s63_t03", "hh_s93_t03", "hh_s109_t03", "hh_s123_t03", "hh_s151_t03")

CONTROL = "control"

ARMS: Dict[str, Dict[str, Any]] = {
    CONTROL: {
        "keep_rival_beliefs": False, "describe_the_person": False,
        "tell_it_the_resident_is_unwell": False,
        "what_it_is": "the nightly note-writing prompt exactly as it stands"},
    # Not one of the three variants. The memory-STYLE contrast, run because with no
    # length limit the wholesale rewrite may stop being a rewrite: with nothing forcing
    # it to drop anything it can simply accumulate, and the two styles could converge on
    # "keep everything". If they do, the distinction between them dissolves and that is
    # an ANSWER to the memory-style question - it would mean the only thing separating
    # them was being made to choose what to lose. Measured directly: how many facts each
    # style holds at days 13, 23 and 31, and whether the rewrite's notes grow
    # monotonically.
    "control_wholesale": {
        "keep_rival_beliefs": False, "describe_the_person": False,
        "tell_it_the_resident_is_unwell": False,
        "how_memory_is_written": "wholesale rewrite",
        "what_it_is": "the unchanged prompt, wholesale rewrite, with the budget "
                      "sentence removed: does a rewrite with no limit still rewrite?"},
    "rival_beliefs": {
        "keep_rival_beliefs": True, "describe_the_person": False,
        "tell_it_the_resident_is_unwell": False,
        "two_edits_per_change": True,
        "what_it_is": "add a claim for the new state and set the old one aside, "
                      "rather than overwriting it"},
    "describe_the_person": {
        "keep_rival_beliefs": False, "describe_the_person": True,
        "tell_it_the_resident_is_unwell": False,
        "what_it_is": "record the household's routine and the people, in the same "
                      "line budget, so routine lines displace object lines"},
    "told_unwell": {
        "keep_rival_beliefs": False, "describe_the_person": False,
        "tell_it_the_resident_is_unwell": True,
        "what_it_is": "one sentence on the first disrupted night and one on the "
                      "first night back, and nothing on any other night"},
    # The SEVENTH variant, added 2026-09-24 on the coordinator's instruction. It is the arm
    # the ACE paper motivates and the only one with outcome feedback, so it is the comparison
    # the write-up most needs - and running it in THIS wave means it shares homes, questions
    # and server conditions with the control instead of being compared across waves.
    #
    # Three things about it that are not true of the other six. It makes TWO model calls a
    # night, a judging call and a writing call, so it costs about twice as much. Its night 1
    # also generates about 16,000 tokens, so it needs the same warm-up. And it is hours old
    # with two faults already found in it, so a failure in its ten cells is expected noise
    # and never a reason to touch the other sixty.
    #
    # It is LAST in this dict, and the launch order follows this dict, so it cannot delay the
    # six variants that are the headline.
    "rival_and_describe": {
        "keep_rival_beliefs": True, "describe_the_person": True,
        "tell_it_the_resident_is_unwell": False,
        "two_edits_per_change": True,
        "what_it_is": "variants 1 and 2 together: a writer asked to describe a "
                      "routine has more reason to keep a superseded belief"},
    "told_if_it_was_right": {
        "keep_rival_beliefs": False, "describe_the_person": False,
        "tell_it_the_resident_is_unwell": False,
        "how_memory_is_written": TOLD_IF_IT_WAS_RIGHT,
        "what_it_is": "the claim store told each night how the day's answers went: a "
                      "judging call, then a writing call that knows which claims helped "
                      "and which misled"},
}

# THE QUIZ LIST IS OFF. Corrected 2026-09-24 on Oliver's decision, after a 42-cell wave
# had run a third of the way through with it ON. That wave is at
# results/self_improve/superseded_the_asked_list_was_in_the_prompt_2026-09-24/.
#
# I did not merely inherit the default, I argued for it: the comment that used to be here
# said enumerating the list was the configuration every earlier note-writing number was
# measured under, and that it made variant 2's null a null in the most hostile
# configuration. That argument was wrong, and `search_driven.extra_note_writing_settings`
# already said so in as many words - False is the PRIMARY configuration there, and it even
# raises rather than run the primary as the comparison by accident. That guard was never
# reached because nothing asked for False.
#
# Why it is not cosmetic. Enumerating the objects the robot will be asked about turns the
# task from describing a house into caching answers to questions it knows are coming. In
# the superseded wave, 94-100% of the claims the arms wrote named an asked-about object. A
# first night written WITHOUT the list produced ten claims of which two named an asked
# object; the rest were the bowl, the cutting board, the dog bowl and the kettle, because
# that is what was in the room it had searched. On the patrol design the same switch moved
# coverage of the quizzed objects from 12.4 of 16.7 down to 2.0.
#
# A deployed robot is not handed the list, so every number measured with it is about writing
# notes when you already know what you will be asked.
NAME_THE_QUIZ_LIST = False


def the_allowance_tonight(household: FrozenHousehold, notes: Any,
                          looks_today: Any, two_edits_per_change: bool) -> int:
    """Tonight's edit allowance, from the shared rule rather than recomputed here.

    `run_one_cell` computes this too, but WITHOUT `two_edits_per_change`, so the two rival
    arms would silently get single revision slots while being asked for two edits per
    change - which is the unevenness this parameter exists to remove. So the wrapper
    recomputes it with the arm's own setting and overrides what `run_one_cell` passed.

    Read through `write_the_notes` so this module cannot drift from the number the writer
    is actually given. The TOKEN budget is left to the shared default, which is now the
    measured `300 + 260 * cap` on this path; passing one from here would mean computing the
    cap twice and the two could disagree.
    """
    from self_improve.write_the_notes import how_many_edits_a_night
    return how_many_edits_a_night(household, notes, looks_today,
                                  two_edits_per_change=two_edits_per_change)


def the_unwell_spells(household: FrozenHousehold) -> List[Tuple[str, int, int]]:
    """Every unwell spell in this household, as (resident, first day, first day back).

    `who_is_unwell_and_when` below returns ONE spell and raises if the days have a gap, which was
    right while every episode had a single illness. The two-illness episodes record both spells
    under the same cause key - `unwell_spell:resident_1` on days 14-23 AND 32-41 - so that
    function raises on them, and the told-it arms would have crashed on their first night rather
    than at launch. Found by reading the bank's own `day_causes` before launching anything.

    This returns them in order and never raises on a gap, because a gap is now meaningful: it is
    the recovery between two illnesses, which is the thing those episodes exist to measure.
    """
    by_resident: Dict[str, List[int]] = {}
    for day, causes in household.header["day_causes"].items():
        for cause in causes:
            if cause.startswith("unwell_spell:"):
                by_resident.setdefault(cause.split(":", 1)[1], []).append(int(day))
    spells: List[Tuple[str, int, int]] = []
    for who, days in by_resident.items():
        run: List[int] = []
        for day in sorted(days):
            if run and day != run[-1] + 1:
                spells.append((who, run[0], run[-1] + 1))
                run = []
            run.append(day)
        if run:
            spells.append((who, run[0], run[-1] + 1))
    if not spells:
        raise ValueError(f"{household.name}: no unwell spell in its day_causes at all")
    return sorted(spells, key=lambda s: s[1])


def who_is_unwell_and_when(household: FrozenHousehold) -> Tuple[str, int, int]:
    """(resident, first disrupted day, first day back), from the bank's own causes.

    Derived from `day_causes` (`unwell_spell:resident_N`), never hard-coded per home,
    and it raises rather than guess: a message delivered on the wrong night would be
    an arm that looks like it complied and did not.
    """
    days: Dict[str, List[int]] = {}
    for day, causes in household.header["day_causes"].items():
        for cause in causes:
            if cause.startswith("unwell_spell:"):
                days.setdefault(cause.split(":", 1)[1], []).append(int(day))
    if len(days) != 1:
        raise ValueError(f"{household.name}: expected exactly one unwell resident, "
                         f"found {sorted(days)}")
    who = next(iter(days))
    spell = sorted(days[who])
    if spell != list(range(spell[0], spell[-1] + 1)):
        raise ValueError(f"{household.name}: the unwell spell has a gap: {spell}")
    return who, spell[0], spell[-1] + 1


def run_one_arm(household: FrozenHousehold, arm: str, client: LLMClient,
                out_dir: pathlib.Path, last_day: int = 31,
                questions_per_day: int = 8, budget: int = 3, seed: int = 0
                ) -> Dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: {sorted(ARMS)}")
    settings = ARMS[arm]
    how = settings.get("how_memory_is_written", "incremental edits")
    who, first_disrupted, first_back = who_is_unwell_and_when(household)
    # The real name, read from the household's own seed rather than guessed. `who_lives_here`
    # raises if the names it rebuilds disagree with the owner names on the objects, so a
    # half-right mapping cannot reach a prompt.
    called_by_name = names_by_resident_id(str(household.bank_path))[who]
    # Restricted to nights this run actually reaches, so a short smoke run asserts
    # what it can rather than failing for not having got to day 24. A full run has
    # last_day 31 and so wants both nights.
    wanted_message_nights = ([d for d in (first_disrupted, first_back) if d <= last_day]
                             if settings["tell_it_the_resident_is_unwell"] else [])

    nights: List[int] = []
    messages_sent: Dict[int, str] = {}
    allowances: List[Tuple[int, int]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    take_the_lock(out_dir, arm, household.name)
    started = time.time()
    beat(out_dir, -1, last_day, arm, household.name, started)

    @functools.wraps(write_the_notes)      # keeps the real signature, which
    def wrapper(notes, household_, day, time_, looks_today, client_,  # run_one_cell
                read_budget_lines=8, **kw):                           # inspects
        # `called` is the person's REAL name and it is the THIRD positional parameter, so the
        # old four-positional call silently passed `first_disrupted` as the name and
        # `first_back` as the first disrupted day - which made the function return None on
        # every night. The told-it arm would not have said the wrong thing, it would have
        # said NOTHING, and become an exact duplicate of the control. The runner's own
        # assertion on the message nights would have refused the cell, which is the only
        # reason this was survivable.
        #
        # The name matters beyond the call: every look this arm has ever seen says "Tomas",
        # so a message saying "resident_1 is unwell" would put an inconsistency in the one
        # sentence we ever tell it.
        message = (the_message_for_tonight(day, who, called_by_name,
                                           first_disrupted, first_back)
                   if settings["tell_it_the_resident_is_unwell"] else None)
        nights.append(day)
        if message:
            messages_sent[day] = message
        beat(out_dir, day, last_day, arm, household.name, started)
        # `run_one_cell` computed tonight's allowance without this arm's
        # `two_edits_per_change`, so recompute and override. The allowance now varies night
        # to night - large on the night a house has to be described, eight on a night that
        # only revises - so it can never be pinned once per cell.
        kw["max_edits"] = the_allowance_tonight(
            household_, notes, looks_today, settings.get("two_edits_per_change", False))
        allowances.append((day, kw["max_edits"]))
        return write_the_notes(
            notes, household_, day, time_, looks_today, client_, read_budget_lines,
            keep_rival_beliefs=settings["keep_rival_beliefs"],
            describe_the_person=settings["describe_the_person"],
            a_message_tonight=message,
            # removed for both formats. It only has an effect on the wholesale one,
            # which is the only writing prompt that ever stated a budget.
            say_the_notes_must_fit_a_budget=False,
            # AND the schema ceiling, which is the bigger half of the asymmetry and is
            # invisible in the prompt. 2400 characters is a cap on the wholesale arm's
            # WHOLE MEMORY for the whole month; the claim store has no such cap because
            # it accumulates. Removing the budget sentence without this would leave a
            # "no length limit" arm that still cannot exceed about 20 lines, ever. See
            # THE_WRITING_PROMPTS_WERE_NOT_THE_SAME.md.
            summary_max_characters=SUMMARY_CEILING_LIFTED,
            wholesale_max_tokens=WHOLESALE_MAX_TOKENS, **kw)

    def refuse(why: str) -> None:
        """Raise, and first make sure the half-written cell cannot look finished.

        `run_one_cell` writes `cell.json` before this function checks anything, so a refused
        cell left a complete-looking `cell.json` on disk - and every "which cells are
        missing" query treats that file as the definition of done. It is renamed, so a
        refused cell is visibly refused rather than quietly counted as a result.
        """
        landed = out_dir / "cell.json"
        if landed.exists():
            landed.rename(out_dir / "cell_REFUSED.json")
        raise AssertionError(why)

    # THE SEVENTH ARM DOES NOT GO THROUGH `write_the_notes`. `run_one_cell` calls
    # `write_the_notes_told_if_right` directly for it, so the wrapper above never sees it -
    # which would mean no heartbeat, no allowance override, and an empty `nights` list that
    # fails this function's own first assertion. So that function is wrapped too, with the
    # same bookkeeping. It takes no prompt-variant arguments, because the seventh arm IS the
    # unchanged prompt plus outcome feedback.
    @functools.wraps(write_the_notes_told_if_right)
    def told_wrapper(notes, household_, day, time_, looks_today, answers_today, client_,
                     max_edits=None, **kw):
        nights.append(day)
        beat(out_dir, day, last_day, arm, household.name, started)
        max_edits = the_allowance_tonight(
            household_, notes, looks_today, settings.get("two_edits_per_change", False))
        allowances.append((day, max_edits))
        return write_the_notes_told_if_right(
            notes, household_, day, time_, looks_today, answers_today, client_,
            max_edits=max_edits, **kw)

    was = search_driven_module.write_the_notes
    was_told = search_driven_module.write_the_notes_told_if_right
    was_locked = search_driven_module.LOCKED
    search_driven_module.write_the_notes = wrapper
    search_driven_module.write_the_notes_told_if_right = told_wrapper
    search_driven_module.LOCKED = NO_LENGTH_LIMIT
    if search_driven_module.LOCKED.read_budget_lines is not None:
        raise AssertionError("the no-length-limit setting did not land on "
                            "search_driven.LOCKED; refusing to run under the cap")
    timed = TimedClient(client, out_dir)
    try:
        result = run_one_cell(
            household, how, MEMORY_GUIDED, timed, out_dir,
            last_day=last_day, questions_per_day=questions_per_day, budget=budget,
            seed=seed, tell_it_what_it_ruled_out=True,
            name_the_objects_it_will_be_quizzed_on=NAME_THE_QUIZ_LIST)
    finally:
        search_driven_module.write_the_notes = was
        search_driven_module.write_the_notes_told_if_right = was_told
        search_driven_module.LOCKED = was_locked

    # ---- the assertions that make this arm believable at all.
    # THE CHECK THAT IS THE WHOLE LESSON OF THE SUPERSEDED WAVE. The cell's own header is
    # read back off disk and the quiz-list flag must be what this module asked for. A whole
    # wave ran a third of the way through with it on the wrong setting because nothing ever
    # looked at the artifact - only at the code that was supposed to produce it.
    header = json.loads((out_dir / "searches.jsonl").open().readline())
    if header.get("name_the_objects_it_will_be_quizzed_on") != NAME_THE_QUIZ_LIST:
        refuse(
            f"{household.name}/{arm}: this cell's own searches.jsonl header says "
            f"name_the_objects_it_will_be_quizzed_on="
            f"{header.get('name_the_objects_it_will_be_quizzed_on')!r} but this module "
            f"asked for {NAME_THE_QUIZ_LIST!r}. Refusing to report it.")
    allowed = {n.get("how_many_edits_it_was_allowed") for n in result["nightly"]
               if n.get("how_many_edits_it_was_allowed") is not None}
    if how == "incremental edits":
        # The allowance is now computed per night, so the check is on its SHAPE: the first
        # night that saw anything has to be big enough to describe a house, and a typical
        # later night should fall back to the revision slots. A cell where every night got
        # the same number means the nightly rule did not reach the writer.
        mine = dict(allowances)
        early = [v for d, v in sorted(mine.items()) if d <= 1 and v > 0]
        later = [v for d, v in sorted(mine.items()) if d >= 5]
        if len(allowed) <= 1 and len(mine) > 3:
            refuse(
                f"{household.name}/{arm}: every night was allowed the same "
                f"{sorted(allowed)} edits, so the nightly allowance did not reach the "
                f"writer - it is meant to be large on the first night and small after.")
        if early and max(early) < 20:
            refuse(
                f"{household.name}/{arm}: the first night was allowed only "
                f"{max(early)} edits, but a night that has to describe a whole house sees "
                f"45 to 57 distinct objects. The allowance is set against the wrong "
                f"quantity again.")
        if later and min(later) > 40:
            refuse(
                f"{household.name}/{arm}: a later night was allowed {min(later)} edits "
                f"and later nights typically have 0 to 2 new objects, so the allowance is "
                f"not tracking what the night saw.")
    # A completion that came back, did not parse and wrote nothing is the failure that no
    # accuracy number would show. It is now reported per night, so it is checked.
    # The WHOLESALE arm has its own silent night: a call that fails or does not parse makes
    # `rewrite_the_notes_wholesale` keep the PREVIOUS summary and set `model_call_failed`,
    # so the night looks like "it chose not to change anything" in every other field.
    if how == "wholesale rewrite":
        failed = [n["day"] for n in result["nightly"] if n.get("model_call_failed")]
        if failed:
            refuse(
                f"{household.name}/{arm}: the summary call failed on nights {failed}, so "
                f"those nights kept the previous summary unchanged. Usually a token budget "
                f"below the character ceiling. Refusing to report the cell.")
    did_not_parse = [n["day"] for n in result["nightly"]
                     if n.get("the_completion_did_not_parse")]
    if did_not_parse:
        refuse(
            f"{household.name}/{arm}: the completion did not parse on nights "
            f"{did_not_parse}, so those nights wrote nothing. Almost always a token budget "
            f"too small for the allowance. Refusing to report the cell.")

    # A NIGHT THAT SAW SOMETHING AND WROTE NOTHING IS A FAILURE, NOT A ZERO. The shared
    # token budget truncated the edits mid-JSON, which leaves `edits` empty while
    # `model_call_failed` stays False - a whole night lost with nothing in the report
    # saying so. Refuse the cell rather than average over empty nights.
    if how == "incremental edits":
        silent = [n["day"] for n in result["nightly"]
                  if n.get("the_look_saw_something_it_is_asked_about")
                  and not any((n.get("applied") or {}).values())]
        if silent:
            refuse(
                f"{household.name}/{arm}: nights {silent} saw an asked-about object and "
                f"applied NO edit. Either the completion was truncated mid-JSON (check "
                f"`the_completion_did_not_parse`) or every edit was rejected. Refusing to "
                f"report a cell with silently empty nights.")

    told_a_budget = [n["day"] for n in result["nightly"]
                     if n.get("the_prompt_told_it_to_fit_a_line_budget")]
    if told_a_budget:
        refuse(
            f"{household.name}/{arm}: {len(told_a_budget)} nights were still told to "
            f"fit a line budget, so the no-length-limit correction did not land.")
    bit = [r for r in result["searches"] if r.get("the_read_budget_bit")]
    if bit:
        refuse(
            f"{household.name}/{arm}: the read budget bit on {len(bit)} questions, so "
            f"the notes were still being truncated at answer time.")
    # The write-time edits-per-night cap (EDITS_SCHEMA's maxItems, separate from the
    # read-time line budget checked above). Found 2026-09-24: keep_rival_beliefs asks
    # for two edits per moved object where the control needs one, so the two arms can
    # hit the same literal cap at different rates for covering the identical set of
    # facts - which would make a difference in claims written or claim-store coverage
    # partly a budget artefact rather than a content one. Reported here so it travels
    # with every number this wave publishes, not read off separately.
    cap_hit_nights = [n["day"] for n in result["nightly"] if n.get("the_edits_cap_bit")]
    if nights != list(range(last_day + 1)):
        refuse(
            f"{household.name}/{arm}: the arm's note-writing wrapper was called on "
            f"nights {nights[:5]}...{nights[-3:]} ({len(nights)} calls), not once on "
            f"each of days 0..{last_day}. The prompt settings this arm claims were "
            f"therefore NOT applied on every night. Refusing to report it.")
    if sorted(messages_sent) != wanted_message_nights:
        refuse(
            f"{household.name}/{arm}: a message was carried on nights "
            f"{sorted(messages_sent)}, wanted exactly {wanted_message_nights}.")

    compliance = {
        "arm": arm,
        "what_it_is": settings["what_it_is"],
        "household": household.name,
        "bank": str(household.bank_path),
        "how_memory_is_written": how,
        "sensing_arm": MEMORY_GUIDED,
        "read_budget_lines": NO_LENGTH_LIMIT.read_budget_lines,
        "name_the_objects_it_will_be_quizzed_on_in_the_header":
            header.get("name_the_objects_it_will_be_quizzed_on"),
        "how_many_edits_a_night_it_was_allowed": sorted(allowed),
        "the_allowance_night_by_night": {str(d): v for d, v in sorted(allowances)},
        "two_edits_per_change": settings.get("two_edits_per_change", False),
        "n_nights_whose_completion_did_not_parse": len(did_not_parse),
        # The seventh arm's own three numbers, which come before any outcome from it: how
        # often the judging step fell back to the rule instead of the model, how often a
        # judgement named a claim that does not exist, and how many joins were the model's
        # rather than the code's. An arm whose feedback is mostly rule-derived is not an
        # arm with outcome feedback.
        "told_if_it_was_right": ({
            "n_nights": sum(1 for n in result["nightly"]
                            if n.get("judgement") is not None
                            or n.get("the_verdicts_were_counted_by_rule_not_by_the_model")
                            is not None),
            "n_nights_counted_by_rule_not_by_the_model": sum(
                1 for n in result["nightly"]
                if (n.get("judgement") or n).get(
                    "the_verdicts_were_counted_by_rule_not_by_the_model")),
            "n_judgements_naming_a_claim_that_does_not_exist": sum(
                (n.get("judgement") or n).get("n_named_a_claim_that_does_not_exist") or 0
                for n in result["nightly"]),
            "n_joins_by_rule_not_by_the_model": sum(
                len(n.get("joined_by_rule_not_by_the_model") or ())
                if isinstance(n.get("joined_by_rule_not_by_the_model"), list)
                else (1 if n.get("joined_by_rule_not_by_the_model") else 0)
                for n in result["nightly"]),
        } if how == TOLD_IF_IT_WAS_RIGHT else None),
        "n_model_calls_timed": timed.n_calls,
        "the_writer_was_told_to_fit_a_budget": False,
        "n_nights_told_to_fit_a_budget": len(told_a_budget),
        "n_questions_where_the_read_budget_bit": len(bit),
        "n_nights_the_edits_cap_bit": len(cap_hit_nights),
        "share_of_nights_the_edits_cap_bit":
            (len(cap_hit_nights) / len(nights)) if nights else None,
        "nights_the_edits_cap_bit": cap_hit_nights,
        "name_the_objects_it_will_be_quizzed_on": NAME_THE_QUIZ_LIST,
        "keep_rival_beliefs": settings["keep_rival_beliefs"],
        "describe_the_person": settings["describe_the_person"],
        "tell_it_the_resident_is_unwell": settings["tell_it_the_resident_is_unwell"],
        "the_unwell_resident": who,
        "first_disrupted_day": first_disrupted,
        "first_day_back": first_back,
        "n_nights_the_writer_ran": len(nights),
        "nights_that_carried_a_message": sorted(messages_sent),
        "the_messages": {str(d): m for d, m in sorted(messages_sent.items())},
        "the_message_appeared_on_exactly_the_intended_nights":
            sorted(messages_sent) == wanted_message_nights,
    }
    (out_dir / "arm.json").write_text(json.dumps(compliance, indent=1))
    result["three_prompts"] = compliance
    (out_dir / "cell.json").write_text(json.dumps(result, indent=1))
    return result


class TimedClient:
    """The model client, with the wall-clock cost of every call recorded.

    The read window is unlimited by Oliver's decision, and he chose to report the cost
    rather than cap it - so the cost has to be a measured number and not an assumption.
    Every call appends one line to `call_times.jsonl` in the cell directory: how long it
    took, how many characters of prompt it carried, and whether it was served from the
    cache. A cell is single-threaded, so the calls are in question order and can be joined
    to `searches.jsonl`, which already records `n_lines_of_notes_available` per question.

    A cached call is recorded and flagged rather than dropped, because a mean over a mix of
    cache hits and real calls is not the cost of anything. `seconds_per_question` must be
    computed over the calls that actually reached the server.
    """

    def __init__(self, inner: Any, out_dir: pathlib.Path) -> None:
        self.inner = inner
        self.path = out_dir / "call_times.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_calls = 0

    @property
    def stats(self):
        return self.inner.stats

    def complete(self, messages, schema, max_tokens: int = 400):
        cached_before = self.inner.stats.get("cached", 0)
        started = time.time()
        got = self.inner.complete(messages, schema, max_tokens)
        took = time.time() - started
        self.n_calls += 1
        was_cached = self.inner.stats.get("cached", 0) > cached_before
        with self.path.open("a") as fh:
            fh.write(json.dumps({
                "n": self.n_calls,
                "at": round(started, 3),
                "seconds": round(took, 4),
                "n_prompt_characters": sum(len(m.get("content") or "") for m in messages),
                "served_from_the_cache": was_cached,
                "max_tokens": max_tokens,
            }) + "\n")
        return got


def cell_dir(out: pathlib.Path, arm: str, household_name: str) -> pathlib.Path:
    return out / "cells" / arm / household_name


def take_the_lock(out_dir: pathlib.Path, arm: str, household: str) -> None:
    """Refuse to start if a LIVE process is already writing this cell.

    Two processes appended to one cell's `looks.jsonl` and overwrote one `notes.json` on
    2026-09-24, because the launcher recorded a wrapping subshell's pid and killing it
    orphaned python. The launcher is fixed, but the guard belongs here rather than there:
    a cell must be unable to have two writers whatever launches it.

    A lock whose process is gone is stale and is taken over, with the takeover recorded,
    because refusing on a stale lock would need a human every time a cell is stopped.
    """
    lock = out_dir / "RUNNING.lock"
    if lock.exists():
        try:
            held = json.loads(lock.read_text())
        except ValueError:
            held = {}
        other = held.get("pid")
        if isinstance(other, int) and other != os.getpid():
            try:
                os.kill(other, 0)
                alive = True
            except (ProcessLookupError, PermissionError) as why:
                alive = isinstance(why, PermissionError)
            if alive:
                raise SystemExit(
                    f"{household}/{arm}: pid {other} is already writing {out_dir} "
                    f"(since {held.get('at')}). Refusing to be a second writer - that "
                    f"corrupts the look stream and the notes. Stop it first.")
    lock.write_text(json.dumps({
        "pid": os.getpid(), "sid": os.getsid(0), "arm": arm, "household": household,
        "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "took_over_a_stale_lock": lock.exists()}, indent=1))


def beat(out_dir: pathlib.Path, day: int, last_day: int, arm: str, household: str,
         started: float) -> None:
    """One timestamped line per completed night, in the cell and in one shared file.

    Added after a wave was stopped and the logs were the ONLY thing that showed where
    it had got to. A per-day line in a per-cell file is not enough on its own: nothing
    was watching it. `heartbeats.jsonl` at the root is one file a reader can open to see
    every cell at once, and `watch_the_heartbeats.py` turns it into a stall report.
    """
    row = {"at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "epoch": time.time(), "pid": os.getpid(), "sid": os.getsid(0),
           "arm": arm, "household": household, "day": day, "last_day": last_day,
           "seconds_so_far": round(time.time() - started, 1)}
    line = json.dumps(row) + "\n"
    with (out_dir / "heartbeat.jsonl").open("a") as fh:
        fh.write(line)
        fh.flush()
    shared = out_dir.parent.parent.parent / "heartbeats.jsonl"
    with shared.open("a") as fh:          # append-only; one line is one atomic write
        fh.write(line)
        fh.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--household", required=True)
    parser.add_argument("--banks", type=pathlib.Path, default=PILOT_BANKS)
    parser.add_argument("--last-day", type=int, default=31)
    parser.add_argument("--questions-per-day", type=int, default=8)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("results/self_improve/three_prompts"))
    parser.add_argument("--cache", type=pathlib.Path,
                        default=pathlib.Path("llm_prior_cache/self_improve"))
    args = parser.parse_args(argv)

    household = FrozenHousehold(args.banks / f"{args.household}.jsonl")
    client = LLMClient(args.cache)
    out_dir = cell_dir(args.out, args.arm, household.name)
    started = time.time()
    result = run_one_arm(household, args.arm, client, out_dir, args.last_day,
                         args.questions_per_day, args.budget, args.seed)
    assay = sanity_assay_on_searches(result)
    (out_dir / "sanity_assay.json").write_text(json.dumps(assay, indent=1))
    print()
    print(json.dumps(result["three_prompts"], indent=1))
    print(json.dumps(result["summary"], indent=1))
    print("sanity assay:", json.dumps(assay, indent=1))
    for concern in assay["concerns"]:
        print("   ASSAY CONCERN:", concern)
    print(f"model calls {client.stats['calls']}, cache hits {client.stats['cached']}, "
          f"{time.time() - started:.0f}s wall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

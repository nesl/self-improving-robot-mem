"""The overnight wave: five memory methods on the ten homes, with every gate kept.

WHY THIS EXISTS RATHER THAN `three_prompts`. That module injected prompt variants by
replacing `search_driven.write_the_notes` in-process. The prompts are now consolidated in
`what_the_robot_is_told`, so there is nothing to inject: an arm here is a `how_memory_is_written`
or a `sensing_arm` and nothing else. So this calls `search_driven.run_one_cell` directly and
keeps only the parts of the old runner that were about catching silent failure.

THE ARMS, in the order they matter. If time runs short the last three go, never the first two.

  1. the log and notes about the routine   ten homes. The robot reads its own record when
                                          asked, so its notes may not just copy places.
  2. claim store told if it was right      ten homes. ACE. TWO model calls a night.
  3. incremental edits                     ten homes. The plain claim store both are forks of.
  4. last seen, no model                   ten homes. Walk back through the rooms it has
                                          been seen in, newest first. Calls NO model at all,
                                          so it is nearly free and needs no warm-up. Renamed
                                          from `newest sighting, no model` on 2026-09-25 when
                                          it was given the whole trail rather than one room -
                                          the old name is not reused, so no directory on disk
                                          holds two different rules under one name.
  5. prior only, no notes                  ten homes. Prices the prior. Without it we cannot
                                          say any of the above is the memory.
  6. wholesale rewrite                     ONE home. The published weak family and ACE's own
                                          baseline: it earns a reference, not ten cells.

FIVE GATES, every one of them because a wave was lost to the failure it catches:

  the header off disk     the cell's own `searches.jsonl` must say the asked-object list was
                          NOT in the prompt. A 42-cell wave ran a third of the way through
                          with it on because nothing read the artifact, only the code.
  a silent night          a night that saw an asked-about object and applied no edit is a
                          refusal. Night 0 has no looks and is out of scope.
  a completion that did   text came back, did not parse, the night wrote nothing, and
  not parse               `model_call_failed` stayed False. Refused.
  the read budget         if it truncated the notes on any question the arm was not run with
                          unlimited memory, whatever this module intended.
  a held cell is          `run_one_cell` writes `cell.json` before any gate runs, and every
  RENAMED                 "which cells are missing" query treats that file as done. A held cell's
                          file becomes `cell_HELD_FOR_REVIEW.json`, with the reason in `why.txt`
                          beside it - NOT `cell_REFUSED.json`, which is what this line said until
                          2026-09-28 and is the name `three_prompts.py` uses for its own gate.

    python -m self_improve.overnight_wave --arm "incremental edits" --household hh_s2_t03
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from baselines.patrol.llm import LLMClient, make_client
from self_improve import search_driven as sd
from self_improve.frozen_household import FrozenHousehold
from self_improve.memory_notes import (A_PROFILE_AND_TOLD, A_PROFILE_OF_EACH_PERSON,
                                      TOLD_AND_ASKED_WHAT_CHANGES,
                                      ACE_AS_PUBLISHED, MEMGPT_AS_PUBLISHED, THE_LOG_ONLY,
                                      THE_LOG_AND_THE_ROUTINE_DERIVED_ALLOWANCE,
                                      THE_LOG_AND_THE_ROUTINE_EIGHT,
                                      THE_LOG_AND_THE_ROUTINE_SIXTEEN,
                                      TOLD_THE_NIGHT_BEFORE, TOLD_ON_THE_FIRST_NIGHT)
from self_improve.study_settings import LOCKED
from self_improve.three_prompts import PILOT_BANKS, PILOT_TEN, TimedClient, beat, take_the_lock

# No memory length limit, in writing or in reading. `search_driven` reads the budget off its
# module-global LOCKED at call time, so it is replaced in THIS PROCESS ONLY; the shared
# settings object is untouched because other jobs import it.
NO_LENGTH_LIMIT = replace(LOCKED, read_budget_lines=None)

# The asked-object list is OFF. Handing the writer the list of things it will be asked about
# turns the task from describing a home into caching answers to questions it knows are coming:
# measured, 94-100% of claims named an asked-about object with the list, 27% without it.
NAME_THE_QUIZ_LIST = False

MEMORY_GUIDED = sd.MEMORY_GUIDED

# arm name -> (how_memory_is_written, sensing_arm, how many homes)
ARMS: Dict[str, Tuple[str, str, int]] = {
    "the log and notes about the routine": ("the log and notes about the routine",
                                           MEMORY_GUIDED, 10),
    "claim store told if it was right": ("claim store told if it was right",
                                        MEMORY_GUIDED, 10),
    # MemGPT. A 1,200-character working memory in every prompt, an unbounded archive reached
    # only by searching, and the model moves notes between them. A write that would overflow
    # the working memory is REFUSED and the refusal handed back, which is the paper's
    # mechanism rather than an error. One call a night, plus a second only after a refusal.
    "a small working memory and an archive": ("a small working memory and an archive",
                                             MEMORY_GUIDED, 10),
    "incremental edits": ("incremental edits", MEMORY_GUIDED, 10),
    # The two published designs, rebuilt from their own papers and their own shipped code
    # rather than from our reading of them: three rounds of reflection only after a wrong
    # answer, embedding grouping with a model-written merge, ADD-only application (ACE); a
    # fixed-size block of free text edited by exact substring replacement with a paginated
    # archive behind it (MemGPT). They had run for FIVE DAYS as smoke tests and were never
    # in this list, so no launcher could queue them and nothing advanced past day 4 - which
    # is exactly the fault of adding a run to the artifact before adding it to the runner.
    # The ceiling test. Our arm on the control's derived allowance, nothing else changed,
    # because the flat 16 binds on 27% of its nights at 24 questions a day. Three homes: it
    # only has to be paired against the cells already run on the same three.
    "ours, sixteen notes a night": (THE_LOG_AND_THE_ROUTINE_SIXTEEN, MEMORY_GUIDED, 3),
    # 2026-09-26. Our arm plus a profile of each person that it keeps up to date, given their
    # names and what they do on the first night and nothing else.
    "ours, a profile of each person": (A_PROFILE_OF_EACH_PERSON, MEMORY_GUIDED, 3),
    # The same arm, plus the one sentence on the first changed day. The pair differ only in
    # whether the sentence has a profile to live in.
    "ours, a profile and told": (A_PROFILE_AND_TOLD, MEMORY_GUIDED, 3),
    # Told the night before AND made to write down what it expects, before it has seen any of it.
    "ours, told and asked what changes": (TOLD_AND_ASKED_WHAT_CHANGES, MEMORY_GUIDED, 3),
    "ours, eight notes a night": (THE_LOG_AND_THE_ROUTINE_EIGHT, MEMORY_GUIDED, 3),
    "ours, allowance derived": (THE_LOG_AND_THE_ROUTINE_DERIVED_ALLOWANCE, MEMORY_GUIDED, 3),
    # The two told arms. Nobody in this study has ever been told the household changed, and the
    # measurement that made these worth running is in memory_notes: 428 notes described where
    # things went while somebody was ill and none of them mentioned the illness. These hand the
    # robot the cause in one sentence and differ only in the night it arrives - before the change
    # or on its first night - so the pair separates "can it use a stated cause" from "does it
    # help to hear it before the evidence".
    "ours, told the night before": (TOLD_THE_NIGHT_BEFORE, MEMORY_GUIDED, 3),
    "ours, told on the first night": (TOLD_ON_THE_FIRST_NIGHT, MEMORY_GUIDED, 3),
    "ACE as published": (ACE_AS_PUBLISHED, MEMORY_GUIDED, 3),
    "MemGPT as published": (MEMGPT_AS_PUBLISHED, MEMORY_GUIDED, 3),
    # The log without the summaries: same search-time prompt as the control, empty notes block,
    # no nightly write. Separates what the raw record is worth from what the notes add.
    "log only, no notes": (THE_LOG_ONLY, MEMORY_GUIDED, 3),
    # 2026-11-02: the claim store plus one block at search and answer time.
    "claim store plus the log": (sd.CLAIM_STORE_PLUS_THE_LOG, MEMORY_GUIDED, 10),
    "claim store plus today's sightings": (sd.CLAIM_STORE_PLUS_TODAY, MEMORY_GUIDED, 10),
    "last seen, no model": ("incremental edits", sd.LAST_SEEN, 10),
    # 2026-11-02: last-seen with today's sightings merged in only at night.
    "last seen, updated each night": ("incremental edits", sd.LAST_SEEN_NIGHTLY, 10),
    "prior only, no notes": ("incremental edits", sd.PRIOR_ONLY, 10),
    "wholesale rewrite": ("wholesale rewrite", MEMORY_GUIDED, 1),
}
THE_ORDER = tuple(ARMS)

# ON HOLD, and the hold is in the code rather than in my head. MemGPT's five-day live check
# on hh_s109_t03 has to finish and be declared passed before its ten cells are queued: three
# new arms tonight have had a fault in their first hour. `cells_to_launch` refuses to emit a
# held arm, so a launcher cannot queue it by accident.
# Released 2026-09-25 after the coordinator read its notes and its sixteen nights of records.
# Nothing is held now; the dict stays so a future arm can be held the same way.
ON_HOLD: Dict[str, str] = {}

# WHAT MEMGPT'S NUMBERS DO AND DO NOT SUPPORT. Over sixteen nights on one home the overflow
# refusal fired ZERO times: the model keeps its working memory 88-96% full and then revises
# what is already there rather than adding - 47 revisions against 10 new notes - so it never
# asks for room it does not have. That is MemGPT behaving as intended, and it means the
# mechanism that makes this arm MemGPT rather than a small claim store with a search has never
# been exercised.
#
# CORRECTED 2026-09-25. The paragraph above said the overflow refusal had fired zero times, and
# that was an absent-field zero twice over. The count was read from
# `n_writes_refused_for_want_of_room`, a key `write_the_notes_memgpt` has never written - it
# writes `refused_because_working_memory_was_full`, a list - and it was read over sixteen nights
# of one home rather than the 320 that have now landed. Read from the right key over all ten
# homes, the refusal fired TWICE:
#
#   hh_s19_t03  night 3  - 1,080 of 1,200 characters held, a 199-character note would not fit
#   hh_s151_t03 night 20 - 1,160 of 1,200 characters held, a 154-character note would not fit
#
# Both were given the second go the method allows and neither was still refused after it, so the
# one-retry limit was reached twice and was enough twice. Two nights in 320 is still too few to
# support a claim about eviction, but "it never fires" is no longer one of the things we know.
#
# So this arm supports "does a 1,200-character working memory plus an archive search help", and
# on eviction it says only that the pressure is rare at this size and that one retry cleared it
# both times it appeared.
MEMGPT_SUPPORTS = (
    "does a 1,200-character working memory plus an archive search help. On eviction it says "
    "only this: over 320 nights on ten homes the overflow refusal fired twice, on hh_s19_t03 "
    "night 3 and hh_s151_t03 night 20, and the single retry the method allows cleared it both "
    "times. Two nights is too few to support a claim about whether eviction helps, and the "
    "earlier claim that it never fired was read from a key this arm does not write")


def cells_to_launch(arms=None, homes=PILOT_TEN):
    """(arm, household) pairs in priority order, with held arms refused.

    `last seen, no model` calls no model at all, so it is emitted first when it is in
    the list: it fills a gap for free while something else warms.
    """
    out = []
    for arm in (arms if arms is not None else THE_ORDER):
        if arm in ON_HOLD:
            raise RuntimeError(f"{arm!r} is ON HOLD: {ON_HOLD[arm]}. Refusing to queue it.")
        _how, _sensing, n_homes = ARMS[arm]
        for household in list(homes)[:n_homes]:
            out.append((arm, household))
    return out

# `last seen, no model` calls no model at all, so it writes no notes and has no
# night-one generation to warm. Everything else does.
NEEDS_WARMING = tuple(a for a in ARMS if a not in ("last seen, no model",
                                                   "last seen, updated each night"))


def cell_dir(out: pathlib.Path, arm: str, household: str) -> pathlib.Path:
    return out / "cells" / arm.replace(" ", "_").replace(",", "") / household


def run_one_arm(household: FrozenHousehold, arm: str, client: LLMClient,
                out_dir: pathlib.Path, last_day: int = 31,
                questions_per_day: int = 8, budget: int = 3, seed: int = 0
                ) -> Dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: {list(ARMS)}")
    how, sensing, _homes = ARMS[arm]
    out_dir.mkdir(parents=True, exist_ok=True)
    take_the_lock(out_dir, arm, household.name)
    crashed = out_dir / "CRASHED.txt"   # from an earlier attempt; this one has just started
    if crashed.exists():
        crashed.rename(out_dir / "CRASHED_on_an_earlier_attempt.txt")
    started = time.time()
    beat(out_dir, -1, last_day, arm, household.name, started)
    # THE DAY-RENDERER GUARD NEEDS NOTHING FROM THIS MODULE ANY MORE.
    #
    # It used to. `run_one_cell` asserts the note-writing prompt went through the
    # search-driven day renderer exactly once a night, and the record-reading arm built its
    # day with its own function that never called the counted one - while that function's
    # docstring claimed it did. So that arm could not start: the assert fired on day 0, and I
    # registered the render here rather than weakening the assert.
    #
    # Both are fixed upstream now. `the_day_this_arm_sees` calls
    # `write_the_notes._what_happened_today`, so the render is counted AND the arm sees
    # exactly what its control sees, absences included. Verified before removing the wrapper:
    # identical text and the counter incremented once, on looks with and without absences.
    #
    # Leaving the wrapper in would have been worse than useless: the upstream rename gave the
    # function a second required argument, so the wrapper's one-argument signature would have
    # raised - and had it matched, it would have counted the render TWICE and failed the very
    # assert it was written to satisfy.


    timed = TimedClient(client, out_dir)
    was_locked = sd.LOCKED
    sd.LOCKED = NO_LENGTH_LIMIT
    if sd.LOCKED.read_budget_lines is not None:
        raise AssertionError("the no-length-limit setting did not land on search_driven.LOCKED")
    try:
        result = sd.run_one_cell(
            household, how, sensing, timed, out_dir, last_day=last_day,
            questions_per_day=questions_per_day, budget=budget, seed=seed,
            tell_it_what_it_ruled_out=True,
            name_the_objects_it_will_be_quizzed_on=NAME_THE_QUIZ_LIST)
    except BaseException as crash:
        # A CRASH MUST LEAVE A MARK A DIRECTORY LISTING SHOWS, and must release the lock.
        # `a small working memory and an archive`/hh_s19_t03 died on day 3 and sat for
        # thirteen minutes looking like a running cell, because the only record of the crash
        # was a traceback at the bottom of a log nobody was tailing.
        (out_dir / "CRASHED.txt").write_text(
            f"CRASHED {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"{arm} / {household.name}\n"
            f"{type(crash).__name__}: {crash}\n\n"
            f"This cell did NOT run to completion - it is not a gate holding it back. Its\n"
            f"partial files are here. The traceback is in this wave's log for the cell.\n")
        lock = out_dir / "RUNNING.lock"
        if lock.exists():
            lock.unlink()
        raise
    finally:
        sd.LOCKED = was_locked

    def refuse(why: str) -> None:
        """Hold the cell back for review, and say so in a way a directory listing can show.

        `cell_REFUSED.json` beside a live-looking `RUNNING.lock` read like a crash, and it
        misled me for hours: a cell held back by a gate is not the same thing as a cell that
        died, and nothing in the directory distinguished them.
        """
        landed = out_dir / "cell.json"
        if landed.exists():
            landed.rename(out_dir / "cell_HELD_FOR_REVIEW.json")
        (out_dir / "why.txt").write_text(
            f"HELD FOR REVIEW - this cell ran to completion and a gate held it back.\n"
            f"It did not crash. Its files are intact and usable if the gate is overruled.\n\n"
            f"{why}\n")
        lock = out_dir / "RUNNING.lock"
        if lock.exists():
            lock.unlink()
        raise AssertionError(why)

    header = json.loads((out_dir / "searches.jsonl").open().readline())
    # READ BACK OFF DISK, for the same reason the quiz-list flag is: the wave at 24 questions
    # a day and the wave at 8 answer different questions, and a cell that quietly ran at the
    # wrong number would be pooled with the others and nothing would say so. The number is in
    # the cell's own header, in arm.json, and in the first line of every report.
    if int(header.get("questions_per_day", -1)) != int(questions_per_day):
        refuse(f"{household.name}/{arm}: the cell's own header says it ran "
               f"{header.get('questions_per_day')!r} questions a day, wanted "
               f"{questions_per_day}")
    if header.get("name_the_objects_it_will_be_quizzed_on") != NAME_THE_QUIZ_LIST:
        refuse(f"{household.name}/{arm}: the cell's own header says the asked-object list "
               f"was {header.get('name_the_objects_it_will_be_quizzed_on')!r}, wanted "
               f"{NAME_THE_QUIZ_LIST!r}")
    bit = [r for r in result["searches"] if r.get("the_read_budget_bit")]
    if bit:
        refuse(f"{household.name}/{arm}: the read budget truncated the notes on {len(bit)} "
               f"questions, so this arm did not run with unlimited memory")
    nightly = result.get("nightly") or []

    # THE CALL IS CHECKED FIRST, so one event is not reported as two. A night whose call
    # failed wrote nothing BECAUSE it got nothing back; reporting that separately as "saw
    # something and applied no edit" made one server failure look like two faults, which is
    # how `control/hh_s63_t03` came to carry both labels for the same night 16.
    failed = sorted(n["day"] for n in nightly if n.get("model_call_failed"))

    # A FAILED CALL NO LONGER HOLDS A CELL BACK. Changed 2026-09-25: the old threshold was
    # "any failed call", which is stricter than the rerun floor this project already reports
    # against, and it cost three of ten cells of the arm we most needed. A failed call is
    # server-side: the notes carry forward unchanged, so the cost is that night's
    # observations never enter the memory. Measured across the four affected cells, that
    # cost is a night's observations never entering the memory. The cause of every failed
    # call in this wave was a socket timeout in the shared client, now fixed, and the five
    # affected cells were rerun rather than annotated - see
    # THE_FAILED_NIGHTS_were_one_bug.md. A failure that gets past this gate is a localised
    # gap to be rerun, not a day to be excluded.
    # What still holds a cell back is anything worse than a localised server gap.
    n_nights = max(1, len(nightly))
    if len(failed) > n_nights // 5:
        refuse(f"{household.name}/{arm}: the nightly call failed on {len(failed)} of "
               f"{n_nights} nights ({failed}), which is more than a fifth - that is not a "
               f"localised gap")

    did_not_parse = [n["day"] for n in nightly if n.get("the_completion_did_not_parse")]
    if did_not_parse:
        refuse(f"{household.name}/{arm}: the completion did not parse on nights "
               f"{did_not_parse}, so those nights wrote nothing while the call SUCCEEDED")

    # a night that saw something and wrote nothing, EXCLUDING nights whose call failed.
    #
    # THIS GATE HAS HAD NO POWER ON FIVE OF THE SEVEN ARMS, and measuring it is how that was
    # found rather than reasoning about it. The first condition reads
    # `the_look_saw_something_it_is_asked_about`, and ONLY `write_the_notes` writes that key.
    # Counted over one landed cell of each arm in the ten-home wave, 32 nights each:
    #
    #   incremental edits / prior only .......... 32 of 32 nights carry the key -> can refuse
    #   claim store told if it was right ........  0 -> the condition is always False
    #   the log and notes about the routine .....  0 -> always False
    #   a small working memory and an archive ...  0 -> always False
    #   wholesale rewrite .......................  0, and no `applied` key either
    #   MemGPT as published .....................  0, and no `applied` key either
    #
    # So for five arms this reported a clean result it had no way of reporting anything else
    # about, which is the shape of fault this project keeps finding. `the_night_wrote_nothing`
    # below computes the signal HERE, from the night's own report, so a writer cannot leave it
    # out. It is recorded for every arm now and refuses only where it already refused: turning
    # refusal on for the other five in the middle of a wave would hold back cells that have
    # run for hours under the weaker gate. That is the next wave's change, not this one's.
    #
    # And for the block arm the bar itself has to be lower. For a claim store, a night that
    # saw an asked-about object and wrote nothing is a refusal, because a sighting is a fact
    # the store exists to record. A fixed block of free text is not a list of facts, and
    # deciding that tonight's block already says what it needs to is an action the method is
    # entitled to take - its own smoke run did exactly that on night 4. So for the block arm
    # the gate is that the model was asked and made no call at all, which is still the
    # failure the gate exists to catch. Nights where it chose to change nothing are counted
    # in the report instead, never hidden.
    # Did tonight change the memory at all? One answer for every arm, from whichever key the
    # arm's own writer reports, and NONE when the report says nothing either way - which is
    # the whole point, because a function that returned True for a night it cannot read would
    # rebuild the fault it replaces one level up. The first draft of this did exactly that and
    # scored the wholesale-rewrite arm as having changed nothing on 32 of 32 nights; that arm
    # rewrites its whole summary, reports neither `applied` nor `did`, and says
    # `identical_to_last_night` instead.
    def the_night_changed_nothing(n: Dict[str, Any]) -> Optional[bool]:
        for key in ("applied", "did"):
            if key in n:
                return not any((n.get(key) or {}).values()) and not n.get(
                    "it_had_nothing_to_add")
        if "identical_to_last_night" in n:
            return bool(n["identical_to_last_night"])
        return None

    read = [(n["day"], the_night_changed_nothing(n)) for n in nightly
            if n["day"] not in failed]
    nights_that_wrote_nothing = [d for d, v in read if v is True]
    nights_with_no_signal = [d for d, v in read if v is None]

    if sensing not in sd.THE_NO_MODEL_ARMS:
        block_arm = how == MEMGPT_AS_PUBLISHED
        if block_arm:
            silent = [n["day"] for n in nightly
                      if int(n.get("n_calls_offered") or 0) == 0
                      and n["day"] not in failed]
            why_silent = ("was asked and made no call to its block or its archive at all")
        else:
            silent = [n["day"] for n in nightly
                      if n.get("the_look_saw_something_it_is_asked_about")
                      and not any((n.get("applied") or {}).values())
                      and not n.get("it_had_nothing_to_add")
                      and n["day"] not in failed]
            why_silent = "saw an asked-about object and applied no edit"
        if silent:
            refuse(f"{household.name}/{arm}: nights {silent} {why_silent}, and the call did "
                   f"NOT fail on those nights")

    about = {
        "arm": arm, "how_memory_is_written": how, "sensing_arm": sensing,
        "household": household.name, "bank": str(household.bank_path),
        "read_budget_lines": NO_LENGTH_LIMIT.read_budget_lines,
        "name_the_objects_it_will_be_quizzed_on_in_the_header":
            header.get("name_the_objects_it_will_be_quizzed_on"),
        "n_questions_where_the_read_budget_bit": len(bit),
        "n_nights_whose_completion_did_not_parse": len(did_not_parse),
        # Reported for every arm, refused for two. Read this before any accuracy number: a
        # memory that changed nothing on many nights is not the method it is named after.
        "nights_that_changed_the_memory_not_at_all": nights_that_wrote_nothing,
        "n_nights_that_changed_the_memory_not_at_all": len(nights_that_wrote_nothing),
        # Nights whose report cannot answer the question. Read the count above against this
        # one, never on its own: 0 of 32 means nothing if 32 of 32 could not be read.
        "n_nights_whose_report_cannot_say": len(nights_with_no_signal),
        "this_arm_can_be_refused_for_a_silent_night": bool(
            any("the_look_saw_something_it_is_asked_about" in n for n in nightly)
            or how == MEMGPT_AS_PUBLISHED),
        "nights_whose_call_failed": failed,
        "exclude_these_days_from_accuracy": [d + 1 for d in failed],
        "questions_per_day": questions_per_day,
        "questions_per_day_in_the_header": header.get("questions_per_day"),
        "n_model_calls_timed": timed.n_calls,
        "seconds": round(time.time() - started, 1),
        "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out_dir / "arm.json").write_text(json.dumps(about, indent=1))
    result["overnight_wave"] = about
    (out_dir / "cell.json").write_text(json.dumps(result, indent=1))

    # RELEASE THE LOCK ON SUCCESS. It used to be removed only by `refuse()`, so a cell that
    # finished cleanly left a `RUNNING.lock` behind for ever and the file meant "was started"
    # while its name says "is running". Nothing was corrupted - `take_the_lock` checks the
    # recorded pid and takes over a stale one - but every count of live cells built from these
    # files was wrong: at 11:30 today three of them (`a small working memory and an
    # archive`/hh_s19, hh_s20, hh_s48) showed as running when two had finished and one had
    # crashed thirteen minutes earlier.
    lock = out_dir / "RUNNING.lock"
    if lock.exists():
        lock.unlink()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--household", required=True)
    parser.add_argument("--banks", type=pathlib.Path, default=PILOT_BANKS)
    parser.add_argument("--last-day", type=int, default=31)
    parser.add_argument("--questions-per-day", type=int, default=8)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("results/self_improve/overnight_wave"))
    parser.add_argument("--cache", type=pathlib.Path,
                        default=pathlib.Path("llm_prior_cache/self_improve"))
    parser.add_argument("--backend", choices=["vllm", "anthropic"], default=None,
                        help="model backend; default vllm, or $PATROL_LLM_BACKEND")
    args = parser.parse_args(argv)

    household = FrozenHousehold(args.banks / f"{args.household}.jsonl")
    client = make_client(args.cache, args.backend)
    out = cell_dir(args.out, args.arm, household.name)
    result = run_one_arm(household, args.arm, client, out, args.last_day,
                         args.questions_per_day, args.budget, args.seed)
    print()
    print(json.dumps(result["overnight_wave"], indent=1))
    print(json.dumps(result["summary"], indent=1))
    print(f"model calls {client.stats['calls']}, cache hits {client.stats['cached']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The study's settings in one place, with the measurement behind each one.

Everything here is a parameter. Nothing about the sensing granularity, the
budget or the visit times is written into the other modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Settings:
    # ---- which scenario. Locked 2026-09-24.
    banks: str = "results/self_improve/runs/illness_v1/banks"
    """One resident off sick, questions asked about EVERYONE's things. Read-only:
    this directory belongs to another paper's finished numbers.

    Why not the owner-only regime: there every asked object belongs to the sick
    resident and nearly all of them land on the same coffee table, so one
    sentence covers the whole disruption and a per-claim memory cannot beat a
    wholesale rewrite by construction. Worse, the sick window there is EASIER
    than ordinary life: answering "coffee table" to everything scores 78% and
    knowing every object's own commonest place perfectly scores 83%, five points
    of headroom against a one-to-two-point rerun floor. In this regime the same
    two baselines are 58% and 80%: twenty-two points of headroom.

    Here each household has 8 to 18 asked objects, 3 to 8 of which move, to 1 to
    5 different destinations, and the rest never move at all - so the memory has
    both revision and preservation to do."""

    exclusion_rule_measured: str = "over daytime hours"
    """Households are dropped when fewer than two asked objects change their
    usual place. WHICH households that drops depends on this choice and the two
    answers differ: measured over daytime hours it drops none, measured at
    question times it drops s4. See which_objects_moved.py."""

    # ---- sensing. Locked 2026-09-24 on measured evidence; see EXPECTATIONS.md.
    granularity: str = "room"
    """Rooms, not places. A rotation over a household's 32 to 47 places takes 11
    to 15 days for one pass, longer than the ten-day disruption, so which
    households ever notice it would come down to where the rotation started."""

    budget_per_look: int = 1
    """One room per look. At three rooms a look a plain rotation sweeps the whole
    house every two to three days, leaving a robot that chooses nothing to find.
    At one room a look a rotation takes six to nine days, so the schedule
    typically sees the disruption late or not at all while a robot reasoning from
    its own failures can be there in a day or two."""

    visit_times: List[str] = field(default_factory=lambda: ["13:00"])
    """Daytime, in the middle of the 10:00-17:00 plateau where 71 to 75% of the
    disruption is visible. No 3am sweep: at 3am only 9 of the 53 asked-about
    objects across the ten households have a different resting place in the
    disrupted period than in the settled one, and in three households none do,
    so a night look cannot see the disruption at all. It would also hand every
    arm the same free evidence and shrink the differences we are measuring."""

    fixed_schedule_kind: str = "fair over every target"
    """The honest control. The alternative - rotate only the rooms that usually
    held things during the ordinary fortnight - never visits the room the
    disruption moves into in 9 of the 10 frozen households, because that room is
    in the settled-period top three in only one of them. A control that cannot
    possibly detect the change is rigged, not fair."""

    shared_warm_start: bool = True
    """One walkthrough of the whole house on day 0 at 18:00, the same for every
    arm. ON. The write-up must say plainly that the settled routine was GIVEN
    rather than learned, because the study is about revision, not acquisition.
    Measured reason it is needed: one room a day gives
    only 4.0 sightings per asked-about object across the whole month, and only
    17.6 days in 32 on which any asked-about object is seen at all (8 in the worst
    household), so claims about the ordinary routine rest on about two sightings
    each. Being identical across arms, it cannot favour one."""

    # ---- the two repairs to the chooser, kept as separate switches so each can be
    # left out in turn. They are different KINDS of change and the write-up must say
    # which is which.
    tell_it_days_since_each_room_was_looked_in: bool = True
    """RESTORES INFORMATION a real robot plainly has about its own history.
    Withholding it handicaps the policy rather than testing it. Measured reason it
    is needed: without it the chooser visited 2 distinct rooms in ten days against
    the schedule's 9, and never the room the disruption moved into."""

    require_the_two_claims_to_be_about_the_same_object: bool = True
    """A CORRECTION TO THE SPECIFICATION, not a change of policy. Two claims about
    two unrelated objects are not a disagreement about the world; the looser
    version was written by accident. Measured reason it is needed: 100% of the
    naive chooser's looks named two expectations that differed, but only 30%
    settled which claim held, because naming two different objects passes the
    weaker test without being a disagreement."""

    shuffle_the_candidate_order_with_seed: Optional[int] = 11
    """The order rooms are PRESENTED to a chooser in, reshuffled every night from this
    seed. Not a nicety - a repair. Measured 2026-09-24: with a fixed order the naive
    chooser picked the same room 8 times of 8 and it sat at position 1 every time;
    shuffled, the room changed with the order and every choice was at position 1 or
    position 6. The collapse was a serial-position artefact, not reasoning about the
    house, so any chooser result gathered with a fixed order is confounded and void.

    Shuffling also tests the repaired chooser for the same fault from the other end: it
    concentrates on a room that sits LAST in the fixed order.

    None restores the fixed order, for reproducing the artefact."""

    rotation_seed: int = 0
    """The order the fixed rota is walked in, per household. When one full pass
    takes longer than the thing being detected, this order carries the result, so
    it is chosen and recorded rather than left to how the room names sort."""

    # ---- the two budgets. They are separate and must not be merged.
    curation_looks_per_asked_object: int = 3
    """Looks spent on keeping the notes, across the whole study: about 36 looks
    in this regime, so roughly one a day, which is what the one-room-a-day
    design already gives."""

    search_looks_when_answering: int = 3
    """Places the robot may check while answering one question. This count is the
    headline search-cost metric and is reported separately from accuracy."""

    # ---- answering. Identical across every arm.
    read_budget_lines: int = 8
    """How many lines of notes either format may put into an answer prompt. One
    number, one unit, both formats - and this is the lever that makes the memory
    factor architectural instead of prompted.

    Eight is below one line per asked-about object in 9 of the 10 households (they
    have 10 to 18; only s2, with exactly 8, is unconstrained and its memory-factor
    result must be read with that said out loud). So the wholesale rewrite, being a
    single artifact, has to decide what to LOSE in order to fit, while the claim
    store keeps everything and decides what to SHOW within the same window.

    Not a length cap imposed on the rewrite by fiat, which would be choosing the
    result: the constraint is at read time and falls on both formats equally."""

    tell_the_model_everything_it_saw: bool = False
    """What the nightly note-writing prompt is shown from each look: every sighting
    and absence (True), or only the objects the robot is quizzed on (False).

    OFF by default, because every number measured up to 2026-09-24 was measured with
    it off and must stay reproducible. Measured reason the switch exists: a look
    RECORDS everything in the room - 67 distinct objects in one household's stream -
    but the description handed to the model was filtered to the 13 asked-about ones.
    The notes then mentioned 12 of those 13 and ZERO of the other 54, and residents
    appeared in 2 look records. So the memory could never learn that a resident is
    home all day, or that tissues and a laptop have appeared by the couch: it caches
    answers to anticipated questions rather than noticing the household changed.

    True passes `only_these_objects=None` to looking.describe_look_for_the_model,
    which renders every sighting and every absence. Nothing else differs - the same
    renderer, the same words, both formats - so the arm is one parameter wide.

    It is not free: the descriptions get longer and more of what is recorded is
    irrelevant, while read_budget_lines stays at 8. Accuracy going DOWN while
    recording goes UP is a real outcome about curation under a read budget.

    MEASURED with it True, patrol looking, day-23 notes, ten households: the wholesale
    rewrite still names ZERO of the 68.6 non-asked objects its looks recorded, exactly
    as with it False, and the claim store moves off zero in 3 of 10 households (+1.4
    objects, 2 se 1.72, so a hint and not an effect). Nearly all of that null is the
    prompt's own quiz list, not the description: see
    name_the_objects_it_will_be_quizzed_on below, and
    results/self_improve/RICH_LOOKS_AND_THE_QUIZ_LIST.md.

    The search-driven job is the consumer of this setting now; the patrol sweeps that
    measured the numbers above are at results/self_improve/memory_factor_rich_looks/."""

    name_the_objects_it_will_be_quizzed_on: bool = True
    """Whether the nightly note-writing prompt ENUMERATES the objects the robot is
    asked about, or only says it will be asked where things are.

    ON by default, because every number measured up to 2026-09-24 was measured with
    the list in the prompt and must stay reproducible. The line it controls is

        "The things the robot is asked about: " + ", ".join(household.asked_objects)

    and it confounds the rich-looks arm above. Measured 2026-09-24: with rich looks
    the notes still named ZERO of the 68.6 non-asked objects the looks recorded, in
    all ten households and both formats - but the prompt had just handed the model the
    quiz list, and one rich summary answers in its own words: "No items from the asked
    list were found in the living room or storage during the 13:00 check." So that
    zero is evidence about instruction-following, not about whether the model can use
    context. Turning this off is the arm that separates the two.

    With it off, the wholesale arm's read-budget instruction also stops quoting the
    NUMBER of asked objects ("this home has 13 things you are asked about"), because
    the count anchors on the same set the list does. The budget itself, the questions
    and the scoring are untouched: only what the writer is told to attend to moves.

    MEASURED with it False, rich looks, wholesale rewrite, day-23 notes, three
    households (s0, s5, s9): the notes name 6.7 non-asked objects and 16.7 non-asked
    class words, up from exactly zero with the list, while naming only 2.0 of 16.7 asked
    objects, down from 14.3 - and the notes are no longer (712 -> 768 characters). A
    fixed read budget makes curation zero-sum, so this is the same purse spent on
    different things. Accuracy under it has NOT been measured.

    The framing that matters: enumerating the quiz set is the study telling the memory
    what it will be tested on, which a deployed robot does not get. Every memory-format
    number measured with it True is about writing notes when you already know what you
    will be asked. The search-driven job is the consumer of this setting now."""

    # ---- the model. Local vLLM only; no hosted API, no spend.
    endpoint: str = "http://localhost:8000"   # placeholder; see PATROL_LLM_ENDPOINT
    temperature: float = 0.0

    # ---- households.
    households: Optional[List[str]] = None
    """None means all ten frozen households. Results are always reported per
    household and then combined; nothing is ever pooled across household sets."""

    # ---- how much of a difference is real.
    noise_floor: float = 0.05
    """Generation on this server is not deterministic: the same notes rerun on
    the same data change about 3 to 5% of their answers. Treat any difference
    smaller than this as noise."""


LOCKED = Settings()

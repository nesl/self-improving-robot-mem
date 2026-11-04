"""Looking: the only way the robot learns anything in this study.

There is no automatic feedback here. Nobody tells the robot where things were
after it answers, and in the primary condition no resident sends it a message.
Every piece of evidence it has comes from a look it took or was scheduled to
take.

A look inspects a list of TARGETS at a VISIT TIME. A target is a room or a
place, and which one it is comes from the configuration, not from this file: we
do not yet know whether the study runs at room level or place level and this
code must not care.

Visit times are a list in the configuration too, because a look at 3am is nearly
blind to a daytime disruption - things are tidied back overnight and the night
world looks the same in both periods - so the times matter as much as the
targets.

Everything a look produces is logged: the targets, the time, every sighting with
its own id, every place found empty, and - for the arm where the model chooses -
the whole structured action, including what it expected to see under each of the
two claims it named.
"""
from __future__ import annotations

import json
import pathlib
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from baselines.types import DAY_SECONDS
from self_improve.frozen_household import (BEYOND_REACH, FrozenHousehold, LookTarget,
                                           period_of_day, plain_place_name)


@dataclass(frozen=True)
class Sighting:
    """One thing the robot saw, with an id the notes can cite."""
    observation_id: str
    object_id: str
    object_class: str
    place_id: str
    room: str
    day: int
    time: int
    revealed_by_target: str

    def as_plain_words(self) -> str:
        return (f"[{self.observation_id}] day {self.day} {hours_and_minutes(self.time)}: "
                f"{self.object_id} on {plain_place_name(self.place_id)} in the {self.room}")


@dataclass(frozen=True)
class Absence:
    """Looked here, and the thing was not here.

    The strongest signal in the whole setup and it was switched off in every
    config on disk (`negative_evidence: false`). Measured: the object is absent
    from where memory expects it on 84-88% of object-days. It is also exactly the
    signal the looking factor depends on - a robot that goes to the desk, finds
    no charger and concludes the desk claim is failing is doing the thing the
    hypothesis is about.

    It has the same shape as a sighting and its own observation id, so a claim can
    cite "looked and it was not there" the same way it cites a sighting.
    """
    observation_id: str
    object_id: str
    place_id: str
    room: str
    day: int
    time: int
    looked_at_target: str

    def as_plain_words(self) -> str:
        return (f"[{self.observation_id}] day {self.day} {hours_and_minutes(self.time)}: "
                f"{self.object_id} was NOT on {plain_place_name(self.place_id)} "
                f"in the {self.room}")


@dataclass
class ChosenLook:
    """The structured action the choosing arm must produce.

    The two lists of expected objects are what make the degenerate case
    checkable: if the two claims predict the same objects at the target, the
    look cannot tell them apart, and no accuracy number would ever reveal that.
    """
    target: str
    target_kind: str
    first_claim_id: str
    first_claim: str
    second_claim_id: str
    second_claim: str
    objects_expected_if_the_first_claim_holds: List[str]
    objects_expected_if_the_second_claim_holds: List[str]
    what_i_would_change_if_the_first_claim_holds: str
    what_i_would_change_if_the_second_claim_holds: str
    reasoning: str = ""
    the_thing_the_two_claims_disagree_about: str = ""
    candidate_rooms_in_the_order_they_were_shown: List[str] = field(default_factory=list)
    position_of_the_room_it_chose: int = 0
    the_candidate_order_was_shuffled: bool = False
    # What the notes said about each candidate when the choice was made. Without this a
    # fixation can be shown not to be positional but cannot be explained: we cannot tell
    # "it picks the room it holds fewest claims about" from "it picks the room whose
    # claims conflict most".
    what_the_notes_predicted_for_each_room: Dict[str, List[str]] = field(default_factory=dict)
    how_many_claims_the_notes_hold_about_each_room: Dict[str, int] = field(default_factory=dict)


@dataclass
class LookRecord:
    """One look, everything about it, as it goes into the log."""
    look_id: str
    day: int
    time: int
    period: str
    chosen_by: str                      # "the fixed schedule" or "the model"
    targets: List[Dict[str, str]]
    sightings: List[Dict[str, Any]] = field(default_factory=list)
    absences: List[Dict[str, Any]] = field(default_factory=list)
    places_found_empty: List[str] = field(default_factory=list)
    residents_seen: List[str] = field(default_factory=list)
    asked_objects_seen: List[str] = field(default_factory=list)
    chosen_look: Optional[Dict[str, Any]] = None
    model_call_failed: bool = False

    def as_plain_words(self) -> str:
        where = ", ".join(t["name"] for t in self.targets)
        return (f"[{self.look_id}] day {self.day} {hours_and_minutes(self.time)}: "
                f"looked at {where}; saw {len(self.sightings)} thing(s)")


def hours_and_minutes(t: int) -> str:
    hh, mm = divmod((int(t) % DAY_SECONDS) // 60, 60)
    return f"{hh:02d}:{mm:02d}"


class TheHouseAsSeen:
    """Performs looks against a frozen household and logs every one of them.

    The household knows the truth; this class is the narrow door through which
    any of it reaches an arm. Nothing else in an arm may read the household's
    truth methods.
    """

    def __init__(self, household: FrozenHousehold, log_path: pathlib.Path,
                 granularity: str,
                 names: Optional[Dict[str, str]] = None) -> None:
        # `names` maps resident_1 to the real name. When it is given, every look records
        # the person by name, so the robot never sees an identifier for a person and never
        # has to work out which identifier is whom - a puzzle about our data rather than
        # about the household. Default None keeps the old behaviour exactly.
        self.names = names or {}
        self.household = household
        self.granularity = granularity
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w")
        self._look_number = 0
        self._sighting_number = 0
        self._absence_number = 0
        self._objects_the_robot_knows_of: List[str] = []
        self.looks: List[LookRecord] = []
        self._log.write(json.dumps({
            "kind": "log_header", "household": household.name,
            "granularity": granularity,
            "n_targets_available": len(household.look_targets(granularity)),
        }) + "\n")

    def close(self) -> None:
        if not self._log.closed:
            self._log.close()

    def available_targets(self) -> List[LookTarget]:
        return self.household.look_targets(self.granularity)

    def look(self, targets: Sequence[LookTarget], day: int, time_of_day_seconds: int,
             chosen_by: str, chosen_look: Optional[ChosenLook] = None,
             model_call_failed: bool = False) -> LookRecord:
        """Inspect these targets at this moment. Returns the record; the record
        is also appended to the log file."""
        at_time = day * DAY_SECONDS + time_of_day_seconds
        self._look_number += 1
        record = LookRecord(
            look_id=f"look_{self._look_number:04d}", day=day, time=at_time,
            period=period_of_day(day), chosen_by=chosen_by,
            targets=[{"name": t.name, "kind": t.kind} for t in targets],
            chosen_look=asdict(chosen_look) if chosen_look is not None else None,
            model_call_failed=model_call_failed)

        seen_residents: List[str] = []
        for target in targets:
            room = self.household.room_containing(target)
            for who in self.household.residents_in_room(room, at_time):
                if who not in seen_residents:
                    seen_residents.append(who)
            for place in self.household.places_behind_target(target):
                contents = self.household.contents_of_place(place, at_time)
                if not contents:
                    record.places_found_empty.append(place)
                for object_id in contents:
                    self._sighting_number += 1
                    sighting = Sighting(
                        observation_id=f"sighting_{self._sighting_number:04d}",
                        object_id=object_id,
                        object_class=self.household.object_class.get(object_id, ""),
                        place_id=place, room=room, day=day, time=at_time,
                        revealed_by_target=target.name)
                    record.sightings.append(asdict(sighting))
        # Negative evidence: for every object the robot already knows exists, any
        # place it looked at and did not find it. Both ways of writing notes
        # receive these through describe_look_for_the_model, so neither format can
        # represent absence better than the other.
        found_now = {s["object_id"] for s in record.sightings}
        looked_at: List[Tuple[str, str, str]] = []
        for target in targets:
            room = self.household.room_containing(target)
            for place in self.household.places_behind_target(target):
                looked_at.append((place, room, target.name))
        for object_id in self._objects_the_robot_knows_of:
            if object_id in found_now:
                continue
            for place, room, target_name in looked_at:
                self._absence_number += 1
                record.absences.append(asdict(Absence(
                    observation_id=f"absence_{self._absence_number:04d}",
                    object_id=object_id, place_id=place, room=room,
                    day=day, time=at_time, looked_at_target=target_name)))
        for object_id in sorted(found_now):
            if object_id not in self._objects_the_robot_knows_of:
                self._objects_the_robot_knows_of.append(object_id)

        record.residents_seen = [self.names.get(who, who) for who in seen_residents]
        record.asked_objects_seen = sorted(
            {s["object_id"] for s in record.sightings
             if s["object_id"] in set(self.household.asked_objects)})

        self.looks.append(record)
        self._log.write(json.dumps({"kind": "look", **asdict(record)}) + "\n")
        self._log.flush()
        return record

    def shared_warm_start(self, day: int = 0, at_clock: str = "18:00") -> LookRecord:
        """One walkthrough of the whole house, identical for every arm.

        Why it is offered: measured on 2026-09-24, one room a day over a fair
        rotation gives on average 4.0 sightings per asked-about object across the
        whole month, only 17.6 days in 32 on which any asked-about object is seen
        at all, and in the worst household 8 days in 32. Half a dozen claims have
        to be written from two settled-period sightings each. That starves the
        memory before the disruption even starts.

        A walkthrough on day 0 fixes the settled-period starvation without
        touching the disruption, and because it is the same for every arm it
        cannot favour one. It is the bank's own opening walkthrough time (18:00 on
        day 0), so it is not an invention of this study.

        It is off by default because turning it on changes what every arm knows;
        it is a setting, and whether it was used is recorded in the log header.
        """
        every_room = [t for t in self.household.look_targets("room")]
        return self.look(every_room, day, clock_to_seconds(at_clock),
                         "the shared walkthrough every arm gets")

    # ------------------------------------------------- what an arm may read --

    def days_since_each_room_was_looked_in(self, day: int) -> Dict[str, Optional[int]]:
        """How long since the robot last looked in each room, as of this day.

        The robot's own history, not a hint about the world: it knows where it has
        been. None means never looked. Used by the repaired chooser so it can weigh
        a stale room against a live disagreement.
        """
        last: Dict[str, int] = {}
        for look in self.looks:
            for target in look.targets:
                room = self.household.room_containing(LookTarget(**target))
                last[room] = max(last.get(room, look.day), look.day)
        return {room: (day - last[room] if room in last else None)
                for room in self.household.rooms}

    def absences_so_far(self) -> List[Absence]:
        return [Absence(**a) for look in self.looks for a in look.absences]

    def sightings_so_far(self) -> List[Sighting]:
        return [Sighting(**s) for look in self.looks for s in look.sightings]

    def sightings_of(self, object_id: str) -> List[Sighting]:
        return [s for s in self.sightings_so_far() if s.object_id == object_id]

    def looks_that_did_not_find(self, object_id: str) -> List[Tuple[str, int, List[str]]]:
        """Negative evidence: looks whose listing was complete and did not show
        this object. (look_id, time, places inspected)."""
        out = []
        for look in self.looks:
            if object_id in {s["object_id"] for s in look.sightings}:
                continue
            places = [p for t in look.targets
                      for p in self.household.places_behind_target(LookTarget(**t))]
            out.append((look.look_id, look.time, places))
        return out


# -------------------------------------------------------- the fixed schedule


class FixedLookSchedule:
    """Where the robot looks when the schedule is fixed in advance.

    Written once, before day 0, and then never changed - that is what makes it
    the control. Two honest ways to write one, and which we use matters more
    than the budget does, so it is a named parameter and the name goes in the
    results:

      "fair over every target"   round-robin over all of the household's
                                 targets, covering the house evenly. Fair, but
                                 it spends most looks on places nothing is ever
                                 kept in, so it is weak in every period, not
                                 just the disrupted one.
      "the usual places, fairly" the targets that most often held an asked-about
                                 object during the ordinary fortnight, rotated
                                 among themselves. This is the schedule a
                                 competent engineer would ship, and it is the
                                 honest control: it fails during the disruption
                                 because the world changed, not because we
                                 crippled it.

    "the usual places, fairly" is built from the settled period ONLY. It never
    sees the disrupted or back-to-normal days, so it is genuinely in advance.
    """

    KINDS = ("fair over every target", "the usual places, fairly")

    def __init__(self, household: FrozenHousehold, granularity: str, kind: str,
                 budget_per_look: int, visit_times: Sequence[str],
                 rotation_seed: int = 0,
                 settled_period_days: Sequence[int] = tuple(range(14))) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"schedule kind must be one of {self.KINDS}, got {kind!r}")
        self.household = household
        self.rotation_seed = rotation_seed
        self.granularity = granularity
        self.kind = kind
        self.budget_per_look = budget_per_look
        self.visit_times = [clock_to_seconds(v) for v in visit_times]
        self.visit_time_words = list(visit_times)
        every = household.look_targets(granularity)
        if kind == "fair over every target":
            self.rota = every
        else:
            self.rota = self._usual_places(settled_period_days, every)
        # The order the rota is walked in carries the result whenever a full pass
        # takes longer than the thing we are trying to detect, so it is a seed we
        # choose and record per household rather than an accident of sorting.
        random.Random(f"{household.name}/{rotation_seed}").shuffle(self.rota)
        self.built_from = kind

    def _usual_places(self, settled_period_days: Sequence[int],
                      every: Sequence[LookTarget]) -> List[LookTarget]:
        """The targets that most often held an asked-about object during the
        ordinary fortnight, at the same times of day the robot will visit.
        Kept to a shortlist the budget can rotate through in a few days."""
        score: Dict[str, int] = {t.name: 0 for t in every}
        for day in settled_period_days:
            for seconds in self.visit_times:
                at_time = day * DAY_SECONDS + seconds
                for object_id in self.household.asked_objects:
                    place = self.household.place_of_object(object_id, at_time)
                    if place is None:
                        continue
                    name = (place if self.granularity == "place"
                            else self.household.place_room.get(place))
                    if name in score:
                        score[name] += 1
        ranked = [t for t in sorted(every, key=lambda t: (-score[t.name], t.name))
                  if score[t.name] > 0]
        shortlist_size = max(self.budget_per_look, min(len(ranked), 3 * self.budget_per_look))
        return ranked[:shortlist_size] or list(every[:self.budget_per_look])

    def targets_for(self, day: int, visit_index: int) -> List[LookTarget]:
        if not self.rota:
            return []
        start = ((day * len(self.visit_times) + visit_index) * self.budget_per_look) % len(self.rota)
        return [self.rota[(start + i) % len(self.rota)] for i in range(min(self.budget_per_look,
                                                                          len(self.rota)))]

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "granularity": self.granularity,
                "budget_per_look": self.budget_per_look,
                "visit_times": self.visit_time_words,
                "rotation_seed": self.rotation_seed,
                "days_for_one_full_pass": self.days_for_one_full_pass(),
                "rota_in_the_order_it_is_walked": [t.name for t in self.rota]}

    def days_for_one_full_pass(self) -> float:
        """How long the schedule takes to visit everything on its rota once.
        If this is longer than the disruption, the schedule can miss it entirely
        and the starting point of the rota decides the result."""
        looks_per_day = len(self.visit_times) * self.budget_per_look
        return len(self.rota) / looks_per_day if looks_per_day else float("inf")


def clock_to_seconds(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 3600 + int(minutes) * 60


def describe_look_for_the_model(record: LookRecord,
                               only_these_objects: Optional[Sequence[str]] = None,
                               list_what_was_absent: bool = True) -> str:
    """What a look tells the robot, in words.

    The SAME function feeds both ways of writing notes. If only one format could
    represent absence, the two factors would be confounded and no difference we
    measured would be attributable to either, so there is exactly one renderer
    and both formats call it.
    """
    # `list_what_was_absent=False` leaves the NOT FOUND block out. A look lists
    # everything in the room, so absence is already implied by what is listed, and naming
    # all sixty things that were not there costs about a thousand characters a look - 14
    # looks a day, thirty-one days. The arm that reads its own record is told once, in
    # words, that anything not listed was not in the room, and passes False. Default True,
    # so every other arm's prompt is unchanged to the character.
    wanted = set(only_these_objects) if only_these_objects is not None else None
    where = ", ".join(t["name"] for t in record.targets)
    lines = [f"Day {record.day} at {hours_and_minutes(record.time)}, "
             f"the robot looked in: {where}."]

    found = [s for s in record.sightings
             if wanted is None or s["object_id"] in wanted]
    if found:
        lines.append("It found:")
        for s in sorted(found, key=lambda s: s["object_id"]):
            lines.append(f"  - {s['object_id']} on {plain_place_name(s['place_id'])} "
                         f"in the {s['room']}  [{s['observation_id']}]")
    else:
        lines.append("It found none of the things it is asked about.")

    # Absence is reported once per object per room looked at, not once per place,
    # or one look at a nine-place kitchen would drown the note in repetition.
    missing: Dict[Tuple[str, str], str] = {}
    for a in record.absences:
        if wanted is not None and a["object_id"] not in wanted:
            continue
        missing.setdefault((a["object_id"], a["room"]), a["observation_id"])
    if missing and list_what_was_absent:
        # Grouped by room and listed, rather than a sentence per object per room with an
        # observation id on each. Measured on one night of one household: 2,036 characters
        # the old way against 388 this way, for the same information. Nothing is ever
        # asked to cite an absence, so the ids were paid for and never used.
        by_room: Dict[str, List[str]] = {}
        for (object_id, room) in sorted(missing):
            by_room.setdefault(room, []).append(object_id)
        lines.append("NOT FOUND:")
        for room, things in by_room.items():
            lines.append(f"  - {room}: " + ", ".join(things))
    if record.residents_seen:
        lines.append(f"People there: {', '.join(record.residents_seen)}.")
    return "\n".join(lines)

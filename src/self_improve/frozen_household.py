"""One frozen household: its rooms, its places, and where everything really is.

We read the frozen question banks at
results/self_improve/varied_homes/<bank set>/banks and never
write to them. Nothing here reaches a prompt: this is the harness side of the
line, the part that knows the truth.

Vocabulary used throughout src/self_improve, deliberately plain:

  room      one of the 6 to 9 rooms of the house ("kitchen", "living")
  place     one spot inside a room where things rest ("desk_o1", "couch_l1");
            there are roughly 33 to 47 of these per household
  target    something the robot can look at: a room or a place. Which of the
            two we use is the study's sensing granularity and it is a
            parameter, never a constant.
  asked object  an object the frozen question bank actually asks about. There
            are only 4 to 7 of these per household, far fewer than the
            household's 70-odd objects.
  settled period   days 0-13, the ordinary fortnight
  disrupted period days 14-23, one resident off sick
  back-to-normal period days 24-31
"""
from __future__ import annotations

import bisect
import collections
import json
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from baselines.types import DAY_SECONDS, ON_PERSON, OUT_OF_HOUSE

# The scenario the study runs on: illness_v1, as the directory stands now. It passes
# all six scenario checks and all four legs of the gate (learn +14.2, break -13.3,
# re-learn +13.5, break again -8.9), and 5 of 10 households show the full
# four-phase shape.
#
# A correction to the record, because it was repeated several times: "illness_v1
# failed seven ways" was true of an EARLIER build and is NOT true of the directory
# now at results/self_improve/runs/illness_v1/. It must not be described as the
# failed scenario anywhere.
#
# illness_v2 is superseded: it fails the fourth leg outright - a forgetting learner
# loses only 1.6 points on the first two days back to normal against a 5-point bar,
# with a household-clustered error of 4.2 - so its return is not a change and no arm
# can be measured on it.
# Public release: only the bank sets under results/self_improve/varied_homes/ are included; the older
# bank directories named below are not, and every paper run passes --banks explicitly.
FROZEN_BANKS = pathlib.Path(
    "results/self_improve/runs/illness_v1/banks")

SUPERSEDED_ILLNESS_V2 = pathlib.Path(
    "results/self_improve/runs/illness_v2/banks")

# The earlier scenarios, kept so the measurements that ruled them out reproduce.
SICK10_PARTIAL_SUPERSEDED = pathlib.Path(
    "results/regime_search/sick10_partial/banks")

# The earlier scenario, kept only so the measurements that ruled it out can be
# reproduced: there every asked object belongs to the sick resident, nearly all
# of them land on the same coffee table, and the sick window is easier than
# ordinary life rather than harder.
OWNER_ONLY_BANKS_RULED_OUT = pathlib.Path(
    "results/fm_memory/banks_f1")

# The simulator files a carried object under a pretend room of this name. It is
# not a room you can walk into, so no look can reach it.
NOT_A_REAL_ROOM = "person_check"

# What we say when an object is somewhere no look can reach: out of the house,
# or in a resident's pocket.
BEYOND_REACH = "beyond the reach of any look"

PERIODS = (("settled", range(0, 14)),
           ("disrupted", range(14, 24)),
           ("back to normal", range(24, 32)))


def period_of_day(day: int) -> str:
    for name, days in PERIODS:
        if day in days:
            return name
    return "unknown"


def plain_place_name(place_id: str) -> str:
    """'coffee_table_l1' -> 'the coffee table'. The bank's place ids carry a
    room suffix that means nothing to a reader, so we drop it for prose while
    keeping the id as the machine key."""
    parts = place_id.split("_")
    if len(parts) > 1 and any(ch.isdigit() for ch in parts[-1]):
        parts = parts[:-1]
    return "the " + " ".join(parts).replace("_", " ")


@dataclass(frozen=True)
class LookTarget:
    """Something the robot can look at."""
    name: str          # a room name, or a place id
    kind: str          # "room" or "place"

    def __post_init__(self) -> None:
        if self.kind not in ("room", "place"):
            raise ValueError(f"LookTarget kind must be 'room' or 'place', got {self.kind!r}")

    def as_plain_words(self) -> str:
        return self.name if self.kind == "room" else plain_place_name(self.name)


class FrozenHousehold:
    """Everything the harness knows about one frozen household."""

    def __init__(self, bank_path: pathlib.Path) -> None:
        self.bank_path = bank_path
        self.name = bank_path.stem
        header = None
        self.questions: List[dict] = []
        self.resident_rows: List[dict] = []
        moves: Dict[str, List[Tuple[int, str]]] = collections.defaultdict(list)
        for line in bank_path.open():
            row = json.loads(line)
            kind = row["kind"]
            if kind == "episode_header":
                header = row
            elif kind == "truth":
                moves[row["object_id"]].append((row["t"], row["receptacle_id"]))
            elif kind == "question":
                self.questions.append(row)
            elif kind == "resident":
                self.resident_rows.append(row)
        if header is None:
            raise ValueError(f"{bank_path} has no episode_header row")
        self.header = header
        for history in moves.values():
            history.sort()
        self.moves = moves

        self.place_room: Dict[str, str] = dict(header["receptacle_rooms"])
        self.rooms: List[str] = sorted(
            {room for room in self.place_room.values() if room != NOT_A_REAL_ROOM})
        self.places_in_room: Dict[str, List[str]] = collections.defaultdict(list)
        for place, room in sorted(self.place_room.items()):
            if room != NOT_A_REAL_ROOM:
                self.places_in_room[room].append(place)
        self.places: List[str] = sorted(
            place for place, room in self.place_room.items() if room != NOT_A_REAL_ROOM)
        self.n_days: int = int(header["n_days"])
        self.day_names: Dict[int, str] = {int(k): v for k, v in header["day_names"].items()}
        self.object_class: Dict[str, str] = dict(header.get("object_classes") or {})
        self.asked_objects: List[str] = sorted({q["object_id"] for q in self.questions})
        self.resident_ids: List[str] = list(header["resident_ids"])

    # ---------------------------------------------------------------- truth --

    def place_of_object(self, object_id: str, at_time: int) -> Optional[str]:
        """The place id holding this object at this moment, or ON_PERSON /
        OUT_OF_HOUSE, or None if it has no history yet."""
        history = self.moves.get(object_id)
        if not history:
            return None
        index = bisect.bisect_right(history, (at_time, "￿")) - 1
        return history[index][1] if index >= 0 else None

    def room_of_object(self, object_id: str, at_time: int) -> str:
        place = self.place_of_object(object_id, at_time)
        if place is None or place in (OUT_OF_HOUSE, ON_PERSON):
            return BEYOND_REACH
        room = self.place_room.get(place)
        if room is None or room == NOT_A_REAL_ROOM:
            return BEYOND_REACH
        return room

    def can_be_looked_at(self, object_id: str, at_time: int) -> bool:
        return self.room_of_object(object_id, at_time) != BEYOND_REACH

    def contents_of_place(self, place_id: str, at_time: int) -> List[str]:
        """Every object resting in this place at this moment. The complete
        listing, so an object's absence from it is real negative evidence."""
        found = []
        for object_id in self.moves:
            if self.place_of_object(object_id, at_time) == place_id:
                found.append(object_id)
        return sorted(found)

    def contents_of_room(self, room: str, at_time: int) -> Dict[str, List[str]]:
        return {place: self.contents_of_place(place, at_time)
                for place in self.places_in_room[room]}

    def residents_in_room(self, room: str, at_time: int) -> List[str]:
        """Who was in this room at this moment, from the simulator's resident
        trajectory rows (the newest row at or before the time, per resident)."""
        latest: Dict[str, Tuple[int, str]] = {}
        for row in self.resident_rows:
            if row["t"] <= at_time:
                who = row["resident_id"]
                if who not in latest or row["t"] >= latest[who][0]:
                    latest[who] = (row["t"], row["room"])
        return sorted(who for who, (_, where) in latest.items() if where == room)

    # -------------------------------------------------------------- targets --

    def look_targets(self, granularity: str) -> List[LookTarget]:
        """Everything the robot could look at, at the requested granularity.
        This is the one place the granularity decision lives."""
        if granularity == "room":
            return [LookTarget(room, "room") for room in self.rooms]
        if granularity == "place":
            return [LookTarget(place, "place") for place in self.places]
        raise ValueError(f"granularity must be 'room' or 'place', got {granularity!r}")

    def places_behind_target(self, target: LookTarget) -> List[str]:
        """The places a single look at this target would inspect."""
        if target.kind == "place":
            if target.name not in self.place_room:
                raise ValueError(f"{self.name}: no such place {target.name!r}")
            return [target.name]
        if target.name not in self.places_in_room:
            raise ValueError(f"{self.name}: no such room {target.name!r}")
        return list(self.places_in_room[target.name])

    def room_containing(self, target: LookTarget) -> str:
        return target.name if target.kind == "room" else self.place_room[target.name]

    # ------------------------------------------------------------ questions --

    def questions_on_day(self, day: int) -> List[dict]:
        return [q for q in self.questions if q["day_index"] == day]

    def true_place_for_question(self, question: dict) -> Optional[str]:
        return self.place_of_object(question["object_id"], question["t_query"])


def load_every_frozen_household(banks_dir: pathlib.Path = FROZEN_BANKS
                                ) -> List[FrozenHousehold]:
    paths = sorted(banks_dir.glob("hh_s*_t03.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no frozen banks under {banks_dir}")
    return [FrozenHousehold(p) for p in paths]

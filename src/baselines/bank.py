"""Episode banks: the JSONL interchange format, its loader, and a synthetic
fixture builder used until real banks exist.

JSONL schema (one JSON object per line; lines of one episode are contiguous
and start with its header; a file may hold many episodes):

    {"kind": "episode_header", "episode_id": str, "household_id": str,
     "receptacle_ids": [str, ...], "object_classes": {object_id: class},
     "budget_per_day": int, "n_days": int,
     "household_type": str (optional bank metadata; enables the
      healthcheck's stratified discriminative gate),
     "unsensable_receptacles": [str, ...] (optional, default none: legal
      ANSWERS that Sense may never target — e.g. OUT_OF_HOUSE; agents can
      only infer them by eliminating every sensable receptacle),
     "receptacle_rooms": {receptacle_id: room_id} (optional: the room
      holding each in-house receptacle; receptacles outside any room —
      OUT_OF_HOUSE — are simply absent. Feeds the room-change sense
      cost),
     "home_base_room": str (optional, required with receptacle_rooms:
      the room the robot starts each day in — the room with the most
      receptacles, ties broken by room id sort order),
     "query_generation": "uniform"|"routine_driven" (optional for old
      banks, which are all uniform; REQUIRED as soon as any question row
      carries an "origin" — a routine-driven bank must declare itself),
     "person_sensing": bool (optional, default false: the bank lets a
      policy look at a resident — see "resident" rows and the "carrier"
      field below; a bank without it loads, runs and scores exactly as
      before the field existed),
     "resident_ids": [str, ...] (optional; required with person_sensing:
      the household roster, vocabulary the agent may see)}

    {"kind": "truth", "episode_id": str, "object_id": str, "t": int,
     "receptacle_id": str, "carrier": str (optional; on a person_sensing
      bank REQUIRED whenever receptacle_id is ON_PERSON: the resident
      carrying the object from t on. Ground truth, harness-only)}
        Piecewise-constant ground truth: object is at receptacle from t
        until its next truth row. Every object needs a t=0 row. Times are
        seconds since episode start (as everywhere in this package).

    {"kind": "resident", "episode_id": str, "resident_id": str, "t": int,
     "room": str}
        Piecewise-constant ground truth of where a resident is: a room of
        the receptacle map, or "AWAY" while out. Every resident needs a
        t=0 row. Harness-only, like truth rows: it feeds the presence
        listing a receptacle sense returns and what a person sense finds,
        and never reaches an agent's context. Only person_sensing banks
        carry these rows.

    {"kind": "observation", "episode_id": str, "object_id": str,
     "receptacle_id": str, "t": int, "source": "initial_tour"|"scripted"}
        The fixed observation stream. Rows with source "initial_tour" are
        the installation walkthrough — one sighting per sensable object at
        the tour instant (t=0 by default; the optional header field
        "tour_t" records a later instant) — and are delivered before any
        other evidence; "scripted" rows are delivered in time order
        interleaved with questions. "sense" never appears in a bank — sense
        observations exist only inside a run.

    {"kind": "question", "episode_id": str, "question_id": str,
     "object_id": str, "t_query": int, "day_index": int,
     "object_class": str (the queried object's class; optional when the
      header's object_classes declares the object — the loader backfills
      from there and rejects a question whose class it cannot determine,
      or whose row class contradicts the header)}

    {"kind": "room_visit", "episode_id": str, "t": int, "room": str,
     "contents": {receptacle_id: [object_id, ...]}}
        One room visit from the ambient observation stream: at ``t``,
        every receptacle in ``contents`` was inspected and found to hold
        exactly the listed objects (empty lists included — that emptiness
        is evidence). The loader delivers a visit to the beliefs as one
        sense result per receptacle, so it carries positive sightings and
        exclusions at once; ``Episode.scripted_observations`` still
        exposes the positive half as ordinary observations for consumers
        that only need sightings (the viewer's "last seen", recency
        statistics).

The loader is strict: unknown kinds, out-of-order episode_ids, references
to undeclared receptacles/objects, missing t=0 truth, or day_index
disagreeing with t_query all raise ``BankFormatError`` with the offending
line number. Real-bank producers conform to this schema or negotiate
changes against it.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import pathlib
import random
from dataclasses import dataclass
from typing import (Any, Dict, Iterator, List, Protocol, Tuple,
                    Union)

from baselines.types import (AWAY, DAY_SECONDS, ON_PERSON, Episode,
                             Observation, Question, SenseResult)

logger = logging.getLogger(__name__)

_BANK_SOURCES = ("initial_tour", "scripted")


class BankFormatError(ValueError):
    """A malformed bank row; message carries file, line number, and context."""


class EpisodeBank(Protocol):
    """What the harness and CLI need from any bank implementation."""

    @property
    def path(self) -> pathlib.Path:
        """Where the bank lives on disk (for provenance)."""

    @property
    def manifest_hash(self) -> str:
        """Stable content hash of the bank file (for provenance)."""

    def episodes(self) -> Iterator[Episode]:
        """Yield episodes in file order."""


@dataclass(frozen=True)
class JsonlBank:
    """Loader for the JSONL schema documented in this module's docstring."""

    path: pathlib.Path

    @property
    def manifest_hash(self) -> str:
        """SHA-256 of the bank file bytes."""
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def episodes(self) -> Iterator[Episode]:
        """Parse and yield each episode, validating as documented."""
        current: _EpisodeAccumulator | None = None
        with open(self.path) as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as err:
                    raise BankFormatError(
                        f"{self.path}:{lineno}: invalid JSON: {err}") from err
                kind = row.get("kind")
                if kind == "episode_header":
                    if current is not None:
                        yield current.finish()
                    current = _EpisodeAccumulator(row, self.path, lineno)
                elif current is None:
                    raise BankFormatError(
                        f"{self.path}:{lineno}: {kind!r} row before any "
                        f"episode_header")
                elif kind in ("truth", "observation", "question",
                              "room_visit", "resident"):
                    current.add(kind, row, lineno)
                else:
                    raise BankFormatError(
                        f"{self.path}:{lineno}: unknown kind {kind!r}")
        if current is not None:
            yield current.finish()


class _EpisodeAccumulator:
    """Builds one Episode from its header + body rows, validating eagerly."""

    def __init__(self, header: Dict[str, Any], path: pathlib.Path,
                 lineno: int) -> None:
        self._path = path
        try:
            self.episode_id = str(header["episode_id"])
            self.household_id = str(header["household_id"])
            self.receptacles = tuple(str(r) for r in header["receptacle_ids"])
            self.object_classes = {
                str(k): str(v)
                for k, v in header["object_classes"].items()}
            self.budget_per_day = int(header["budget_per_day"])
            self.n_days = int(header["n_days"])
            raw_type = header.get("household_type")
            self.household_type = None if raw_type is None else str(raw_type)
            self.premises: Dict[str, str] = {
                str(k): str(v) for k, v in dict(header.get("premises") or {}).items()}
            self.unsensable = tuple(
                str(r) for r in header.get("unsensable_receptacles", []))
            self.receptacle_rooms = {
                str(k): str(v)
                for k, v in (header.get("receptacle_rooms") or {}).items()}
            raw_home = header.get("home_base_room")
            self.home_base_room = None if raw_home is None else str(raw_home)
            raw_qg = header.get("query_generation")
            self.query_generation = None if raw_qg is None else str(raw_qg)
            raw_tour = header.get("tour_t")
            self.tour_t = None if raw_tour is None else int(raw_tour)
            self.protocol = dict(header.get("protocol") or {})
            self.person_sensing = bool(header.get("person_sensing", False))
            self.resident_ids = tuple(
                str(r) for r in header.get("resident_ids", []))
        except (KeyError, TypeError, AttributeError) as err:
            raise BankFormatError(
                f"{path}:{lineno}: bad episode_header: {err}") from err
        if self.query_generation not in (None, "uniform", "routine_driven"):
            raise BankFormatError(
                f"{path}:{lineno}: query_generation "
                f"{self.query_generation!r} not in "
                f"('uniform', 'routine_driven')")
        unknown_roomed = set(self.receptacle_rooms) - set(self.receptacles)
        if unknown_roomed:
            raise BankFormatError(
                f"{path}:{lineno}: receptacle_rooms keys "
                f"{sorted(unknown_roomed)} not in receptacle_ids")
        if self.receptacle_rooms and self.home_base_room is None:
            raise BankFormatError(
                f"{path}:{lineno}: receptacle_rooms without home_base_room")
        if (self.home_base_room is not None
                and self.home_base_room not in set(
                    self.receptacle_rooms.values())):
            raise BankFormatError(
                f"{path}:{lineno}: home_base_room {self.home_base_room!r} "
                f"is not a room of any receptacle")
        if self.person_sensing and not self.resident_ids:
            raise BankFormatError(
                f"{path}:{lineno}: person_sensing without resident_ids")
        self._truth: Dict[str, List[Tuple[int, str]]] = {}
        self._carriers: Dict[Tuple[str, int], str] = {}
        self._resident_rooms: Dict[str, List[Tuple[int, str]]] = {}
        self._observations: List[Observation] = []
        self._questions: List[Question] = []
        self._room_visits: List[Tuple[int, Dict[str, List[str]]]] = []

    def _check(self, cond: bool, lineno: int, msg: str) -> None:
        if not cond:
            raise BankFormatError(
                f"{self._path}:{lineno} (episode {self.episode_id}): {msg}")

    def add(self, kind: str, row: Dict[str, Any], lineno: int) -> None:
        self._check(row.get("episode_id") == self.episode_id, lineno,
                    f"episode_id {row.get('episode_id')!r} does not match header")
        if kind == "truth":
            obj, rec, t = str(row["object_id"]), str(row["receptacle_id"]), int(row["t"])
            self._check(obj in self.object_classes, lineno, f"unknown object {obj!r}")
            self._check(rec in self.receptacles, lineno, f"unknown receptacle {rec!r}")
            self._truth.setdefault(obj, []).append((t, rec))
            carrier = row.get("carrier")
            if carrier is not None:
                self._check(rec == ON_PERSON, lineno,
                            f"carrier on a truth row at {rec!r} (only "
                            f"{ON_PERSON} rows carry one)")
                self._check(str(carrier) in self.resident_ids, lineno,
                            f"unknown carrier {carrier!r}")
                self._carriers[(obj, t)] = str(carrier)
        elif kind == "resident":
            res = str(row["resident_id"])
            self._check(res in self.resident_ids, lineno,
                        f"unknown resident {res!r}")
            room = str(row["room"])
            self._check(room == AWAY or room in set(self.receptacle_rooms.values()),
                        lineno, f"unknown room {room!r}")
            self._resident_rooms.setdefault(res, []).append(
                (int(row["t"]), room))
        elif kind == "observation":
            source = str(row["source"])
            self._check(source in _BANK_SOURCES, lineno,
                        f"bank observation source must be one of {_BANK_SOURCES}, "
                        f"got {source!r}")
            obj = str(row["object_id"])
            self._check(obj in self.object_classes, lineno, f"unknown object {obj!r}")
            rec = str(row["receptacle_id"])
            self._check(rec in self.receptacles, lineno, f"unknown receptacle {rec!r}")
            self._observations.append(Observation(
                object_id=obj, object_class=self.object_classes[obj],
                receptacle_id=rec, t=int(row["t"]), source=source))
        elif kind == "room_visit":
            contents = row["contents"]
            self._check(bool(isinstance(contents, dict) and contents), lineno,
                        "room_visit needs a non-empty contents mapping")
            for rec, objs in contents.items():
                self._check(str(rec) in self.receptacles, lineno,
                            f"unknown receptacle {rec!r}")
                for obj in objs:
                    self._check(str(obj) in self.object_classes, lineno,
                                f"unknown object {obj!r}")
            self._room_visits.append(
                (int(row["t"]), {str(r): [str(o) for o in objs]
                                 for r, objs in contents.items()}))
        else:  # question
            obj = str(row["object_id"])
            self._check(obj in self.object_classes, lineno, f"unknown object {obj!r}")
            self._check("origin" not in row or self.query_generation is not None,
                        lineno,
                        "question carries an origin but the episode header "
                        "declares no query_generation — a routine-driven "
                        "bank must say so in its header")
            row_class = row.get("object_class")
            object_class = (str(row_class) if row_class is not None
                            else self.object_classes.get(obj))
            self._check(object_class is not None, lineno,
                        f"question for {obj!r} carries no object_class and "
                        f"the header declares none")
            assert object_class is not None
            header_class = self.object_classes.get(obj)
            self._check(row_class is None or header_class is None
                        or object_class == header_class, lineno,
                        f"question object_class {object_class!r} contradicts "
                        f"header class {header_class!r} for {obj!r}")
            try:
                q = Question(
                    question_id=str(row["question_id"]), object_id=obj,
                    t_query=int(row["t_query"]), day_index=int(row["day_index"]),
                    object_class=object_class)
            except ValueError as err:
                raise BankFormatError(
                    f"{self._path}:{lineno} (episode {self.episode_id}): {err}"
                ) from err
            self._check(q.day_index < self.n_days, lineno,
                        f"question day_index {q.day_index} >= n_days {self.n_days}")
            self._questions.append(q)

    def finish(self) -> Episode:
        for obj in self.object_classes:
            traj = sorted(self._truth.get(obj, []))
            if not traj or traj[0][0] != 0:
                raise BankFormatError(
                    f"{self._path} (episode {self.episode_id}): object {obj!r} "
                    f"has no t=0 truth row")
        by_day: List[List[Question]] = [[] for _ in range(self.n_days)]
        for q in self._questions:
            by_day[q.day_index].append(q)
        initial = tuple(o for o in self._observations if o.source == "initial_tour")
        if self.tour_t is not None:
            stray = sorted({o.t for o in initial} - {self.tour_t})
            if stray:
                raise BankFormatError(
                    f"{self._path} (episode {self.episode_id}): initial_tour "
                    f"rows at t={stray} but header tour_t={self.tour_t}")
        # Room visits contribute their positive half to scripted_observations
        # (so recency readouts and the viewer keep working unchanged), and
        # their full evidence — one SenseResult per inspected receptacle,
        # emptiness included — to scripted_evidence, which is what belief
        # models are fed. A bank never mixes duplicated information: an
        # explicit "observation" row and a room_visit sighting are distinct
        # events, and room-visit exports write only the visit rows.
        visit_positives = tuple(
            Observation(object_id=obj, object_class=self.object_classes[obj],
                        receptacle_id=rec, t=t, source="scripted")
            for t, contents in self._room_visits
            for rec, objs in sorted(contents.items()) for obj in objs)
        scripted = tuple(sorted(
            (*(o for o in self._observations if o.source == "scripted"),
             *visit_positives), key=lambda o: o.t))
        explicit = tuple(o for o in self._observations
                         if o.source == "scripted")
        resident_rooms = {res: tuple(sorted(traj))
                          for res, traj in self._resident_rooms.items()}
        for res, traj in resident_rooms.items():
            if traj[0][0] != 0:
                raise BankFormatError(
                    f"{self._path} (episode {self.episode_id}): resident "
                    f"{res!r} has no t=0 room row")
        if self.person_sensing:
            missing = sorted(set(self.resident_ids) - set(resident_rooms))
            if missing:
                raise BankFormatError(
                    f"{self._path} (episode {self.episode_id}): "
                    f"person_sensing bank has no room rows for residents "
                    f"{missing}")
        # Under person_sensing an ambient visit also lists who was in the
        # room: the same presence listing a paid sense returns. Older
        # banks carry no roster and get the empty default.
        presence = _PresenceLookup(self.resident_ids, resident_rooms)
        visit_evidence = tuple(
            SenseResult(receptacle_id=rec, t=t, contents=tuple(objs),
                        object_classes={o: self.object_classes[o]
                                        for o in objs},
                        residents_present=(
                            presence.in_room(self.receptacle_rooms.get(rec), t)
                            if self.person_sensing else ()))
            for t, contents in sorted(self._room_visits,
                                      key=lambda v: v[0])
            for rec, objs in sorted(contents.items()))
        merged: List[Union[Observation, SenseResult]] = [*explicit,
                                                         *visit_evidence]
        merged.sort(key=lambda e: e.t)
        evidence = tuple(merged)
        try:
            episode = Episode(
                episode_id=self.episode_id, household_id=self.household_id,
                receptacle_ids=self.receptacles, object_classes=self.object_classes,
                initial_observations=initial, scripted_observations=scripted,
                questions_by_day=tuple(
                    tuple(sorted(day, key=lambda q: q.t_query)) for day in by_day),
                budget_per_day=self.budget_per_day,
                trajectories={obj: tuple(sorted(self._truth[obj]))
                              for obj in self.object_classes},
                household_type=self.household_type,
                premises=self.premises,
                unsensable_receptacle_ids=self.unsensable,
                receptacle_rooms=self.receptacle_rooms,
                home_base_room=self.home_base_room,
                scripted_evidence=evidence,
                person_sensing=self.person_sensing,
                resident_ids=self.resident_ids,
                resident_rooms=resident_rooms,
                carriers=dict(self._carriers),
                protocol=self.protocol)
        except ValueError as err:
            raise BankFormatError(
                f"{self._path} (episode {self.episode_id}): {err}") from err
        logger.debug("loaded episode %s: %d objects, %d questions",
                     episode.episode_id, len(episode.object_classes),
                     sum(len(d) for d in episode.questions_by_day))
        return episode


class _PresenceLookup:
    """Who is in which room at ``t``, from the resident room rows: the
    loader's own copy of :meth:`Episode.residents_in_room`, used before
    the Episode exists to stamp presence on ambient visits."""

    def __init__(self, resident_ids: Tuple[str, ...],
                 resident_rooms: Dict[str, Tuple[Tuple[int, str], ...]]
                 ) -> None:
        self._ids = resident_ids
        self._rooms = resident_rooms

    def room_of(self, resident_id: str, t: int) -> Union[str, None]:
        traj = self._rooms.get(resident_id)
        if not traj:
            return None
        idx = bisect.bisect_right([p[0] for p in traj], t) - 1
        room = traj[idx][1]
        return None if room == AWAY else room

    def in_room(self, room: Union[str, None], t: int) -> Tuple[str, ...]:
        if room is None:
            return ()
        return tuple(r for r in self._ids if self.room_of(r, t) == room)


# --------------------------------------------------------------------------
# Synthetic fixture bank
# --------------------------------------------------------------------------

_H = 3600
_FIXTURE_RECEPTACLES = ("counter_k", "desk_o", "entry_e", "shelf_l")
_TRAIN_DAYS = range(0, 4)
_QUESTION_DAYS = range(4, 7)
_FIXTURE_N_DAYS = 7
_FIXTURE_BUDGET = 2


def write_synthetic_bank(path: pathlib.Path) -> JsonlBank:
    """Write the deterministic three-object fixture bank and return its loader.

    Construction (all times seconds since episode start; days 0-3 provide
    scripted sightings, days 4-6 carry the questions):

    * ``mug_static`` never leaves ``counter_k``; observed only on the
      initial tour. Every belief should localize it perfectly.
    * ``keys_periodic`` follows a strict daily schedule — ``entry_e``
      overnight, ``desk_o`` 09:00-18:00 — with scripted sightings at 10:00
      (desk) and 20:00 (entry) on each train day. Question times alternate
      10:05 / 20:05, so beliefs that ignore time-of-day (frequency and
      recency alike) are right only at 20:05 — the periodic stress case.
    * ``laptop_mover`` sits on ``desk_o`` until noon of day 3, then moves
      permanently to ``shelf_l``; sightings at 09:00 on days 0-1 (desk) and
      14:00 on day 3 (shelf). Last-observation localizes it after the
      move; most-frequent stays wrong (2 desk sightings + tour vs 1 shelf).

    The file round-trips through :class:`JsonlBank` so loader and schema
    are exercised by every test that touches the fixture.
    """
    rows: List[Dict[str, Any]] = [{
        "kind": "episode_header", "episode_id": "synthetic_ep0",
        "household_id": "synthetic_hh", "receptacle_ids": list(_FIXTURE_RECEPTACLES),
        "object_classes": {"mug_static": "mug", "keys_periodic": "keys",
                           "laptop_mover": "laptop"},
        "budget_per_day": _FIXTURE_BUDGET, "n_days": _FIXTURE_N_DAYS,
    }]

    def truth(obj: str, t: int, rec: str) -> None:
        rows.append({"kind": "truth", "episode_id": "synthetic_ep0",
                     "object_id": obj, "t": t, "receptacle_id": rec})

    def obs(obj: str, t: int, rec: str, source: str) -> None:
        rows.append({"kind": "observation", "episode_id": "synthetic_ep0",
                     "object_id": obj, "t": t, "receptacle_id": rec,
                     "source": source})

    truth("mug_static", 0, "counter_k")
    truth("laptop_mover", 0, "desk_o")
    truth("laptop_mover", 3 * DAY_SECONDS + 12 * _H, "shelf_l")
    for d in range(_FIXTURE_N_DAYS):
        base = d * DAY_SECONDS
        truth("keys_periodic", base, "entry_e")
        truth("keys_periodic", base + 9 * _H, "desk_o")
        truth("keys_periodic", base + 18 * _H, "entry_e")

    obs("mug_static", 0, "counter_k", "initial_tour")
    obs("keys_periodic", 0, "entry_e", "initial_tour")
    obs("laptop_mover", 0, "desk_o", "initial_tour")
    for d in _TRAIN_DAYS:
        base = d * DAY_SECONDS
        obs("keys_periodic", base + 10 * _H, "desk_o", "scripted")
        obs("keys_periodic", base + 20 * _H, "entry_e", "scripted")
    obs("laptop_mover", 0 * DAY_SECONDS + 9 * _H, "desk_o", "scripted")
    obs("laptop_mover", 1 * DAY_SECONDS + 9 * _H, "desk_o", "scripted")
    obs("laptop_mover", 3 * DAY_SECONDS + 14 * _H, "shelf_l", "scripted")

    qn = 0
    for d in _QUESTION_DAYS:
        base = d * DAY_SECONDS
        for obj, t in (("keys_periodic", base + 10 * _H + 300),
                       ("mug_static", base + 11 * _H),
                       ("laptop_mover", base + 12 * _H + 300),
                       ("keys_periodic", base + 20 * _H + 300)):
            rows.append({"kind": "question", "episode_id": "synthetic_ep0",
                         "question_id": f"q{qn:03d}", "object_id": obj,
                         "t_query": t, "day_index": d})
            qn += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("wrote synthetic bank: %d rows -> %s", len(rows), path)
    return JsonlBank(path=path)


# --------------------------------------------------------------------------
# Negative-evidence fixture bank
# --------------------------------------------------------------------------

_NEG_RECEPTACLES = ("shelf_a", "shelf_b", "shelf_c", "shelf_d")


def write_negative_evidence_bank(path: pathlib.Path) -> JsonlBank:
    """A fixture where negative evidence is decisive (times in seconds).

    ``wallet_hidden`` is sighted three times at ``shelf_a`` (tour plus two
    scripted sightings on day 0) and then silently moves to ``shelf_c`` on
    day 1 at 22:00. Every basic belief therefore favors ``shelf_a`` at the
    day-2 question — recency and frequency directly, the timetable via its
    empty-bin fallback. A search that ignores negative evidence senses
    ``shelf_a`` (whose contents are non-empty: the static ``coin_decoy``
    is there), learns nothing, and answers ``shelf_a`` — wrong even at
    unlimited budget. With negative evidence, the fresh miss suppresses
    ``shelf_a`` fully and the search sweeps the remaining three
    receptacles, finding the wallet in at most four senses total.

    ``coin_decoy`` stays at ``shelf_a`` forever, so the sensed receptacle
    is never empty — the negative evidence must come from the wallet's
    absence, not from an empty result.
    """
    rows: List[Dict[str, Any]] = [{
        "kind": "episode_header", "episode_id": "negative_ep0",
        "household_id": "negative_hh",
        "receptacle_ids": list(_NEG_RECEPTACLES),
        "object_classes": {"wallet_hidden": "wallet", "coin_decoy": "coin"},
        "budget_per_day": 8, "n_days": 3,
    }]

    def add(kind: str, **fields: Any) -> None:
        rows.append({"kind": kind, "episode_id": "negative_ep0", **fields})

    add("truth", object_id="coin_decoy", t=0, receptacle_id="shelf_a")
    add("truth", object_id="wallet_hidden", t=0, receptacle_id="shelf_a")
    add("truth", object_id="wallet_hidden", t=DAY_SECONDS + 22 * 3600,
        receptacle_id="shelf_c")
    for obj in ("wallet_hidden", "coin_decoy"):
        add("observation", object_id=obj, t=0, receptacle_id="shelf_a",
            source="initial_tour")
    for t in (10 * 3600, 18 * 3600):
        add("observation", object_id="wallet_hidden", t=t,
            receptacle_id="shelf_a", source="scripted")
    add("question", question_id="q_wallet", object_id="wallet_hidden",
        t_query=2 * DAY_SECONDS + 12 * 3600, day_index=2)
    add("question", question_id="q_coin", object_id="coin_decoy",
        t_query=2 * DAY_SECONDS + 13 * 3600, day_index=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("wrote negative-evidence bank: %d rows -> %s", len(rows), path)
    return JsonlBank(path=path)


# --------------------------------------------------------------------------
# Healthcheck gate-test banks
# --------------------------------------------------------------------------

_PASS_RECEPTACLES = ("kitchen_c", "desk_d", "entry_e", "shelf_s",
                     "sofa_f", "drawer_w", "bath_b", "bin_n")
_PASS_N_DAYS = 14
_PASS_TRAIN_DAYS = range(0, 4)
_PASS_QUESTION_DAYS = range(4, 14)
_PASS_BUDGET = 24


def _loc_at(trajectory: List[Tuple[int, str]], t: int) -> str:
    """Receptacle at time ``t`` for a sorted change-point list (t0 == 0)."""
    location = trajectory[0][1]
    for change_t, receptacle in trajectory:
        if change_t > t:
            break
        location = receptacle
    return location


def _pass_bank_trajectories(
        rng: random.Random) -> Dict[str, List[Tuple[int, str]]]:
    """Ground truth for the PASS bank's four dynamics families:
    static / strictly periodic / nightly drifter / fast roamer."""
    h = 3600
    trajectories: Dict[str, List[Tuple[int, str]]] = {
        "mug_anchor": [(0, "kitchen_c")]}
    for obj in ("keys_shift", "badge_shift"):
        traj: List[Tuple[int, str]] = []
        for d in range(_PASS_N_DAYS):
            base = d * DAY_SECONDS
            traj += [(base, "entry_e"), (base + 9 * h, "desk_d"),
                     (base + 18 * h, "entry_e")]
        trajectories[obj] = traj
    for obj in ("drift_a", "drift_b", "drift_c"):
        traj = [(0, rng.choice(_PASS_RECEPTACLES))]
        for d in range(_PASS_N_DAYS):
            traj.append((d * DAY_SECONDS + 21 * h,
                         rng.choice(_PASS_RECEPTACLES)))
        trajectories[obj] = traj
    roam = [(0, rng.choice(_PASS_RECEPTACLES))]
    for d in range(_PASS_N_DAYS):
        for move_hour in (5, 11, 17, 23):
            roam.append((d * DAY_SECONDS + move_hour * h,
                         rng.choice(_PASS_RECEPTACLES)))
    trajectories["roam_fast"] = roam
    return trajectories


def _pass_bank_observations(
        trajectories: Dict[str, List[Tuple[int, str]]]
        ) -> List[Tuple[str, int, str, str]]:
    """(object, t, receptacle, source) rows: tour at t=0 for everything,
    then scripted sightings per family (periodic objects only on train
    days, so recency stays stuck; drifters every even day at noon; the
    roamer daily at 07:00)."""
    h = 3600
    obs: List[Tuple[str, int, str, str]] = [
        (obj, 0, _loc_at(traj, 0), "initial_tour")
        for obj, traj in trajectories.items()]
    for obj in ("keys_shift", "badge_shift"):
        for d in _PASS_TRAIN_DAYS:
            base = d * DAY_SECONDS
            obs.append((obj, base + 10 * h, "desk_d", "scripted"))
            obs.append((obj, base + 20 * h, "entry_e", "scripted"))
    for obj in ("drift_a", "drift_b", "drift_c"):
        for d in range(0, _PASS_N_DAYS, 2):
            t = d * DAY_SECONDS + 12 * h
            obs.append((obj, t, _loc_at(trajectories[obj], t), "scripted"))
    for d in range(_PASS_N_DAYS):
        t = d * DAY_SECONDS + 7 * h
        obs.append(("roam_fast", t, _loc_at(trajectories["roam_fast"], t),
                    "scripted"))
    return obs


_PASS_DAILY_QUESTIONS: Tuple[Tuple[str, int], ...] = (
    # (object, seconds into the day); 31 per day x 10 question days = 310.
    # The mix is weighted toward the drifters so the passive ceiling stays
    # under the not_trivial gate even for the decayed (honest-strong)
    # frequency/timetable beliefs, which nearly solve the periodic pair.
    ("mug_anchor", 11 * 3600), ("mug_anchor", 14 * 3600),
    ("mug_anchor", 16 * 3600),
    ("keys_shift", 10 * 3600 + 900), ("keys_shift", 20 * 3600 + 2700),
    ("badge_shift", 10 * 3600 + 1200), ("badge_shift", 20 * 3600 + 3000),
    ("drift_a", 13 * 3600 + 1800), ("drift_a", 14 * 3600 + 900),
    ("drift_a", 15 * 3600), ("drift_a", 15 * 3600 + 1800),
    ("drift_a", 16 * 3600 + 900), ("drift_a", 16 * 3600 + 1800),
    ("drift_a", 17 * 3600 + 600),
    ("drift_b", 13 * 3600 + 2100), ("drift_b", 14 * 3600 + 1200),
    ("drift_b", 15 * 3600 + 300), ("drift_b", 15 * 3600 + 2100),
    ("drift_b", 16 * 3600 + 1200), ("drift_b", 16 * 3600 + 2100),
    ("drift_b", 17 * 3600 + 900),
    ("drift_c", 13 * 3600 + 2400), ("drift_c", 14 * 3600 + 1500),
    ("drift_c", 15 * 3600 + 600), ("drift_c", 15 * 3600 + 2400),
    ("drift_c", 16 * 3600 + 1500), ("drift_c", 16 * 3600 + 2400),
    ("drift_c", 17 * 3600 + 1200),
    ("roam_fast", 15 * 3600 + 1200), ("roam_fast", 18 * 3600 + 1800),
    ("roam_fast", 19 * 3600 + 600),
)


def write_gate_pass_bank(path: pathlib.Path, seed: int = 0) -> JsonlBank:
    """A bank engineered to PASS all five healthcheck gates.

    Dynamics families and their intended NeverSense profiles (derivation
    mirrors ``write_synthetic_bank``; exact values are seeded-random but
    stable): a static anchor (everyone 1.0), two strictly periodic objects
    whose sightings stop after day 3 (timetable 1.0, recency/frequency
    0.5), three nightly drifters sighted every other noon (recency ~0.55,
    frequency/timetable far worse), and a four-moves-a-day roamer nobody
    tracks. Weighted by the daily question mix this lands every belief
    under the not_trivial ceiling with clear spread (discriminative), 310
    questions (powered), and a budget generous enough that search buys
    accuracy (not_impossible). Solvable holds for any well-formed bank.

    ``household_type`` metadata is present so the stratified
    discriminative path is exercised.
    """
    rng = random.Random(seed)
    trajectories = _pass_bank_trajectories(rng)
    rows: List[Dict[str, Any]] = [{
        "kind": "episode_header", "episode_id": f"gate_pass_ep{seed}",
        "household_id": "gate_pass_hh", "household_type": "synthetic_mixed",
        "receptacle_ids": list(_PASS_RECEPTACLES),
        "object_classes": {
            "mug_anchor": "mug", "keys_shift": "keys",
            "badge_shift": "badge", "drift_a": "tote", "drift_b": "tote",
            "drift_c": "tote", "roam_fast": "toy"},
        "budget_per_day": _PASS_BUDGET, "n_days": _PASS_N_DAYS,
    }]
    episode_id = f"gate_pass_ep{seed}"
    for obj, traj in trajectories.items():
        for t, rec in traj:
            rows.append({"kind": "truth", "episode_id": episode_id,
                         "object_id": obj, "t": t, "receptacle_id": rec})
    for obj, t, rec, source in _pass_bank_observations(trajectories):
        rows.append({"kind": "observation", "episode_id": episode_id,
                     "object_id": obj, "t": t, "receptacle_id": rec,
                     "source": source})
    qn = 0
    for d in _PASS_QUESTION_DAYS:
        for obj, offset in _PASS_DAILY_QUESTIONS:
            rows.append({"kind": "question", "episode_id": episode_id,
                         "question_id": f"q{qn:04d}", "object_id": obj,
                         "t_query": d * DAY_SECONDS + offset,
                         "day_index": d})
            qn += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("wrote gate-pass bank: %d rows -> %s", len(rows), path)
    return JsonlBank(path=path)


_FAIL_RECEPTACLES = ("kitchen_c", "desk_d", "entry_e",
                     "shelf_s", "sofa_f", "drawer_w")


def write_gate_fail_static_bank(path: pathlib.Path) -> JsonlBank:
    """A static-world bank engineered to FAIL the not_trivial gate.

    Six objects never move and the initial tour reveals all of them, so
    every belief model answers every question correctly without sensing:
    NeverSense accuracy is 1.0, far above the not_trivial ceiling (and the
    spread is exactly 0, so discriminative fails too — a static world
    cannot rank modeling assumptions). 300 questions keep the powered
    gate out of the way; the failure is about dynamics, not scale. No
    ``household_type`` metadata, so the stratified check reports SKIPPED.
    """
    objects = {f"item_{i}": "widget" for i in range(6)}
    episode_id = "gate_fail_ep0"
    rows: List[Dict[str, Any]] = [{
        "kind": "episode_header", "episode_id": episode_id,
        "household_id": "gate_fail_hh",
        "receptacle_ids": list(_FAIL_RECEPTACLES),
        "object_classes": objects, "budget_per_day": 4, "n_days": 12,
    }]
    for i, obj in enumerate(objects):
        home = _FAIL_RECEPTACLES[i % len(_FAIL_RECEPTACLES)]
        rows.append({"kind": "truth", "episode_id": episode_id,
                     "object_id": obj, "t": 0, "receptacle_id": home})
        rows.append({"kind": "observation", "episode_id": episode_id,
                     "object_id": obj, "t": 0, "receptacle_id": home,
                     "source": "initial_tour"})
    qn = 0
    for d in range(2, 12):
        for k in range(30):
            obj = f"item_{k % len(objects)}"
            rows.append({"kind": "question", "episode_id": episode_id,
                         "question_id": f"q{qn:04d}", "object_id": obj,
                         "t_query": d * DAY_SECONDS + (8 + k // 2) * 1800
                         + 8 * 3600, "day_index": d})
            qn += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("wrote gate-fail bank: %d rows -> %s", len(rows), path)
    return JsonlBank(path=path)


# --------------------------------------------------------------------------
# Passive-protocol fixture banks (analytically known answers)
# --------------------------------------------------------------------------

_PROBE_RECEPTACLES = ("entry_e", "desk_d", "shelf_s")


def write_periodic_probe_bank(path: pathlib.Path) -> JsonlBank:
    """A strictly periodic object with dense sightings: the analytic case
    where time-conditioned models must be perfect at short horizons.

    ``badge_shift`` follows the same schedule every one of 14 days —
    ``entry_e`` overnight, ``desk_d`` 09:00-18:00 — and is sighted every
    day at 10:00 (desk) and 20:00 (entry). Questions on days 7-9 at 10:30
    and 20:30. At a checkpoint with several observed days of history and
    horizons <= 1 day, ``timetable`` answers from the exactly-matching
    time-of-day bin and ``periodic_persistence`` puts its post-hazard
    remainder on that bin's receptacle, so BOTH must score 1.0 (asserted
    in tests). ``mug_static`` never moves (tour only); ``shelf_s`` stays
    empty so wrong mass has somewhere to sit.
    """
    h, n_days, episode_id = 3600, 14, "periodic_probe_ep0"
    rows: List[Dict[str, Any]] = [{
        "kind": "episode_header", "episode_id": episode_id,
        "household_id": "periodic_probe_hh",
        "receptacle_ids": list(_PROBE_RECEPTACLES),
        "object_classes": {"badge_shift": "badge", "mug_static": "mug"},
        "budget_per_day": 2, "n_days": n_days}]

    def add(kind: str, **fields: Any) -> None:
        rows.append({"kind": kind, "episode_id": episode_id, **fields})

    add("truth", object_id="mug_static", t=0, receptacle_id="shelf_s")
    add("observation", object_id="mug_static", t=0,
        receptacle_id="shelf_s", source="initial_tour")
    add("observation", object_id="badge_shift", t=0,
        receptacle_id="entry_e", source="initial_tour")
    for d in range(n_days):
        base = d * DAY_SECONDS
        add("truth", object_id="badge_shift", t=base,
            receptacle_id="entry_e")
        add("truth", object_id="badge_shift", t=base + 9 * h,
            receptacle_id="desk_d")
        add("truth", object_id="badge_shift", t=base + 18 * h,
            receptacle_id="entry_e")
        add("observation", object_id="badge_shift", t=base + 10 * h,
            receptacle_id="desk_d", source="scripted")
        add("observation", object_id="badge_shift", t=base + 20 * h,
            receptacle_id="entry_e", source="scripted")
    qn = 0
    for d in range(7, 10):
        for offset in (10 * h + 1800, 20 * h + 1800):
            add("question", question_id=f"q{qn:03d}",
                object_id="badge_shift", t_query=d * DAY_SECONDS + offset,
                day_index=d)
            qn += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return JsonlBank(path=path)


_REGIME_RECEPTACLES = ("desk_d", "couch_c", "shelf_s")
_REGIME_OBJECTS = ("laptop_1", "book_1", "headphones_1", "charger_1")


def write_two_regime_bank(path: pathlib.Path) -> JsonlBank:
    """A weekday/weekend household: regime inference is the whole game.

    Four objects sit on ``desk_d`` all day on weekdays and on ``couch_c``
    all weekend (day 0 = Monday; weekends are day % 7 in {5, 6}), for 28
    days, each sighted daily at noon. Questions are on days 26 (Saturday)
    and 27 (Sunday). At a checkpoint frozen before day 26, a
    24 h-half-life frequency model is dominated by Friday's desk
    sightings and answers desk (wrong all weekend), while a day-type
    model that clusters weekend days together and reads the query's
    day-of-week must answer couch — the margin a test asserts.
    """
    h, n_days, episode_id = 3600, 28, "two_regime_ep0"
    rows: List[Dict[str, Any]] = [{
        "kind": "episode_header", "episode_id": episode_id,
        "household_id": "two_regime_hh",
        "receptacle_ids": list(_REGIME_RECEPTACLES),
        "object_classes": {o: o.rsplit("_", 1)[0] for o in _REGIME_OBJECTS},
        "budget_per_day": 2, "n_days": n_days}]

    def add(kind: str, **fields: Any) -> None:
        rows.append({"kind": kind, "episode_id": episode_id, **fields})

    for obj in _REGIME_OBJECTS:
        add("observation", object_id=obj, t=0, receptacle_id="desk_d",
            source="initial_tour")
        for d in range(n_days):
            where = "couch_c" if d % 7 in (5, 6) else "desk_d"
            add("truth", object_id=obj, t=d * DAY_SECONDS if d else 0,
                receptacle_id=where)
            add("observation", object_id=obj, t=d * DAY_SECONDS + 12 * h,
                receptacle_id=where, source="scripted")
    qn = 0
    for d in (26, 27):
        for i, obj in enumerate(_REGIME_OBJECTS):
            add("question", question_id=f"q{qn:03d}", object_id=obj,
                t_query=d * DAY_SECONDS + (10 + i) * h, day_index=d)
            qn += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return JsonlBank(path=path)


_CHURN_RECEPTACLES = ("bin_a", "bin_b", "bin_c", "bin_d")


def write_fast_churn_bank(path: pathlib.Path, seed: int = 7) -> JsonlBank:
    """An object that outruns every model: seeded aperiodic churn.

    ``toy_churn`` jumps to a uniformly drawn receptacle every 3 h for 10
    days (seeded, so the bank is reproducible but carries no learnable
    period). Sightings are dense on days 0-2 (8/day) and absent
    afterwards; questions cover days 3-9, so time-since-last-sighting
    grows to a week. As it grows, no model can beat the ~1/4 frequency
    floor — a test asserts every model converges to the frequency
    model's stale-bin accuracy.
    """
    h, n_days, episode_id = 3600, 10, f"fast_churn_ep{seed}"
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = [{
        "kind": "episode_header", "episode_id": episode_id,
        "household_id": "fast_churn_hh",
        "receptacle_ids": list(_CHURN_RECEPTACLES),
        "object_classes": {"toy_churn": "toy"},
        "budget_per_day": 2, "n_days": n_days}]
    trajectory: List[Tuple[int, str]] = [(0, rng.choice(_CHURN_RECEPTACLES))]
    for t in range(3 * h, n_days * DAY_SECONDS, 3 * h):
        trajectory.append((t, rng.choice(_CHURN_RECEPTACLES)))

    def add(kind: str, **fields: Any) -> None:
        rows.append({"kind": kind, "episode_id": episode_id, **fields})

    for t, rec in trajectory:
        add("truth", object_id="toy_churn", t=t, receptacle_id=rec)
    add("observation", object_id="toy_churn", t=0,
        receptacle_id=trajectory[0][1], source="initial_tour")
    for d in range(3):
        for k in range(8):
            t = d * DAY_SECONDS + (2 + 2 * k) * h + 900
            add("observation", object_id="toy_churn", t=t,
                receptacle_id=_loc_at(trajectory, t), source="scripted")
    qn = 0
    for d in range(3, n_days):
        for k in range(6):
            add("question", question_id=f"q{qn:03d}",
                object_id="toy_churn",
                t_query=d * DAY_SECONDS + (3 * k + 5) * h + 600,
                day_index=d)
            qn += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return JsonlBank(path=path)


# --------------------------------------------------------------------------
# Room-change-cost fixture bank
# --------------------------------------------------------------------------

_ROOM_COST_ROOMS: Dict[str, str] = {
    "counter_k": "kitchen", "sink_k": "kitchen", "table_k": "kitchen",
    "desk_s": "study",
    "hook_h": "hall",
}
"""Receptacle -> room for :func:`write_room_cost_bank`. The kitchen holds
three receptacles, so it is the home base by the most-receptacles rule."""


def write_room_cost_bank(path: pathlib.Path) -> JsonlBank:
    """A fixture with rooms, a patrol, and hand-derivable travel costs.

    Layout: kitchen (``counter_k``, ``sink_k``, ``table_k``), study
    (``desk_s``), hall (``hook_h``); the kitchen is therefore the home
    base. ``mug_k`` sits on ``counter_k`` forever and ``keys_s`` on
    ``desk_s`` forever, so truth is trivial and every assertion is about
    position and cost rather than about belief dynamics.

    Two days, two questions each at 12:00 and 13:00. One room visit per
    day at 11:00: to the STUDY on day 0 and to the HALL on day 1. So at
    the day-0 12:00 question the robot stands in the study (a passive
    visit moved it off the home base), and at the day-1 12:00 question in
    the hall — while at the instant each day BEGINS it is back in the
    kitchen, which is what makes the day-start reset observable.
    """
    h, episode_id = 3600, "room_cost_ep0"
    receptacles = [*_ROOM_COST_ROOMS, "OUT_OF_HOUSE"]
    rows: List[Dict[str, Any]] = [{
        "kind": "episode_header", "episode_id": episode_id,
        "household_id": "room_cost_hh", "receptacle_ids": receptacles,
        "object_classes": {"mug_k": "mug", "keys_s": "keys"},
        "budget_per_day": 4, "n_days": 2,
        "unsensable_receptacles": ["OUT_OF_HOUSE"],
        "receptacle_rooms": dict(_ROOM_COST_ROOMS),
        "home_base_room": "kitchen"}]

    def add(kind: str, **fields: Any) -> None:
        rows.append({"kind": kind, "episode_id": episode_id, **fields})

    add("truth", object_id="mug_k", t=0, receptacle_id="counter_k")
    add("truth", object_id="keys_s", t=0, receptacle_id="desk_s")
    add("observation", object_id="mug_k", t=0, receptacle_id="counter_k",
        source="initial_tour")
    add("observation", object_id="keys_s", t=0, receptacle_id="desk_s",
        source="initial_tour")
    add("room_visit", t=11 * h, room="study", contents={"desk_s": ["keys_s"]})
    add("room_visit", t=DAY_SECONDS + 11 * h, room="hall",
        contents={"hook_h": []})
    qn = 0
    for d in range(2):
        for obj, hour in (("mug_k", 12), ("keys_s", 13)):
            add("question", question_id=f"q{qn:03d}", object_id=obj,
                t_query=d * DAY_SECONDS + hour * h, day_index=d)
            qn += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("wrote room-cost bank: %d rows -> %s", len(rows), path)
    return JsonlBank(path=path)

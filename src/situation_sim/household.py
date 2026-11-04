"""Household sampler: residents, rooms, spots, objects, traits, habits, groups.

Everything is sampled from code with one seeded ``random.Random``, driven
by the class catalogue in ``objects.yaml`` and the habit library in
``activities.yaml``. No LLM authoring and no per-household YAML. Ids follow
the existing bank conventions (``counter_k1``, ``nightstand_b1``,
``laptop_marco``, ``ON_PERSON``, ``OUT_OF_HOUSE``).
"""
from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import yaml

from situation_sim import timing_constants as tc

ON_PERSON = "ON_PERSON"
OUT_OF_HOUSE = "OUT_OF_HOUSE"

# Spot kinds per room kind: (kind, capacity). Suffix letters follow the
# old banks (k kitchen, l living, e entry, ba bathroom, b bedroom, o office,
# d dining) plus s storage and y balcony.
ROOM_SPOTS = {
    "kitchen": ("k", [("counter", 10), ("sink", 8), ("cupboard", 16), ("dish_rack", 8),
                      ("kitchen_table", 10), ("chair", 3), ("drawer_k", 8),
                      ("pantry_shelf", 10), ("floor_k", 12)]),
    "living": ("l", [("couch", 6), ("coffee_table", 8), ("tv_stand", 5), ("bookshelf", 14),
                     ("armchair", 3), ("side_table", 4), ("floor_l", 12)]),
    "entry": ("e", [("entry_table", 8), ("entry_hook", 6), ("shoe_rack", 8), ("entry_floor", 12)]),
    "bathroom": ("ba", [("bathroom_shelf", 8), ("towel_rack", 4), ("medicine_cabinet", 8),
                        ("sink_ba", 4)]),
    "bedroom": ("b", [("bed", 5), ("nightstand", 5), ("desk", 8), ("dresser", 8),
                      ("wardrobe", 14), ("bedroom_floor", 12)]),
    "office": ("o", [("desk", 8), ("office_shelf", 12), ("office_chair", 3), ("floor_o", 12)]),
    "dining": ("d", [("dining_table", 10), ("sideboard", 8), ("chair_d", 3)]),
    "storage": ("s", [("storage_shelf", 14), ("storage_floor", 14)]),
    "balcony": ("y", [("balcony_table", 6), ("balcony_floor", 10)]),
}

FIRST_NAMES = ["marco", "priya", "elena", "tomas", "aisha", "leo", "nora",
               "sam", "ines", "kwame", "yuki", "dana", "omar", "hana", "felix", "zara"]

ROLES = ["worker_out", "worker_home", "retired", "student", "shift_worker"]
ROLE_WEIGHTS = [0.4, 0.25, 0.12, 0.13, 0.10]
FORGET_LEVELS = ["rarely", "sometimes", "often"]

_CATALOGUE: Optional[dict] = None


def catalogue() -> dict:
    global _CATALOGUE
    if _CATALOGUE is None:
        with open(pathlib.Path(__file__).with_name("objects.yaml")) as f:
            _CATALOGUE = yaml.safe_load(f)
    return _CATALOGUE


def size_of(cls: str) -> str:
    return catalogue()["classes"][cls]["size"]


def accepts(spot_kind: str, cls: str) -> bool:
    """May an object of class ``cls`` sit on a spot of kind ``spot_kind``?"""
    cat = catalogue()
    if cls in cat["class_overrides"] and spot_kind in cat["class_overrides"][cls]:
        return True
    ok = cat["size_accepts"].get(spot_kind)
    if ok is None:
        return True
    return "*" in ok or size_of(cls) in ok


@dataclass
class Receptacle:
    id: str
    kind: str
    room: str
    capacity: int


@dataclass
class Resident:
    id: str
    name: str
    role: str
    bedroom: str
    workspace: str
    tidiness: float
    jitter_scale: float
    forget_level: str
    forget_p: float
    mood_sensitivity: float
    base_energy: float
    base_hurry: float
    base_distraction: float
    hobbies: List[str] = field(default_factory=list)
    chores: List[str] = field(default_factory=list)


@dataclass
class Obj:
    id: str
    cls: str
    owner: Optional[str]
    home: List[str]
    size: str = "small"
    pocket: bool = False
    outdoor: bool = False
    rain_only: bool = False
    static: bool = False
    group: Optional[str] = None
    after_use: Optional[str] = None
    allowed: List[str] = field(default_factory=list)


@dataclass
class Group:
    name: str
    kind: str
    owner: str
    leader: str
    members: List[str]


@dataclass
class Household:
    id: str
    seed: int
    household_type: str
    rooms: List[str]
    receptacles: Dict[str, Receptacle]
    residents: Dict[str, Resident]
    objects: Dict[str, Obj]
    groups: Dict[str, Group]
    pet: Optional[str] = None
    carer: Optional[str] = None

    def recs_in(self, room: str) -> List[Receptacle]:
        return sorted((r for r in self.receptacles.values() if r.room == room), key=lambda r: r.id)

    def rec_of_kind(self, kind: str, room: Optional[str] = None) -> Optional[str]:
        cands = sorted(r.id for r in self.receptacles.values()
                       if r.kind == kind and (room is None or r.room == room))
        return cands[0] if cands else None

    def has_room(self, kind: str) -> Optional[str]:
        rooms = sorted(r for r in self.rooms if r.split("_")[0] == kind)
        return rooms[0] if rooms else None

    def objects_of(self, resident: Optional[str]) -> List[Obj]:
        return sorted((o for o in self.objects.values() if o.owner == resident), key=lambda o: o.id)

    def resolve_room(self, token: str, res: Optional[Resident]) -> Optional[str]:
        bedroom = res.bedroom if res else "bedroom_1"
        workspace = res.workspace if res else (self.has_room("office") or "bedroom_1")
        table = {
            "bedroom": bedroom, "workspace": workspace,
            "kitchen": "kitchen", "living": "living", "bathroom": "bathroom", "entry": "entry",
            "dining_or_kitchen": self.has_room("dining") or "kitchen",
            "storage_or_bedroom": self.has_room("storage") or bedroom,
            "balcony_or_living": self.has_room("balcony") or "living",
            "office_or_living": self.has_room("office") or "living",
        }
        room = table.get(token, token)
        return room if room in self.rooms else None

    def resolve_home(self, tokens: List[str], res: Optional[Resident]) -> List[str]:
        out: List[str] = []
        for tok in tokens:
            kind, _, roomtok = tok.partition("@")
            room = self.resolve_room(roomtok, res) if roomtok else None
            if roomtok and room is None:
                continue
            rec = self.rec_of_kind(kind, room)
            if rec and rec not in out:
                out.append(rec)
        return out

    def to_json(self) -> dict:
        return {
            "id": self.id, "seed": self.seed, "household_type": self.household_type,
            "pet": self.pet, "carer": self.carer, "rooms": list(self.rooms),
            "receptacles": {k: asdict(v) for k, v in sorted(self.receptacles.items())},
            "residents": {k: asdict(v) for k, v in sorted(self.residents.items())},
            "objects": {k: asdict(v) for k, v in sorted(self.objects.items())},
            "groups": {k: asdict(v) for k, v in sorted(self.groups.items())},
        }


def _u(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 3)


def sample_household(seed: int, acts: dict, hh_id: Optional[str] = None) -> Household:
    rng = random.Random(f"household:{seed}")
    cat = catalogue()
    hh_id = hh_id or f"hh_s{seed}"

    # --- residents and rooms -------------------------------------------
    n_res = rng.choice([2, 2, 3])
    household_type = (rng.choice(["couple", "couple", "flatmates"]) if n_res == 2
                      else rng.choice(["couple_plus_one", "flatmates"]))
    names = rng.sample(FIRST_NAMES, n_res)
    rooms = ["kitchen", "living", "entry", "bathroom"]
    if household_type == "couple":
        bedrooms, res_bedroom = ["bedroom_1"], ["bedroom_1", "bedroom_1"]
    elif household_type == "couple_plus_one":
        bedrooms, res_bedroom = ["bedroom_1", "bedroom_2"], ["bedroom_1", "bedroom_1", "bedroom_2"]
    else:
        bedrooms = [f"bedroom_{i + 1}" for i in range(n_res)]
        res_bedroom = list(bedrooms)
    rooms += bedrooms
    for opt, p in (("office", 0.5), ("dining", 0.4), ("balcony", 0.5), ("storage", 0.4)):
        if rng.random() < p:
            rooms.append(opt)
    rooms = sorted(rooms)

    receptacles: Dict[str, Receptacle] = {}
    for room in rooms:
        kind = room.split("_")[0]
        suffix, spots = ROOM_SPOTS[kind]
        idx = room.split("_")[1] if "_" in room else "1"
        for skind, cap in spots:
            rid = f"{skind}_{suffix}{idx}"
            receptacles[rid] = Receptacle(rid, skind, room, cap)

    roles = [rng.choices(ROLES, weights=ROLE_WEIGHTS)[0] for _ in range(n_res)]
    if all(r == "retired" for r in roles):
        roles[0] = "worker_out"
    office = "office" if "office" in rooms else None
    residents: Dict[str, Resident] = {}
    for i, name in enumerate(names):
        rid = f"resident_{i + 1}"
        fl = rng.choices(FORGET_LEVELS, weights=[0.4, 0.4, 0.2])[0]
        residents[rid] = Resident(
            id=rid, name=name, role=roles[i], bedroom=res_bedroom[i],
            workspace=(office if office and roles[i] in ("worker_home", "student") else res_bedroom[i]),
            tidiness=_u(rng, 0.25, 0.95),
            jitter_scale=round(min(tc.JITTER_SCALE_MAX, max(tc.JITTER_SCALE_MIN, rng.lognormvariate(0.0, 0.3))), 3),
            forget_level=fl, forget_p=tc.FORGET_LEVELS[fl],
            mood_sensitivity=_u(rng, 0.2, 1.0),
            base_energy=_u(rng, 0.35, 0.75), base_hurry=_u(rng, 0.25, 0.65),
            base_distraction=_u(rng, 0.2, 0.6),
        )
    seen_office = False
    for r in sorted(residents.values(), key=lambda r: r.id):
        if r.workspace == "office":
            if seen_office:
                r.workspace = r.bedroom
            seen_office = True

    hh = Household(hh_id, seed, household_type, rooms, receptacles, residents, {}, {})

    # --- habits -----------------------------------------------------------
    hobbies = acts["habits"]["hobbies"]
    chores = acts["habits"]["chores"]
    for r in sorted(residents.values(), key=lambda r: r.id):
        have = [h for h in sorted(hobbies) if rng.random() < hobbies[h]["p_have"]
                and r.role in hobbies[h].get("roles", [r.role])]
        rng.shuffle(have)
        r.hobbies = sorted(have[:rng.choice([2, 3, 3, 4])])
        r.chores = sorted(c for c in sorted(chores) if rng.random() < chores[c]["p_have"])
    # pet
    pet_cfg = acts["habits"]["pet"]["dog"]
    if rng.random() < pet_cfg["p_have"]:
        hh.pet = "dog"
        hh.carer = sorted(residents)[rng.randrange(n_res)]

    # --- objects ---------------------------------------------------------
    objects: Dict[str, Obj] = {}
    groups: Dict[str, Group] = {}
    any_hobby = sorted({h for r in residents.values() for h in r.hobbies})

    def make(cls: str, spec: dict, owner: Optional[str], oid: str) -> Optional[Obj]:
        res = residents[owner] if owner else None
        home = hh.resolve_home(spec["home"], res)
        if not home:
            return None
        o = Obj(oid, cls, owner, home, size=spec["size"], pocket=bool(spec.get("pocket")),
                outdoor=bool(spec.get("outdoor")), rain_only=bool(spec.get("rain_only")),
                static=bool(spec.get("static")))
        if spec.get("after_use"):
            au = hh.rec_of_kind(spec["after_use"], hh.receptacles[home[0]].room) or hh.rec_of_kind(spec["after_use"])
            o.after_use = au
        objects[oid] = o
        return o

    for cls in sorted(cat["classes"]):
        spec = cat["classes"][cls]
        if spec["per"] == "resident":
            for rid, res in sorted(residents.items()):
                if spec.get("roles") and res.role not in spec["roles"]:
                    continue
                if spec.get("hobby") and spec["hobby"] not in res.hobbies:
                    continue
                if rng.random() >= spec["p"]:
                    continue
                make(cls, spec, rid, f"{cls}_{res.name}")
        else:
            if spec.get("hobby") and spec["hobby"] not in any_hobby:
                continue
            if spec.get("pet") and hh.pet != spec["pet"]:
                continue
            if rng.random() >= spec["p"]:
                continue
            n = int(spec.get("count", 1))
            for k in range(n):
                oid = f"{cls}_shared" if n == 1 else f"{cls}_{k + 1}_shared"
                o = make(cls, spec, None, oid)
                if o and n > 1 and len(o.home) > 1:
                    # spread duplicates over their home options
                    o.home = o.home[k % len(o.home):] + o.home[:k % len(o.home)]

    # --- groups: bags and gym bags -----------------------------------------
    for rid, res in sorted(residents.items()):
        own = [o for o in objects.values() if o.owner == rid]
        leaders = sorted(o.id for o in own if cat["classes"][o.cls].get("bag_leader"))
        if not leaders and res.role in ("worker_out", "student", "shift_worker"):
            o = make("backpack", cat["classes"]["backpack"], rid, f"backpack_{res.name}")
            if o:
                leaders = [o.id]
        if leaders:
            leader = leaders[0]
            for extra in leaders[1:]:          # one bag per person is enough
                del objects[extra]
            cands = sorted(o.id for o in own if cat["classes"][o.cls].get("bag_candidate") and o.id in objects)
            members = sorted(m for m in cands if rng.random() < 0.75)
            g = Group(f"bag_{res.name}", "bag", rid, leader, members)
            groups[g.name] = g
            objects[leader].group = g.name
            for m in members:
                objects[m].group = g.name
        gyms = sorted(o.id for o in own if cat["classes"][o.cls].get("gym_leader") and o.id in objects)
        if gyms:
            cands = sorted(o.id for o in own if cat["classes"][o.cls].get("gym_candidate")
                           and o.id in objects and objects[o.id].group is None)
            members = sorted(m for m in cands if rng.random() < 0.6)
            g = Group(f"gym_{res.name}", "gym", rid, gyms[0], members)
            groups[g.name] = g
            objects[gyms[0]].group = g.name
            for m in members:
                objects[m].group = g.name

    hh.objects = dict(sorted(objects.items()))
    hh.groups = dict(sorted(groups.items()))
    return hh

"""The real names of the people in a household, keyed by the id the record uses.

The frozen bank files carry only `resident_1` and `resident_2`, while the objects carry
their owners' names - `mug_yuki`. So the robot could see both halves and never the link,
and an earlier version of the prompt asked it to work the link out for itself. That was a
puzzle about identifiers, not about the household, and it is not what this study asks.

The names are recoverable exactly, because the simulator samples them from the seed that is
written into every bank file's episode id. Rebuilding the household from that seed gives
`Resident.id` and `Resident.name` together.

THE REBUILD IS CHECKED, EVERY TIME. If the set of rebuilt names is not exactly the set of
owner names on the objects, the rebuild did not reproduce this household and this raises
rather than returning a mapping that is half right. Verified on all ten households of the
varied set: ten of ten reproduce exactly.

A weaker method was tried first and rejected: matching each resident to the owner whose
things are most often in the room they are in. It happened to give the right answer on
every household tested, but with margins as thin as 51% against 49%, and a guess told to
the model as a fact is worse than telling it nothing.
"""
from __future__ import annotations

import functools
import json
import pathlib
import re
from typing import Dict


class CouldNotWorkOutWhoLivesHere(Exception):
    """The household did not rebuild to the same people. Better to stop than to guess."""


@functools.lru_cache(maxsize=256)
def names_by_resident_id(bank_path: str) -> Dict[str, str]:
    """{"resident_1": "Yuki", ...} for one household, capitalised for reading."""
    path = pathlib.Path(bank_path)
    lines = path.read_text().splitlines()
    seed = None
    owners = set()
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if seed is None and isinstance(row.get("episode_id"), str):
            found = re.search(r"seed(\d+)", row["episode_id"])
            if found:
                seed = int(found.group(1))
        thing = row.get("object_id") or ""
        if "_" in thing and thing.rsplit("_", 1)[1] != "shared":
            owners.add(thing.rsplit("_", 1)[1])
    if seed is None:
        raise CouldNotWorkOutWhoLivesHere(f"{path.name} carries no seed in its episode id")

    from situation_sim.household import sample_household
    from situation_sim.run import load_activities
    household = sample_household(seed, load_activities())
    mapping = {person.id: person.name for person in household.residents.values()}
    if set(mapping.values()) != owners:
        raise CouldNotWorkOutWhoLivesHere(
            f"{path.name}: rebuilding from seed {seed} gave "
            f"{sorted(mapping.values())} but its objects belong to {sorted(owners)}, so "
            f"the rebuild is not this household and the names cannot be trusted")
    return {rid: name.capitalize() for rid, name in mapping.items()}

# What plain words each generator role deserves. The bank carries a role, not a job title, and
# these are the five it uses. NO AGE EXISTS IN THIS DATA: the households carry a name, a role, a
# bedroom, a workspace, hobbies and chores, and nothing about how old anybody is. An age would have
# to be invented, so none is given.
WHAT_THEY_DO = {"worker_home": "works from home",
                "worker_out": "goes out to work",
                "shift_worker": "works shifts, so the hours change",
                "student": "is a student",
                "retired": "is retired"}


def who_lives_here_in_plain_words(bank_path: str) -> str:
    """The one paragraph a profile-keeping arm is given on its first night.

    Only what a person would tell a robot on the day it arrived: who lives here, what each of them
    does, and whether they are a couple or share the place. NOT their hobbies, their chores, their
    bedroom or their workspace - those are in the household record and they are exactly what the
    robot is supposed to work out for itself. Checked against the bank by the same rebuild that
    `names_by_resident_id` uses, so a household that does not match its seed raises instead of
    handing over somebody else's details.
    """
    import re
    path = pathlib.Path(bank_path)
    seed = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row.get("episode_id"), str):
            found = re.search(r"seed(\d+)", row["episode_id"])
            if found:
                seed = int(found.group(1))
                break
    if seed is None:
        raise CouldNotWorkOutWhoLivesHere(f"{path.name} carries no seed in its episode id")
    names = names_by_resident_id(bank_path)          # raises if the rebuild is not this household
    from situation_sim.household import sample_household
    from situation_sim.run import load_activities
    household = sample_household(seed, load_activities())
    bits = []
    for rid, person in sorted(household.residents.items()):
        said = WHAT_THEY_DO.get(person.role)
        if said is None:
            raise CouldNotWorkOutWhoLivesHere(
                f"{path.name}: {person.role!r} is a role with no plain words, so the profile "
                f"would either invent one or leave a person out")
        bits.append(f"{names[rid]} {said}")
    together = ("They are a couple." if household.household_type == "couple"
                else "They share the place." if household.household_type == "flatmates"
                else "")
    return " ".join([(", ".join(bits) + "."), together]).strip()


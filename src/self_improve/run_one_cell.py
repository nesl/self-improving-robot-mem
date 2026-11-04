"""One cell of the two-by-three: one household, one way of writing notes, one way
of choosing where to look.

    python -m self_improve.run_one_cell --household hh_s3_t03 \
        --memory-format "incremental edits" --looking-arm "the repaired chooser"

Writes to results/self_improve/cells/<household>/<memory format>__<looking arm>/
so the directory name says what the cell is without needing a key.

Stage one, which this does: build the look stream and the notes across the month,
snapshot the notes at every freeze point, and record per-night diagnostics. Stage
two, the frozen-memory answering, is a separate pass over the snapshots, because
it costs about twenty times as many model calls and because every one of the
checks we monitor during a sweep is about stage one.

A property the design depends on and that is asserted here rather than trusted:
**the two fixed-rotation cells of a household must see byte-identical looks.** The
fixed rota does not depend on what the robot remembers, so if those two look
streams differ, the memory comparison is confounded by different observations and
the cell is void.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any, Dict, List, Optional

from baselines.patrol.llm import LLMClient
from self_improve.choose_where_to_look import choose_and_look
from self_improve.frozen_household import FROZEN_BANKS, FrozenHousehold, period_of_day
from self_improve.looking import (FixedLookSchedule, TheHouseAsSeen, clock_to_seconds)
from self_improve.memory_notes import Notes
from self_improve.study_settings import LOCKED
from self_improve.write_the_notes import write_the_notes

MEMORY_FORMATS = ("wholesale rewrite", "incremental edits")

# name -> (told its own look history, must name one thing the claims disagree
#          about, candidates are every room rather than only rooms the notes
#          already mention)
LOOKING_ARMS: Dict[str, Optional[tuple]] = {
    "the fixed fair rotation": None,
    # Kept as an arm, not discarded: the hypothesis taken literally. Its candidates
    # come only from disagreements already written in the notes, so it cannot
    # consider a room the notes never mention - and it collapsed onto one room ten
    # times out of ten, never reaching the room the change moved into.
    "the naive chooser": (False, False, False),
    # Candidates are every room in the house, each with what the notes predict for
    # it, explicitly including "the notes say nothing about this room". Also keeps
    # the one-thing-disagreement requirement, which took the share of looks that
    # settled which claim held from 0% to 80%. The look-history information is
    # available but is NOT the repair - it changed nothing at all.
    "the repaired chooser": (True, True, True),
}

FREEZE_DAYS = (13, 16, 23, 28)
WRITE_NOTES_AT = 23 * 3600 + 30 * 60


def cell_directory(root: pathlib.Path, household: str, memory_format: str,
                   looking_arm: str) -> pathlib.Path:
    return root / household / f"{memory_format.replace(' ', '_')}__{looking_arm.replace(' ', '_')}"


def run_cell(household: FrozenHousehold, memory_format: str, looking_arm: str,
             client: LLMClient, out_dir: pathlib.Path, last_day: int = 31,
             settings=LOCKED) -> Dict[str, Any]:
    if memory_format not in MEMORY_FORMATS:
        raise ValueError(f"unknown memory format {memory_format!r}")
    if looking_arm not in LOOKING_ARMS:
        raise ValueError(f"unknown looking arm {looking_arm!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    switches = LOOKING_ARMS[looking_arm]

    eyes = TheHouseAsSeen(household, out_dir / "looks.jsonl", settings.granularity)
    schedule = FixedLookSchedule(household, settings.granularity,
                                 settings.fixed_schedule_kind, settings.budget_per_look,
                                 settings.visit_times, settings.rotation_seed)
    notes = Notes(out_dir / "notes.json", household.name,
                  f"{memory_format} / {looking_arm}", memory_format)

    warm_start = None
    if settings.shared_warm_start:
        record = eyes.shared_warm_start()
        warm_start = {"n_things_seen": len(record.sightings),
                      "n_asked_objects_seen": len(record.asked_objects_seen),
                      "of_how_many": len(household.asked_objects)}

    nightly: List[Dict[str, Any]] = []
    for day in range(last_day + 1):
        looks_today: List[Any] = []
        for visit_index, when in enumerate(settings.visit_times):
            if switches is None:
                looks_today.append(eyes.look(schedule.targets_for(day, visit_index), day,
                                             clock_to_seconds(when), "the fixed schedule"))
            else:
                history, same_object, every_room = switches
                looks_today.append(choose_and_look(
                    eyes, household, notes, day, when, clock_to_seconds(when), client,
                    fall_back_to=schedule, visit_index=visit_index,
                    tell_it_days_since_each_room_was_looked_in=history,
                    require_the_two_claims_to_be_about_the_same_object=same_object,
                    enumerate_every_room=every_room,
                    shuffle_the_candidate_order_with_seed=
                        settings.shuffle_the_candidate_order_with_seed))
        if day == 0 and warm_start is not None:
            looks_today = [eyes.looks[0]] + looks_today

        report = write_the_notes(notes, household, day, day * 86400 + WRITE_NOTES_AT,
                                 looks_today, client, settings.read_budget_lines)
        report["period"] = period_of_day(day)
        report["rooms_looked_in"] = [t["name"] for look in looks_today for t in look.targets]
        report["n_claims_after_tonight"] = len(notes.claims)
        report["n_summary_lines_after_tonight"] = (
            len(notes.nightly_summaries[-1]["summary"].splitlines())
            if notes.nightly_summaries else 0)
        nightly.append(report)
        notes.save()
        # The notes as they read at the end of each night. The timeliness measure
        # needs to know WHEN the notes first said something true about a moved
        # object, and a single overwritten notes.json cannot answer that.
        with (out_dir / "notes_by_night.jsonl").open("a") as history_file:
            history_file.write(json.dumps({
                "day": day,
                "text": (notes.newest_summary() or "") if memory_format == "wholesale rewrite"
                        else "\n".join(c.as_plain_words() for c in notes.claims),
            }) + "\n")
        # written after every night so a sweep can be analysed while it runs
        (out_dir / "nightly.json").write_text(json.dumps(nightly, indent=1))
        if day in FREEZE_DAYS:
            notes.snapshot_to(out_dir / f"notes_frozen_at_day_{day}.json")
    eyes.close()

    looks_bytes = (out_dir / "looks.jsonl").read_bytes()
    result = {
        "household": household.name,
        "memory_format": memory_format,
        "looking_arm": looking_arm,
        "warm_start": warm_start,
        "read_budget_lines": settings.read_budget_lines,
        "schedule": schedule.describe(),
        "the_candidate_order_was_shuffled_with_seed":
            settings.shuffle_the_candidate_order_with_seed,
        "n_looks": len(eyes.looks),
        "n_sightings": len(eyes.sightings_so_far()),
        "n_absences_recorded": len(eyes.absences_so_far()),
        "look_stream_sha256": hashlib.sha256(looks_bytes).hexdigest(),
        "look_stream_bytes": len(looks_bytes),
        "n_claims_at_the_end": len(notes.claims),
        "nightly": nightly,
        "model": {**client.stats, "model": client.model},
    }
    (out_dir / "cell.json").write_text(json.dumps(result, indent=1))
    return result


def assert_the_fixed_cells_saw_the_same_looks(root: pathlib.Path, household: str) -> Dict[str, Any]:
    """The fixed rota cannot depend on what the robot remembers, so both fixed
    cells of a household must have byte-identical look streams. If they do not, the
    memory comparison for that household is confounded and the cells are void."""
    streams = {}
    for memory_format in MEMORY_FORMATS:
        path = cell_directory(root, household, memory_format,
                              "the fixed fair rotation") / "looks.jsonl"
        if path.exists():
            data = path.read_bytes()
            streams[memory_format] = (len(data), hashlib.sha256(data).hexdigest())
    if len(streams) < 2:
        return {"household": household, "checked": False,
                "why": "both fixed cells have not written a look stream yet",
                "streams": streams}
    hashes = {h for _, h in streams.values()}
    return {"household": household, "checked": True,
            "the_two_fixed_cells_saw_identical_looks": len(hashes) == 1,
            "streams": {k: {"bytes": b, "sha256": h} for k, (b, h) in streams.items()},
            "verdict": ("identical, as the design requires" if len(hashes) == 1 else
                        "DIFFERENT - the memory comparison for this household is "
                        "confounded by different observations and these cells are void")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household", required=True)
    parser.add_argument("--memory-format", required=True, choices=list(MEMORY_FORMATS))
    parser.add_argument("--looking-arm", required=True, choices=list(LOOKING_ARMS))
    parser.add_argument("--banks", type=pathlib.Path, default=FROZEN_BANKS)
    parser.add_argument("--last-day", type=int, default=31)
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path("results/self_improve/cells"))
    parser.add_argument("--cache", type=pathlib.Path,
                        default=pathlib.Path("llm_prior_cache/self_improve"))
    args = parser.parse_args(argv)

    household = FrozenHousehold(args.banks / f"{args.household}.jsonl")
    client = LLMClient(args.cache)
    out_dir = cell_directory(args.root, args.household, args.memory_format,
                             args.looking_arm)
    result = run_cell(household, args.memory_format, args.looking_arm, client, out_dir,
                      args.last_day)
    print(f"{args.household} | {args.memory_format} | {args.looking_arm}: "
          f"{result['n_looks']} looks, {result['n_claims_at_the_end']} claims, "
          f"{result['model'].get('lost', 0)} lost calls", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

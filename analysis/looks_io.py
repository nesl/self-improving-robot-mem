"""Read a cell's looks (the robot's record of where it looked and what it saw).

The public release ships `looks_without_absences.jsonl.gz` in every cell: the cell's looks.jsonl with
each look's `absences` list removed (that list is ~95% of the file and no paper number needs it).
Everything else on each look line (day, time, targets, sightings, ...) is unchanged. If a full
looks.jsonl is present (a rerun), it is read instead.
"""
import gzip
import json
import pathlib


def read_looks(cell):
    cell = pathlib.Path(cell)
    full = cell / "looks.jsonl"
    if full.exists():
        with open(full) as f:
            return [json.loads(line) for line in f]
    with gzip.open(cell / "looks_without_absences.jsonl.gz", "rt") as f:
        return [json.loads(line) for line in f]

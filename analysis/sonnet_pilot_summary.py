"""Sonnet one-household pilot (hh_s2_t03, ten-homes banks, 8 a day, 31 days, claude-sonnet-5-5, thinking off):
first room right per arm, and the room choices that stayed refused after the reworded retry and fell back to the
fixed room order. Written for the public release; reads finished cells only. CPU only.

    PYTHONPATH=src python3 analysis/sonnet_pilot_summary.py
"""
import json
import pathlib

CELLS = pathlib.Path("results/self_improve/camera_ready/sonnet_pilot_v2/cells")
ARMS = {"log_only_no_notes": "history only", "ACE_as_published": "ACE as published",
        "the_log_and_notes_about_the_routine": "history and notes", "incremental_edits": "claim store"}

print(__doc__)
print(f"{'arm':22s} {'cell dir':38s} {'first room right':>16s} {'questions':>9s} {'choices':>8s} {'choices lost':>12s}")
for arm, name in ARMS.items():
    rows = [r for r in map(json.loads, open(CELLS / arm / "hh_s2_t03" / "searches.jsonl")) if r.get("kind") == "search"]
    frr = sum(r.get("found_at_step") == 1 for r in rows) / len(rows)
    choices = sum(r["n_rooms_opened"] for r in rows)
    lost = sum(r["n_choice_calls_that_failed"] for r in rows)
    print(f"{name:22s} {arm:38s} {frr:16.3f} {len(rows):9d} {choices:8d} {lost:12d}")

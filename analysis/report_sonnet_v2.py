"""Per-arm report for Sonnet pilot v2: tokens and cost ($2/$10, cache write $2.50, read $0.20), wall time, refusals
and reworded retries by call type, lost calls, item-count retries, clipped strings, empty-edit nights."""
import json, pathlib, collections
D = pathlib.Path("results/self_improve/camera_ready/sonnet_pilot_v2")   # run from the repository root
KIND = {("confidence", "location", "reasoning"): "answer", ("room", "why"): "choice", ("edits",): "night edits",
        ("verdicts", "what_to_change_tonight"): "ACE judge", ("merges",): "ACE merge"}
def usd(rs): return sum(r.get("prompt_tokens", 0)*2e-6 + r.get("cache_creation_input_tokens", 0)*2.5e-6
                        + r.get("cache_read_input_tokens", 0)*.2e-6 + r.get("completion_tokens", 0)*10e-6 for r in rs)
for arm in ["incremental_edits", "log_only_no_notes", "the_log_and_notes_about_the_routine", "ACE_as_published"]:
    u = D / f"{arm}.usage.jsonl"
    if not u.exists(): continue
    rs = [json.loads(l) for l in open(u)]
    tf = D / (f"{arm}.rerun.time" if arm.startswith("ACE") else f"{arm}.time")
    wall = json.load(open(tf))["wall_seconds"] / 60 if tf.exists() else None
    print(f"== {arm}: {len(rs)} API calls | in {sum(r.get('prompt_tokens',0) for r in rs)} cw "
          f"{sum(r.get('cache_creation_input_tokens',0) for r in rs)} cr {sum(r.get('cache_read_input_tokens',0) for r in rs)} "
          f"out {sum(r.get('completion_tokens',0) for r in rs)} | ${usd(rs):.2f} | wall {wall and round(wall)} min")
    by = collections.defaultdict(collections.Counter)
    for r in rs:
        k = KIND.get(tuple(r.get("schema_fields") or ()), str(r.get("schema_fields")))
        c = by[k]; c["calls"] += 1
        c["reworded retry"] += bool(r.get("reworded_after_refusal"))
        c["refused at the end"] += r.get("stop_reason") == "refusal"
        c["lost"] += not r["ok"]
        c["item-count retry"] += bool(r.get("retried_for_item_count"))
        c["strings/lists cut"] += r.get("cuts_to_schema_limits", 0)
        c["hit max_tokens"] += r.get("stop_reason") == "max_tokens"
    for k, c in sorted(by.items()): print(f"   {k:12s} {dict(c)}")
    cell = D / "cells" / arm / "hh_s2_t03" / "cell.json"
    if cell.exists():
        ns = json.load(open(cell))["nightly"]
        e = [x.get("n_edits_offered") for x in ns]
        print(f"   nights {len(ns)} | empty-edit nights {sum(1 for x in e if x == 0)} (night 0 included) | "
              f"failed nights {sum(bool(x.get('model_call_failed')) for x in ns)} | edits/night {e}")

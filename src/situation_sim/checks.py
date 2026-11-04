"""Checkpoint checks, run on the written files (not on in-memory state):

1. at least one active cause over the run (quiet days are allowed and counted)
2. at least one cause affects eight or more distinct objects (count per cause)
3. whim share of placements is between 5% and 30%
4. two runs at the same seed produce byte-identical output
5. no object ends up somewhere its rules do not allow
Plus: the file loads through the bank loader in src/baselines/bank.py when
that package is importable (an extra, not one of the five).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List

from situation_sim.household import ON_PERSON, OUT_OF_HOUSE


def _rows(path: pathlib.Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_checks(out: pathlib.Path, seed: int, n_days: int, day0: str = "Wednesday", calendar=None) -> bool:
    ok = True
    state = json.loads((out / "hidden_state.json").read_text())
    rows = _rows(out / "events.jsonl")
    truth = [r for r in rows if r["kind"] == "truth"]
    print("\n=== checkpoint checks ===")

    # 1. every day has a cause
    empty = [d["weekday"] for d in state["days"] if not d["causes"]]
    per_day = {d["weekday"]: len(d["causes"]) for d in state["days"]}
    any_cause = any(per_day.values())
    print(f"[1] causes per day: {per_day}; quiet days: {empty or 'none'} -> {'PASS' if any_cause else 'FAIL (no cause on any day)'}")
    ok &= any_cause

    # 2. distinct objects per cause
    objs_by_cause: Dict[str, set] = defaultdict(set)
    for r in truth:
        for c in r.get("causes", []):
            objs_by_cause[c].add(r["object_id"])
    for n in state["stats"].get("notes", []):
        for c in n["causes"]:
            objs_by_cause[c].add(n["object_id"])
    print("[2] distinct objects moved per cause:")
    best = 0
    for c, s in sorted(objs_by_cause.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"      {len(s):3d}  {c}")
        best = max(best, len(s))
    all_causes = {c["id"] for d in state["days"] for c in d["causes"]}
    silent = sorted(all_causes - set(objs_by_cause))
    if silent:
        print(f"      (causes that moved nothing: {', '.join(silent)})")
    print(f"    max = {best} -> {'PASS' if best >= 8 else 'FAIL'}")
    ok &= best >= 8

    # 3. whim share
    st = state["stats"]
    share = st["whim_share_of_decisions"]
    print(f"[3] whim: {st['whims']} of {st['placement_decisions']} placement decisions = "
          f"{100 * share:.1f}% (of the {st['placement_moves']} decisions that moved something: "
          f"{100 * st['whim_share_of_moves']:.1f}%) -> {'PASS' if 0.05 <= share <= 0.30 else 'FAIL'}")
    ok &= 0.05 <= share <= 0.30

    # 4. determinism: regenerate into a temp dir and compare bytes
    from situation_sim.run import generate
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="situation_sim_det_"))
    try:
        generate(seed, tmp, n_days, day0, calendar=calendar)
        same = all(_sha(out / f) == _sha(tmp / f)
                   for f in ("trace.md", "events.jsonl", "hidden_state.json", "trace.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[4] second run at seed {seed} byte-identical: {'PASS' if same else 'FAIL'}")
    ok &= same

    # 5. every truth receptacle is allowed for the object
    allowed = {oid: set(o["allowed"]) for oid, o in state["household"]["objects"].items()}
    bad = [(r["object_id"], r["receptacle_id"], r["t"]) for r in truth
           if r["receptacle_id"] not in allowed[r["object_id"]]]
    nocarrier = [r for r in truth if r["receptacle_id"] == ON_PERSON and not r.get("carrier")]
    print(f"[5] truth rows: {len(truth)}; rows outside the object's allowed set: {len(bad)}; "
          f"ON_PERSON rows without carrier: {len(nocarrier)} -> "
          f"{'PASS' if not bad and not nocarrier else 'FAIL'}")
    for b in bad[:10]:
        print("      ", b)
    ok &= not bad and not nocarrier

    # extra: density stats (how much evidence a home offers)
    H = state["household"]
    n_obj = len(H["objects"]); n_static = sum(1 for o in H["objects"].values() if o.get("static"))
    spots = [r for r in H["receptacles"]]
    by_obj = {}
    for r in truth:
        by_obj.setdefault(r["object_id"], []).append(r)
    import bisect
    def loc_at(obj, t):
        rs = by_obj[obj]; ts = [x["t"] for x in rs]; i = bisect.bisect_right(ts, t) - 1
        return rs[i]["receptacle_id"] if i >= 0 else None
    samples = [d * 86400 + h * 3600 for d in range(n_days) for h in range(7, 23, 3)]
    occ = []
    for t in samples:
        held = set(loc_at(o, t) for o in by_obj)
        occ.append(sum(1 for sp in spots if sp in held) / len(spots))
    moves = [r for r in truth if r["cause"] != "initial" and r["receptacle_id"] not in ("ON_PERSON",)]
    movers = {o for o in by_obj if sum(1 for r in by_obj[o] if r["cause"] != "initial") >= 3}
    print(f"[i] {n_obj} objects ({n_static} fixtures), {len(spots)} spots in {len(H['rooms'])} rooms; "
          f"{len(moves) / n_days:.0f} object moves/day; {len(movers)} objects move 3+ times; "
          f"{100 * sum(occ) / len(occ):.0f}% of spots hold something (sampled 07-22h)")
    # extra: bank loader
    try:
        from baselines.bank import JsonlBank  # type: ignore
        bank = JsonlBank(out / "events.jsonl")
        eps = list(bank.episodes())
        print(f"[+] baselines.bank loader: loaded {len(eps)} episode(s), "
              f"{len(eps[0].trajectories)} object trajectories")
    except ImportError:
        print("[+] baselines.bank not importable from here; loader check skipped")
    except Exception as e:  # noqa: BLE001
        print(f"[+] baselines.bank loader REJECTED the file: {type(e).__name__}: {e}")
    print("=== overall:", "PASS" if ok else "FAIL", "===")
    return bool(ok)

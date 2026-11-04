"""Fixed random sample (seed 0) of WRONG FIRST-ROOM choices during and after the illness, log only vs log and
notes, ten homes at 8 a day. For each: the model's stated reason for its first room, and the object's true history
up to the question: every dated sighting and EVERY failed look (not folded), so a coder can judge the reason."""
import json, os, random, glob, pathlib, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from looks_io import read_looks
OUT = pathlib.Path(os.environ.get('OUT_DIR', 'out')) / 'absence_misreads'
OUT.mkdir(parents=True, exist_ok=True)
ARMS = {"log only": "results/self_improve/wave_log_only/cells/log_only_no_notes",
        "log and notes": "results/self_improve/overnight_wave/cells/the_log_and_notes_about_the_routine"}
def hhmm(t): t %= 86400; return f"{t//3600:02d}:{t%3600//60:02d}"
rng = random.Random(0); items = []
for arm, base in ARMS.items():
    pool = []
    for d in sorted(glob.glob(f"{base}/*/")):
        rs = [json.loads(l) for l in open(d + "searches.jsonl")]
        rs = [r for r in rs if r.get("kind") == "search" and r["period"] in ("disrupted", "back to normal")
              and r["rooms_opened"] and r["rooms_opened"][0] != r["true_room"] and r["why_each_room"]]
        pool += [(d, r) for r in rs]
    rng.shuffle(pool); per_home = {}; chosen = []
    for d, r in pool:
        if per_home.get(d, 0) >= 3: continue
        per_home[d] = per_home.get(d, 0) + 1; chosen.append((d, r))
        if len(chosen) == 20: break
    for d, r in chosen:
        looks = read_looks(d)
        looks = [l for l in looks if l.get("kind") == "look" and l["time"] < r["time"]]
        hist = []
        for l in looks:
            rooms = [t["name"] for t in l["targets"]]
            seen = [s for s in l["sightings"] if s["object_id"] == r["object_id"]]
            if seen: hist.append(f"day {l['day']} {hhmm(l['time'])}: SEEN in {seen[0]['room']} ({seen[0]['place_id']})")
            else: hist.append(f"day {l['day']} {hhmm(l['time'])}: looked in {', '.join(rooms)}, NOT there")
        folded = {}
        for l in looks:
            if r["object_id"] in {s["object_id"] for s in l["sightings"]}: continue
            for t in l["targets"]: folded.setdefault(t["name"], []).append((l["day"], l["time"]))
        shown = [f"- {room}: {len(w)} look(s), most recently day {max(w)[0]} at {hhmm(max(w)[1])}"
                 for room, w in sorted(folded.items(), key=lambda kv: max(kv[1]), reverse=True)]
        items.append({"what_the_model_was_shown_about_failed_looks (folded, as in the prompt)": shown,"id": f"{'LO' if arm=='log only' else 'LN'}{len(items)%20+1:02d}", "arm": arm,
                      "home": pathlib.Path(d).name, "object": r["object_id"], "day": r["day"],
                      "time": hhmm(r["time"]), "period": r["period"], "true_room": r["true_room"],
                      "first_room_chosen": r["rooms_opened"][0], "reason_for_first_room": r["why_each_room"][0],
                      "history_last_25_looks_mentioning_relevant_rooms": hist[-25:]})
json.dump(items, open(OUT / "sample.json", "w"), indent=1)
print(len(items), "items;", sum(i["arm"]=="log only" for i in items), "log only")

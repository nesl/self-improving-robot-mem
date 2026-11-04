import json, glob, os, sys, statistics as st, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from looks_io import read_looks
base = "results/self_improve/"
def q(xs, ps=(0.1,0.5,0.9)):
    xs = sorted(xs); n=len(xs)
    return [xs[min(n-1,int(p*n))] for p in ps]
for wave in ["overnight_wave", "overnight_wave_24_questions"]:
  for arm in ["the_log_and_notes_about_the_routine"]:
    nq=[]; per_obj=[]; capped=0; tot_sight=[]; tot_looks=[]; full_lines=[]; final_obj=[]; days=[]
    for c in sorted(glob.glob(f"{base}{wave}/cells/{arm}/*/")):
        looks=read_looks(c); looks=[l for l in looks if l.get("kind")=="look"]
        srch=[json.loads(l) for l in open(c+"searches.jsonl")]; srch=[s for s in srch if s.get("kind")=="search"]
        nq.append(len(srch)); days.append(max(s["day"] for s in srch))
        sight=[(s["time"],s["object_id"]) for l in looks for s in l["sightings"]]
        # global time: use day*86400+time if time is within-day
        def T(d,t): return t if t>86400 else d*86400+t
        sight=[(T(s["day"],s["time"]),s["object_id"]) for l in looks for s in l["sightings"]]
        lk=[T(l["day"],l["time"]) for l in looks]
        for s in srch:
            t=T(s["day"],s["time"])
            n=sum(1 for (ts,o) in sight if ts<t and o==s["object_id"])  # strictly before the question
            per_obj.append(n); capped += n>40
            tot_sight.append(sum(1 for (ts,o) in sight if ts<t)); tot_looks.append(sum(1 for x in lk if x<t))
        tot=collections.Counter(o for _,o in sight)
        final_obj += [tot[o] for o in set(s["object_id"] for s in srch)]
        full_lines.append((len(looks), len(sight)))
    print(f"== {wave} / {arm}: {len(nq)} sessions")
    print(" questions/session median", st.median(nq), "range", min(nq), max(nq), " days", st.median(days))
    print(" sightings of asked object before question: p10/50/90", q(per_obj), "mean %.1f"%st.mean(per_obj), "max", max(per_obj), f" >40: {capped}/{len(per_obj)} = {capped/len(per_obj):.1%}")
    print(" whole trace before question: sightings p10/50/90", q(tot_sight), " looks", q(tot_looks))
    print(" end-of-session trace (looks, sightings) median", st.median(a for a,_ in full_lines), st.median(b for _,b in full_lines))
    print(" per asked object end-of-session sightings p50/p90/max", q(final_obj,(0.5,0.9)), max(final_obj))

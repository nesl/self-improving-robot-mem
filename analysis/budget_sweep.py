"""Our arm against last-seen at 4, 8 and 24 questions a day, the same five wider homes."""
import json, pathlib, statistics
HOMES=["hh_s32_t03","hh_s48_t03","hh_s63_t03","hh_s93_t03","hh_s151_t03"]
WAVES={4:"wave_the_budget_sweep_q4",8:"wave_the_budget_sweep_q8",24:"wave_wider_five"}
OURS="the_log_and_notes_about_the_routine"; RULE="last_seen_no_model"
# Public release: the 8- and 24-a-day history-and-notes cells are not shipped (that arm is not in the camera-ready
# paper, and they were cut for size), so only the budgets whose cells are present are computed.
BUDGETS=[q for q in (4,8,24) if (pathlib.Path('results/self_improve')/WAVES[q]/'cells'/OURS).exists()]
print('budgets computed: ' + ', '.join(f'{q} a day' for q in BUDGETS)
      + ''.join(f'; {q} a day skipped (cells not in this release)' for q in (4,8,24) if q not in BUDGETS))
WIN={"settled 1-13":range(1,14),"spell 14-23":range(14,24),"back 24-31":range(24,32),"whole month":range(1,32)}
MEAS={"first":lambda r:r.get("found_at_step")==1,"found":lambda r:bool(r.get("found_it"))}
FLOOR={"first":1.9,"found":2.2}
def rows(w,arm,h,movers_only=False):
    f=pathlib.Path("results/self_improve")/WAVES[w]/"cells"/arm/h/"searches.jsonl"
    out=[]
    for l in f.open():
        r=json.loads(l)
        if r.get("kind")!="search": continue
        if movers_only and not r.get("is_a_mover"): continue
        out.append(r)
    return out
for movers_only in (False,True):
    tag="MOVED OBJECTS ONLY" if movers_only else "EVERY QUESTION"
    print(f"\n################ {tag}")
    for m in ("first","found"):
        print(f"\n=== {m}: ours minus last-seen, paired on home, 2 SE across the five homes, floor {FLOOR[m]}")
        print(f"{'window':16s}" + "".join(f"{f'{q} a day':>34s}" for q in BUDGETS))
        for w in WIN:
            line=f"{w:16s}"
            for q in BUDGETS:
                d=[]; ns=[]
                for h in HOMES:
                    a=[r for r in rows(q,OURS,h,movers_only) if r["day"] in WIN[w]]
                    b=[r for r in rows(q,RULE,h,movers_only) if r["day"] in WIN[w]]
                    if len(a)<8 or len(b)<8: continue
                    ns.append(len(a))
                    d.append(100*sum(1 for r in a if MEAS[m](r))/len(a)
                             -100*sum(1 for r in b if MEAS[m](r))/len(b))
                if len(d)<2: line+=f"{'-':>34s}"; continue
                mn=statistics.mean(d); tse=2*statistics.stdev(d)/len(d)**0.5
                up=sum(1 for v in d if v>0)
                v="CLEARS" if abs(mn)>tse and abs(mn)>FLOOR[m] else ""
                line+=f"{f'{mn:+5.1f} 2SE {tse:4.1f} n={len(d)} up{up} q~{int(statistics.mean(ns))} {v}':>34s}"
            print(line)
    for q in BUDGETS:
        print(f"  {q} a day, whole month level: ours " + " ".join(
            f"{100*sum(1 for r in rows(q,OURS,h,movers_only) if MEAS['first'](r))/len(rows(q,OURS,h,movers_only)):.0f}" for h in HOMES)
            + " | last-seen " + " ".join(
            f"{100*sum(1 for r in rows(q,RULE,h,movers_only) if MEAS['first'](r))/len(rows(q,RULE,h,movers_only)):.0f}" for h in HOMES))

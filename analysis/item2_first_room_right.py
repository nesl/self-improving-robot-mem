"""Paired first room right, Qwen3.8-27B, ten-homes banks, 8 a day, 31 days: arm B minus arm A, per period, 2 SE over homes.
usage: item2_first_room_right.py [arm_dir_B] [arm_dir_A] (defaults claim_store_plus_the_log, incremental_edits = claim store)."""
import json,glob,sys,statistics as S
P=('settled','disrupted','back to normal')
base='results/self_improve/camera_ready/qwen_ten_homes_8_a_day/cells/'
B=sys.argv[1] if len(sys.argv)>1 else 'claim_store_plus_the_log'; A=sys.argv[2] if len(sys.argv)>2 else 'incremental_edits'
def score(p):
    r={k:[0,0] for k in P+('all',)}
    for l in open(p):
        d=json.loads(l)
        if d.get('kind')!='search': continue
        ok=bool(d['rooms_opened']) and d['rooms_opened'][0]==d['true_room']
        for k in (d['period'],'all'): r[k][0]+=ok; r[k][1]+=1
    return {k:v[0]/v[1] for k,v in r.items()}, r['all']
def done(arm,h): return bool(glob.glob(base+arm+'/'+h+'/searches.jsonl')) and not glob.glob(base+arm+'/'+h+'/RUNNING.lock')
homes=sorted(h for h in (p.split('/')[-1] for p in glob.glob(base+B+'/*')) if done(A,h) and done(B,h))  # finished in both arms
print(f"First room right, {A} -> {B}. Source: {base}<arm>/<home>/searches.jsonl")
diffs={k:[] for k in P+('all',)}; pa=[0,0]; pb=[0,0]
for h in homes:
    (a,ca),(b,cb)=score(f'{base}{A}/{h}/searches.jsonl'),score(f'{base}{B}/{h}/searches.jsonl')
    pa=[pa[0]+ca[0],pa[1]+ca[1]]; pb=[pb[0]+cb[0],pb[1]+cb[1]]
    print(f"{h:12s} "+"  ".join(f"{k}: {a[k]:.2f}->{b[k]:.2f}" for k in P+('all',)))
    for k in diffs: diffs[k].append(b[k]-a[k])
print(f"n homes = {len(homes)}; pooled {100*pa[0]/pa[1]:.1f} -> {100*pb[0]/pb[1]:.1f} ({pa[1]} questions each)")
for k,v in diffs.items():
    m=S.mean(v); se=S.stdev(v)/len(v)**.5 if len(v)>1 else float('nan')
    print(f"{k:15s} {B} minus {A}: {100*m:+.1f} +- {200*se:.1f} pts (2 SE), same sign {sum(x>0 for x in v)} of {len(v)}")

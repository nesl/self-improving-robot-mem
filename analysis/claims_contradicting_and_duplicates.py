"""Per finished claim-store run: claims, claims citing contradicting sightings, set aside, standing, extra verbatim copies among standing."""
import json,glob,collections,os,sys
os.chdir('results/self_improve/camera_ready')   # run from the repository root
print(__doc__)
tot=collections.Counter()
# The paper (E6) counts the two claim-store arms that had finished when this file was written: 10 homes x
# {incremental_edits, claim_store_plus_the_log} = 20 runs. `--all` also counts claim_store_plus_today's_sightings.
QWEN_ARMS = ['*'] if '--all' in sys.argv else ['incremental_edits', 'claim_store_plus_the_log']
qwen_paths = [q for a in QWEN_ARMS
              for q in glob.glob(f'qwen_ten_homes_8_a_day/cells/{a}/*/notes.json')]
for p in sorted(qwen_paths)+sorted(glob.glob('sonnet_pilot_v2/cells/*/*/notes.json'))+['../overnight_wave/cells/incremental_edits/hh_s2_t03/notes.json']:
    if os.path.exists(p.replace('notes.json','RUNNING.lock')): continue
    cl=json.load(open(p))['claims']
    if not cl: continue
    con=sum(1 for c in cl if c.get('contradicting_observation_ids'))
    st=[c for c in cl if c.get('standing')=='still standing']
    cnt=collections.Counter(c['statement'].strip().lower() for c in st)
    dup=sum(v-1 for v in cnt.values() if v>1); grp=sum(1 for v in cnt.values() if v>1)
    tag='qwen' if 'qwen' in p else ('sonnet' if 'sonnet' in p else 'table1')
    print(f"{tag:7s} {p.split('cells/')[1][:-11]:52s} claims {len(cl):3d} contra {con:2d} set_aside {len(cl)-len(st):2d} standing {len(st):3d} dup_copies {dup:2d} groups {grp}")
    if tag=='qwen':
        a=p.split('cells/')[1].split('/')[0]
        for k,v in (('runs',1),('claims',len(cl)),('contra',con),('standing',len(st)),('dup',dup)): tot[a,k]+=v
for a in sorted({a for a,_ in tot}): print('QWEN TOTAL',a,{k:tot[a,k] for k in ('runs','claims','contra','standing','dup')})

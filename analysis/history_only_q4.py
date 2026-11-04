"""History only at 4 a day on the five homes (v3 banks, 31 days, budget 3), paired against the other 4-a-day arms on the
same homes and questions. First room right, 2 SE over homes. Cells: wave_log_only_q4 (new, 2026-11-04) and
wave_the_budget_sweep_q4 (existing)."""
import json, statistics as S
R = 'results/self_improve/'
HOMES = ['hh_s32_t03', 'hh_s48_t03', 'hh_s63_t03', 'hh_s93_t03', 'hh_s151_t03']
NEW = R + 'wave_log_only_q4/cells/log_only_no_notes'
OTHERS = {'history and notes': R + 'wave_the_budget_sweep_q4/cells/the_log_and_notes_about_the_routine',
          'last-seen': R + 'wave_the_budget_sweep_q4/cells/last_seen_no_model',
          'last-seen updated each night': R + 'wave_the_budget_sweep_q4/cells/last_seen_updated_each_night'}
P = ('settled', 'disrupted', 'back to normal', 'all')
def score(cell):
    s = {x['question_id']: x for x in map(json.loads, open(cell + '/searches.jsonl')) if x.get('kind') == 'search'}
    return s
def frac(s, ids, p):
    v = [s[i]['rooms_opened'][:1] == [s[i]['true_room']] for i in ids if p == 'all' or s[i]['period'] == p]
    return sum(v) / len(v)
print(__doc__)
new = {h: score(f'{NEW}/{h}') for h in HOMES}
print('history only, per home: ' + '  '.join(f'{h[:-4]} {frac(new[h], new[h], "all"):.3f}' for h in HOMES))
for name, base in OTHERS.items():
    print(f'\n== history only minus {name}')
    d = {p: [] for p in P}
    for h in HOMES:
        o = score(f'{base}/{h}'); ids = sorted(set(o) & set(new[h]))
        assert len(ids) == len(new[h]) == len(o), (h, len(ids), len(new[h]), len(o))
        assert all(o[i]['object_id'] == new[h][i]['object_id'] and o[i]['time'] == new[h][i]['time'] for i in ids), h
        for p in P: d[p].append(frac(new[h], ids, p) - frac(o, ids, p))
        print(f'  {h[:-4]:8s} {name} {frac(o, ids, "all"):.3f} -> history only {frac(new[h], ids, "all"):.3f}')
    for p in P:
        v = d[p]; print(f'  {p:15s} {100*S.mean(v):+.1f} ± {200*S.stdev(v)/len(v)**.5:.1f} pts (2 SE, {len(v)} homes), higher in {sum(x>0 for x in v)} of {len(v)}')

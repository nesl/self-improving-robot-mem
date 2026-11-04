"""History only on the three 50-day homes (24 a day, days 1-49, budget 3), paired question by question against the
other 50-day arms in wave_the_second_illness. First room right, 2 SE over homes (3 homes). CPU only, finished cells.
Periods: settled 1-13, disrupted 14-23, back to normal 24-31, days 32-49 (second illness onward; the searches label it "unknown")."""
import json, statistics as S
R = 'results/self_improve/'
HOMES = ['hh_s2_t03', 'hh_s32_t03', 'hh_s48_t03']
NEW = R + 'wave_log_only_50day/cells/log_only_no_notes'
W = R + 'wave_the_second_illness/cells/'
OTHERS = {'history and notes': W + 'the_log_and_notes_about_the_routine', 'claim store': W + 'incremental_edits',
          'ACE as published': W + 'ACE_as_published', 'last-seen': W + 'last_seen_no_model',
          'last-seen updated each night': W + 'last_seen_updated_each_night'}
P = ('settled', 'disrupted', 'back to normal', 'unknown', 'all')
LAB = {'unknown': 'days 32-49'}
def load(cell):
    cj = json.load(open(cell + '/cell.json'))
    s = {x['question_id']: x for x in map(json.loads, open(cell + '/searches.jsonl')) if x.get('kind') == 'search'}
    return s, (cj.get('budget_rooms_per_question'), cj.get('questions_per_day'), cj.get('last_day'))
def frac(s, ids, p):
    v = [s[i]['rooms_opened'][:1] == [s[i]['true_room']] for i in ids if p == 'all' or s[i]['period'] == p]
    return sum(v) / len(v)
print(__doc__)
new = {h: load(f'{NEW}/{h}') for h in HOMES}
print('history only, per home: ' + '  '.join(f'{h[:-4]} {frac(new[h][0], new[h][0], "all"):.3f}' for h in HOMES))
for name, base in OTHERS.items():
    print(f'\n== history only minus {name}')
    d = {p: [] for p in P}
    for h in HOMES:
        o, oc = load(f'{base}/{h}'); n, nc = new[h]; ids = sorted(set(o) & set(n))
        assert oc == nc, (h, name, oc, nc)
        assert len(ids) == len(n) == len(o), (h, len(ids), len(n), len(o))
        assert all(o[i]['object_id'] == n[i]['object_id'] and o[i]['time'] == n[i]['time'] and o[i]['true_room'] == n[i]['true_room'] for i in ids), h
        for p in P: d[p].append(frac(n, ids, p) - frac(o, ids, p))
        print(f'  {h[:-4]:8s} {name} {frac(o, ids, "all"):.3f} -> history only {frac(n, ids, "all"):.3f}   ' +
              '  '.join(f'{LAB.get(p,p)} {frac(o, ids, p):.3f}->{frac(n, ids, p):.3f}' for p in P[:-1]))
    for p in P:
        v = d[p]; print(f'  {LAB.get(p,p):15s} {100*S.mean(v):+.1f} ± {200*S.stdev(v)/len(v)**.5:.1f} pts (2 SE, {len(v)} homes), higher in {sum(x>0 for x in v)} of {len(v)}')

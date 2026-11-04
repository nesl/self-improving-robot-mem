"""First-room-right ceiling from an observer that knows each household's true schedule (CPU only, no model).

For every question the Qwen runs scored (from incremental_edits searches.jsonl), the observer guesses the room
that most often held the object at the same clock time (+-0, 15, 30 min) on the OTHER days of the same period
(settled / disrupted / back to normal). So it knows the routine, including the illness routine, exactly, but not
that day's chance moves. Variant 'routine only' uses settled days only, i.e. it never learns the illness routine.
Objects out of reach (carried, out of the house) count as a room 'beyond reach' that can't be right.
"""
import collections, json, pathlib, statistics as S, sys
sys.path.insert(0, 'src')
from self_improve.frozen_household import FrozenHousehold

R = pathlib.Path('results/self_improve')
BANKS = R / 'varied_homes/ten_homes/banks'
CELLS = R / 'camera_ready/qwen_ten_homes_8_a_day/cells/incremental_edits'
DAY = 86400
OFFS = (0, -900, 900, -1800, 1800)

def guess(hh, obj, t, days):
    tod = t % DAY
    c = collections.Counter(hh.room_of_object(obj, d * DAY + tod + o) for d in days for o in OFFS)
    c.pop('beyond reach', None)  # never guess an unreachable place
    return c.most_common(1)[0][0] if c else None

rows = []
for cell in sorted(CELLS.iterdir()):
    sj = cell / 'searches.jsonl'
    if not sj.exists() or (cell / 'RUNNING.lock').exists():
        continue
    hh = FrozenHousehold(BANKS / f'{cell.name}.jsonl')
    qs = [json.loads(l) for l in open(sj)]
    qs = [q for q in qs if q.get('kind') == 'search']
    period_of_day = {q['day']: q['period'] for q in qs}
    days_in = collections.defaultdict(list)
    for d, p in period_of_day.items():
        days_in[p].append(d)
    for q in qs:
        same = [d for d in days_in[q['period']] if d != q['day']]
        settled = [d for d in days_in['settled'] if d != q['day']]
        g1 = guess(hh, q['object_id'], q['time'], same)
        g2 = guess(hh, q['object_id'], q['time'], settled)
        model = bool(q['rooms_opened']) and q['rooms_opened'][0] == q['true_room']
        rows.append((cell.name, q['period'], g1 == q['true_room'], g2 == q['true_room'], model))

print(__doc__)
print(f'{len(rows)} questions, {len({r[0] for r in rows})} homes\n')
print(f"{'':16s} {'knows schedule':>15s} {'routine only':>13s} {'claim store (Qwen)':>19s}")
for p in ('settled', 'disrupted', 'back to normal', None):
    rr = [r for r in rows if p is None or r[1] == p]
    f = lambda i: 100 * sum(r[i] for r in rr) / len(rr)
    print(f"{p or 'all':16s} {f(2):15.1f} {f(3):13.1f} {f(4):19.1f}   n={len(rr)}")
print('\nPer home, all periods (knows schedule):')
for h in sorted({r[0] for r in rows}):
    rr = [r for r in rows if r[0] == h]
    print(f'  {h:12s} {100*sum(r[2] for r in rr)/len(rr):5.1f}')

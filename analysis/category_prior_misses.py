"""Wrong first rooms that are the object type's usual room while the object's own latest sighting points elsewhere.
Ten homes, 8 a day, 31 days, Qwen3.8-27B. CPU only, reads finished cells.

Rule (fixed before counting):
  usual room of an object class = the room the no-memory arm (prior_only_no_notes, overnight_wave) opens FIRST most
    often for that class, pooled over the ten homes. That arm sees no sightings, so its first room is the model's
    category prior. Classes it never searched have no usual room and are never counted.
  a question counts as a CATEGORY-PRIOR MISS when, in the arm being scored:
    (a) the first room opened is wrong (not the true room),
    (b) the first room opened is the class's usual room, and
    (c) the object's latest sighting in that arm's own record before the question (looks.jsonl, time < question time)
        is in a different room. Questions with no earlier sighting are not counted (nothing points elsewhere).
  Also reported: how many of those had the latest sighting in the TRUE room (the record had the answer).
  STRICT count (the one to cite): (a)-(c) AND the reason for the first room makes a generic statement about the object
    type (regex GEN below, e.g. "mugs are most commonly found in the kitchen", "the most likely place for a glass").
    (a)-(c) alone over-count: they also catch choices argued from a claim or from the object's own past pattern
    (e.g. "glass_tomas is consistently found in the kitchen in the mornings"). Hand check 2026-11-04: about 16 of
    19 sampled STRICT cases are clearly category-prior reasoning; the rest mix in object-specific evidence.
"""
import json, collections, glob, os, re, statistics as S, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from looks_io import read_looks
R = 'results/self_improve/'
ARMS = {
    'claim store': R + 'camera_ready/qwen_ten_homes_8_a_day/cells/incremental_edits',
    'claim store plus today\'s sightings': R + "camera_ready/qwen_ten_homes_8_a_day/cells/claim_store_plus_today's_sightings",
    'claim store plus history': R + 'camera_ready/qwen_ten_homes_8_a_day/cells/claim_store_plus_the_log',
    'history only': R + 'wave_log_only/cells/log_only_no_notes',
    'history and notes': R + 'overnight_wave/cells/the_log_and_notes_about_the_routine',
}
N = r"(mugs|glasses|drinking glasses|water bottles|bottles|chargers|books|towels|tablets|notebooks|razors|medications?)"
A = r"(a mug|a glass|a water bottle|a charger|a book|a towel|a tablet|a notebook|a razor|medication)"
GEN = re.compile(rf"\b{N}\b (are|is) (most )?(commonly|typically|usually|often|frequently)|\b{N}\b are (most )?(commonly|typically|usually|often)? ?(found|kept|stored|used|left)|(place|location|spot) (for|to find) {A}\b|common place for {N}|\b{A} (is|would) (typically|usually|most likely) (be )?(found|kept)", re.I)
PRIOR = R + 'overnight_wave/cells/prior_only_no_notes'
HOMES = sorted(os.path.basename(h) for h in glob.glob(PRIOR + '/hh_*'))

def searches(cell):
    return [r for r in map(json.loads, open(cell + '/searches.jsonl')) if r.get('kind') == 'search']

first = collections.defaultdict(collections.Counter)
for h in HOMES:
    for r in searches(f'{PRIOR}/{h}'):
        if r['rooms_opened']: first[r['object_class']][r['rooms_opened'][0]] += 1
usual = {c: k.most_common(1)[0][0] for c, k in first.items()}

def sightings(cell):
    by = collections.defaultdict(list)
    for lk in read_looks(cell):
        if lk.get('kind') != 'look': continue
        for s in lk['sightings']: by[s['object_id']].append((s['time'], s['room']))
    for v in by.values(): v.sort()
    return by

print(__doc__)
print('usual room by class (no-memory arm, first room opened; count of that room / all its first rooms):')
for c in sorted(usual): print(f'  {c:18s} {usual[c]:10s} {first[c][usual[c]]}/{sum(first[c].values())}')
print()
for arm, base in ARMS.items():
    rows = []; tot = collections.Counter()
    for h in HOMES:
        cell = f'{base}/{h}'
        if not os.path.exists(cell + '/searches.jsonl') or os.path.exists(cell + '/RUNNING.lock'): continue
        sg = sightings(cell); c = collections.Counter()
        for r in searches(cell):
            if not r['rooms_opened'] or r['rooms_opened'][0] == r['true_room']: continue
            c['wrong'] += 1
            u = usual.get(r['object_class'])
            prev = [room for t, room in sg.get(r['object_id'], []) if t < r['time']]
            if u and r['rooms_opened'][0] == u and prev and prev[-1] != u:
                c['prior'] += 1; c['prior_' + r['period']] += 1
                if prev[-1] == r['true_room']: c['prior_last_seen_true'] += 1
                if GEN.search(r['why_each_room'][0]): c['strict'] += 1; c['strict_' + r['period']] += 1
        rows.append((h, c)); tot.update(c)
    sh = [c['prior'] / c['wrong'] for _, c in rows if c['wrong']]
    print(f'== {arm}: {tot["prior"]} of {tot["wrong"]} wrong first rooms ({100*tot["prior"]/tot["wrong"]:.1f}%) are '
          f'category-prior misses; per-home share {100*S.mean(sh):.1f} ± {200*S.stdev(sh)/len(sh)**.5:.1f} (2 SE, {len(sh)} homes); '
          f'latest sighting was in the true room in {tot["prior_last_seen_true"]} of them; by period: '
          f'settled {tot["prior_settled"]}, disrupted {tot["prior_disrupted"]}, back to normal {tot["prior_back to normal"]}')
    print('   ' + '  '.join(f'{h.split("_t")[0]} {c["prior"]}/{c["wrong"]}' for h, c in rows))
    st = [c['strict'] / c['wrong'] for _, c in rows if c['wrong']]
    print(f'   STRICT: {tot["strict"]} of {tot["wrong"]} wrong first rooms ({100*tot["strict"]/tot["wrong"]:.1f}%); per-home share '
          f'{100*S.mean(st):.1f} ± {200*S.stdev(st)/len(st)**.5:.1f} (2 SE); by period: settled {tot["strict_settled"]}, '
          f'disrupted {tot["strict_disrupted"]}, back to normal {tot["strict_back to normal"]}')
    print('   STRICT per home: ' + '  '.join(f'{h.split("_t")[0]} {c["strict"]}/{c["wrong"]}' for h, c in rows))

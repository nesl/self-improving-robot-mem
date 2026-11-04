"""Export quiz runs pulled from the inspector artifact's database into the repo.

The page stores every player's live run (``chrono/<gen>s<seed>[__<player>]``),
archived runs (``runs/<id>``) and per-household notes (``notes/<gen>s<seed>``)
in the artifact db. That store is only reachable through the artifact runtime
or Claude's ArtifactData tool, so the monitor pulls the three collections to a
scratch directory (one JSON per document) and this script files them here:

    python3 -m situation_sim.export_runs --pulled <dir with chrono/ runs/ notes/> \
        --out ../data/situation_sim/human_runs

For every run document it writes
  human_runs/<player>/<household>__<doc id>.json     the raw document + source info
  human_runs/<player>/<household>__<doc id>.thoughts.md   the per-question notes and
                                                     the household notes, in play order
and refreshes human_runs/INDEX.md (player, household, started, progress, score
for finished runs, looks). Unchanged documents (same version) are skipped.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import sys
from typing import Dict, List

DAYS = ["Wed", "Thu", "Fri", "Sat", "Sun"]
ARTIFACT = "https://claude.ai/artifact/GHhmiBKaSwKvLBa38CzfrC"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-") or "anonymous"


def load_pulled(pulled: pathlib.Path) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}
    for coll in ("chrono", "runs", "notes"):
        out[coll] = {}
        for f in sorted((pulled / coll).glob("*.json")) if (pulled / coll).exists() else []:
            raw = json.loads(f.read_text())
            # ArtifactData saves either the bare document or {id, data, version}
            doc = raw.get("data") if isinstance(raw, dict) and "data" in raw and "version" in raw else raw
            out[coll][f.stem] = {"doc": doc, "version": raw.get("version") if isinstance(raw, dict) else None}
    return out


def household_of(doc_id: str, doc: dict) -> str:
    if doc.get("household"):
        return str(doc["household"])
    m = re.search(r"s(\d+)", doc_id)
    return f"hh_s{m.group(1)}" if m else "hh_unknown"


def player_of(doc_id: str, doc: dict) -> str:
    if doc.get("player"):
        return str(doc["player"])
    parts = doc_id.split("__")
    if len(parts) >= 2 and not re.match(r"\d{4}-", parts[1]):
        return parts[1]
    return "anonymous"


def thoughts_md(doc_id: str, doc: dict, notes: List[dict], hh: str, player: str) -> str:
    A = doc.get("answers", [])
    looks = doc.get("looks", [])
    L = [l for l in looks if not l.get("failed")] if isinstance(looks, list) else []   # the g1 page stored a count
    n_total = doc.get("n_questions") or (doc.get("perDay", 6) * 4)
    finished = len(A) >= n_total
    lines = [f"# {hh} · {player} · run {doc_id}", "",
             f"started {doc.get('started', '')[:16].replace('T', ' ')} · last activity {(doc.get('updated') or doc.get('archived') or '')[:16].replace('T', ' ')} · "
             f"{len(A)}/{n_total} answered · {len(L)} looks · budget {sum(l.get('cost', 0) for l in L)}"
             + (f" · **{sum(1 for a in A if a.get('hit'))}/{len(A)} right**" if finished else " · in progress (verdicts sealed)"), ""]
    lines += ["## Per-question thinking", ""]
    for a in A:
        d = DAYS[a["day"]] if 0 <= a.get("day", 0) < len(DAYS) else str(a.get("day"))
        verdict = (" ✓" if a.get("hit") else f" ✗ (truth {a.get('truth')})") if finished else ""
        lines.append(f"**Q{a['i'] + 1} {d} {a['minute'] // 60:02d}:{a['minute'] % 60:02d} `{a['obj']}` → {a.get('guess')}**{verdict} · {a.get('looksUsed', 0)} look(s)")
        if a.get("note"):
            lines.append(f"> {a['note'].strip()}")
        lines.append("")
    mine = [n for n in notes if not n.get("player") or n.get("player") == player]
    if mine:
        lines += ["## Household notes", ""]
        for n in mine:
            lines.append(f"- {n.get('ts', '')[:16].replace('T', ' ')} · {n.get('ctx', '')} — {n.get('text', '').strip()}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulled", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args(argv)
    data = load_pulled(a.pulled)
    notes_by_hh: Dict[str, List[dict]] = {}
    for did, rec in data["notes"].items():
        notes_by_hh.setdefault(household_of(did, rec["doc"]), []).extend(rec["doc"].get("entries", []))
    index_rows = []
    written = 0
    for coll in ("chrono", "runs"):
        for did, rec in data[coll].items():
            doc = rec["doc"]
            if not isinstance(doc, dict) or not doc.get("answers"):
                continue
            hh, player = household_of(did, doc), player_of(did, doc)
            pdir = a.out / slug(player)
            pdir.mkdir(parents=True, exist_ok=True)
            base = pdir / f"{hh}__{did}"
            meta = {"source": f"artifact db {ARTIFACT}", "collection": coll, "doc_id": did,
                    "version": rec["version"], "exported": datetime.datetime.now().isoformat(timespec="seconds"),
                    "run": doc, "notes": notes_by_hh.get(hh, [])}
            jf = base.with_suffix(".json")
            if jf.exists():
                try:
                    old = json.loads(jf.read_text())
                    if old.get("version") == rec["version"] and rec["version"] is not None:
                        index_rows.append((player, hh, did, coll, doc))
                        continue
                except Exception:
                    pass
            jf.write_text(json.dumps(meta, indent=1))
            base.with_suffix(".thoughts.md").write_text(thoughts_md(did, doc, notes_by_hh.get(hh, []), hh, player))
            written += 1
            index_rows.append((player, hh, did, coll, doc))
    index_rows.sort(key=lambda r: r[4].get("started", ""), reverse=True)
    L = ["# Human runs (exported from the inspector artifact)", "",
         f"Refreshed {datetime.datetime.now().isoformat(timespec='minutes')} · {len(index_rows)} runs", "",
         "| player | household | started | progress | score | looks | store | doc |", "|---|---|---|---|---|---|---|---|"]
    for player, hh, did, coll, doc in index_rows:
        A = doc.get("answers", []); n_total = doc.get("n_questions") or (doc.get("perDay", 6) * 4)
        fin = len(A) >= n_total
        L.append(f"| {player} | {hh} | {doc.get('started', '')[:16].replace('T', ' ')} | {len(A)}/{n_total} | "
                 f"{(str(sum(1 for x in A if x.get('hit'))) + '/' + str(len(A))) if fin else 'sealed'} | "
                 f"{(sum(1 for l in doc['looks'] if not l.get('failed')) if isinstance(doc.get('looks'), list) else doc.get('looks', 0))} | {coll} | `{slug(player)}/{hh}__{did}` |")
    (a.out / "INDEX.md").write_text("\n".join(L) + "\n")
    print(f"{written} run(s) written/updated, {len(index_rows)} indexed -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

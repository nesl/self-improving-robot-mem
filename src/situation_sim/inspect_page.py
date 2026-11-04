"""Build the human-inspection page: generate several households, then embed
their runs into one self-contained HTML file.

    cd src && python3 -m situation_sim.inspect_page --seeds 0 1 2 3 4 5 \
        --out ../data/situation_sim

Writes data/situation_sim/runs/hh_s<N>/ (the usual four files per run) and
data/situation_sim/inspect/index.html.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from situation_sim.run import generate

TEMPLATE = pathlib.Path(__file__).with_name("inspect_template.html")
PLACEHOLDER = "/*__RUNS_JSON__*/"


def load_run(d: pathlib.Path) -> dict:
    state = json.loads((d / "hidden_state.json").read_text())
    rows = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
    truth = [{k: r[k] for k in ("object_id", "t", "receptacle_id", "cause", "causes", "whim", "reason", "carrier", "intended")
              if k in r} for r in rows if r["kind"] == "truth"]
    residents = [{k: r[k] for k in ("resident_id", "t", "room")} for r in rows if r["kind"] == "resident"]
    trace = json.loads((d / "trace.json").read_text())
    state.pop("event_library", None)
    state.pop("episode_library", None)
    return {"state": state, "truth": truth, "residents": residents, "trace": trace["days"]}


def build(seeds, out: pathlib.Path, n_days: int = 5) -> pathlib.Path:
    runs = {}
    for seed in seeds:
        d = out / "runs" / f"hh_s{seed}"
        generate(seed, d, n_days)
        runs[str(seed)] = load_run(d)
    first = json.loads((out / "runs" / f"hh_s{seeds[0]}" / "hidden_state.json").read_text())
    payload = json.dumps({"runs": runs, "events": first["event_library"],
                          "episodes": first.get("episode_library", {}),
                          "default_seed": str(seeds[0])},
                         separators=(",", ":"), sort_keys=True)
    html = TEMPLATE.read_text()
    assert PLACEHOLDER in html
    page = out / "inspect" / "index.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(html.replace(PLACEHOLDER, payload))
    return page


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--days", type=int, default=5)
    a = ap.parse_args(argv)
    page = build(a.seeds, a.out, a.days)
    print(f"wrote {page} ({page.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

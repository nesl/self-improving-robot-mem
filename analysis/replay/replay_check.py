#!/usr/bin/env python3
"""Rerun one existing cell from the response cache and count calls that missed it.

    PYTHONPATH=src python3 analysis/replay/replay_check.py OUT \
        [WAVE ARM HOUSEHOLD QPD LAST_DAY BANKS]

Replays `incremental edits` on hh_s2_t03, 8 a day, 31 days, the ten-home banks, through the same
overnight_wave.run_one_arm the paper cell used. The client is replay-only: a miss returns no text
and is counted, so zero misses means the code change left the arm's prompts unchanged.
"""
import json
import os
import pathlib
import sys

from baselines.patrol.llm import LLMClient
from self_improve import overnight_wave as ow
from self_improve.frozen_household import FrozenHousehold

BANKS = pathlib.Path("results/self_improve/varied_homes/ten_homes/banks")
# The response cache is not in the public release; point $LLM_CACHE at a copy of it.
CACHE = pathlib.Path(os.environ.get("LLM_CACHE", "llm_cache/self_improve"))


class CountingReplay(LLMClient):
    def __init__(self, cache):
        super().__init__(cache, replay_only=True)
        self.misses = []

    def complete(self, messages, schema, max_tokens):
        text, usage = super().complete(messages, schema, max_tokens)
        if text is None:
            self.misses.append(self.key(messages, schema, max_tokens))
        return text, usage


def main() -> int:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/replay_check")
    wave, arm, hh, qpd, last, banks = (sys.argv[2:8] if len(sys.argv) > 7 else
                                       ("overnight_wave", "incremental edits", "hh_s2_t03", 8, 31, BANKS))
    qpd, last = int(qpd), int(last)
    paper_cell = ow.cell_dir(pathlib.Path("results/self_improve") / wave, arm, hh)
    client = CountingReplay(CACHE)
    household = FrozenHousehold(pathlib.Path(banks) / f"{hh}.jsonl")
    try:
        ow.run_one_arm(household, arm, client, out / "cell", last, qpd, 3, 0)
    except AssertionError as e:   # a gate can refuse a cell whose calls missed; the count still stands
        print("gate:", e)
    new = [json.loads(l) for l in (out / "cell" / "searches.jsonl").open() if '"search"' in l]
    old = [r for r in (json.loads(l) for l in (paper_cell / "searches.jsonl").open() if '"search"' in l)
           if r["day"] <= last]
    same = sum(1 for a, b in zip(new, old)
               if (a["rooms_opened"], a["answer_place"], a["found_at_step"]) ==
               (b["rooms_opened"], b["answer_place"], b["found_at_step"]))
    res = {"cache_hits": client.stats["cached"], "new_model_calls_needed": len(client.misses),
           "searches_replayed": len(new), "searches_in_paper_cell": len(old),
           "searches_identical_to_paper_cell": same}
    print(json.dumps(res, indent=1))
    (out / "replay_check.json").write_text(json.dumps(res, indent=1))
    return 0 if not client.misses and same == len(old) == len(new) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Ten-household run: the day-14 fall on spell-1 movers, and recovery by day 17.

Recovery is each method against ITS OWN days 8 to 13 level, on the same objects, so a method that
was never good does not look recovered. Reported per household and then averaged, with 2 SE.

    python3 analysis/day14_and_recovery.py
"""
import glob
import json
import statistics

WAVE = "results/self_improve/overnight_wave/cells"
NAME = {"the_log_and_notes_about_the_routine": "log and notes",
        "last_seen_no_model": "last-seen",
        "incremental_edits": "claim store",
        "claim_store_told_if_it_was_right": "reduced ACE",
        "a_small_working_memory_and_an_archive": "small working memory",
        "prior_only_no_notes": "notes hidden"}


def movers_rows(arm, home, days):
    out = []
    for line in open(f"{WAVE}/{arm}/{home}/searches.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "search" and r.get("is_a_mover") and r["day"] in days:
            out.append(r)
    return out


def share(rows):
    return 100 * sum(1 for r in rows if r.get("found_at_step") == 1) / len(rows) if rows else None


def main() -> int:
    homes = sorted(p.split("/")[-1] for p in glob.glob(f"{WAVE}/last_seen_no_model/*"))
    print("first room right on spell-1 movers\n")
    print(f"{'method':24s}{'days 8-13':>11s}{'day 14':>9s}{'day 17':>9s}"
          f"{'day 17 as % of 8-13':>22s}{'2 SE':>8s}   questions on day 14")
    for arm, label in NAME.items():
        before, d14, d17, ratio, n14 = [], [], [], [], 0
        for home in homes:
            b = share(movers_rows(arm, home, set(range(8, 14))))
            a14 = movers_rows(arm, home, {14})
            a17 = share(movers_rows(arm, home, {17}))
            n14 += len(a14)
            if b:
                before.append(b)
                if a14:
                    d14.append(share(a14))
                if a17 is not None:
                    d17.append(a17)
                    ratio.append(100 * a17 / b)
        tse = 2 * statistics.stdev(ratio) / len(ratio) ** 0.5 if len(ratio) > 1 else float("nan")
        print(f"{label:24s}{statistics.mean(before):11.1f}{statistics.mean(d14):9.1f}"
              f"{statistics.mean(d17):9.1f}{statistics.mean(ratio):22.0f}{tse:8.0f}   {n14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

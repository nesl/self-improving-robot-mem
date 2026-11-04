#!/usr/bin/env python3
"""Camera-ready figures for the CSIR paper, drawn to Oliver's figure brief of 2026-11-04.

Four figures, each a vector PDF at its final print size (never rescaled by LaTeX), with a .json
beside it holding every plotted value. Every value the brief lists is asserted against the data
the figure plots; a mismatch raises and nothing is reported as drawn.

    brief name   file                                  size (in)
    Figure 2     figureA1_first_illness.pdf            5.5 x 2.2
    Figure 3     figure2_both_illnesses.pdf            5.5 x 1.8
    Figure 4     figureA2_what_the_notes_held.pdf      3.4 x 1.9
    Figure 5     figure3_told_the_cause.pdf            5.5 x 1.5

CPU only, no model calls; reads finished cells only. Run from prior_work/:

    PYTHONPATH=src python3 results/self_improve/paper/scripts/camera_ready_figures.py

The text width is 5.5 in (396 pt). The template is not on this machine; 396 pt is what the
earlier draft PDF's justified body text spans (x = 108 to 504 pt on a letter page).
"""
import collections
import json
import os
import pathlib
import statistics
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import FancyBboxPatch                        # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_data as D                                              # noqa: E402
from self_improve import search_driven as sd                        # noqa: E402
from self_improve.frozen_household import FrozenHousehold           # noqa: E402

R = pathlib.Path("results/self_improve")
# Public release: figures are written to $FIG_OUT (default out/figures/camera_ready); the paper's copies are in
# results/self_improve/paper/figures/camera_ready/.
OUT = pathlib.Path(os.environ.get("FIG_OUT", "out/figures/camera_ready"))
TEXT_WIDTH_IN = 5.5

# ------------------------------------------------------------------ style, one set for all four
FONT = 8.0          # body of every figure
SMALL = 7.0         # the floor: "ill" labels, brackets, in-segment counts
INK, MUTED = "#1a1a1a", "#5f5f5f"
SHADE = "#e9e9e9"

# REVISED BRIEF (2026-11-04, v2): the history arms and "notes hidden" are out of the paper, and the
# word "notes" appears in no figure text (checked on every PDF by FORBIDDEN below).
NAMES = ["last-seen", "last-seen, updated nightly", "claim store", "claim store + today's sightings",
         "claim store + history", "ACE-style"]
COLOR = {"last-seen": "#000000", "last-seen, updated nightly": "#999999",
         "claim store": "#0072B2", "claim store + today's sightings": "#4A9FD8",
         "claim store + history": "#9CCBEB", "ACE-style": "#CC79A7"}
FORBIDDEN = [r"\bnotes\b", r"\bnote\b", r"history only", r"history and", r"notes hidden"]
# a distinct dash pattern and marker per method, so every figure still reads in greyscale
DASH = {"last-seen": (0, (5.0, 1.6)), "last-seen, updated nightly": (0, (2.6, 1.6)),
        "claim store": (0, (1.0, 1.1)), "claim store + today's sightings": (0, (3.0, 1.0, 1.0, 1.0)),
        "claim store + history": (0, (3.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
        "ACE-style": (0, (1.0, 1.0, 3.0, 1.0))}
MARK = {"last-seen": "o", "last-seen, updated nightly": "v", "claim store": "^",
        "claim store + today's sightings": "<", "claim store + history": ">",
        "ACE-style": "P"}

TEN_HOMES = "ten households, 8 questions a day, 31 days, Qwen"
ARMS_10 = {   # method -> (cells directory, arm directory)
    "last-seen": (R / "overnight_wave/cells", "last_seen_no_model"),
    "last-seen, updated nightly": (R / "overnight_wave/cells", "last_seen_updated_each_night"),
    "claim store": (R / "camera_ready/qwen_ten_homes_8_a_day/cells", "incremental_edits"),
    "claim store + today's sightings": (R / "camera_ready/qwen_ten_homes_8_a_day/cells",
                                        "claim_store_plus_today's_sightings"),
    "claim store + history": (R / "camera_ready/qwen_ten_homes_8_a_day/cells", "claim_store_plus_the_log"),
}
# Table 1's other rows, for the day-14 range only (not drawn). The first two and notes hidden are
# out of the paper now; they stay here only so the range can be given with and without them.
DROPPED_FROM_PAPER = ("history and notes", "history only", "notes hidden")
TABLE1_OTHERS = {
    "history and notes": (R / "overnight_wave/cells", "the_log_and_notes_about_the_routine"),
    "history only": (R / "wave_log_only/cells", "log_only_no_notes"),
    "claim store (Table 1 older-code cell)": (R / "overnight_wave/cells", "incremental_edits"),
    "reduced ACE": (R / "overnight_wave/cells", "claim_store_told_if_it_was_right"),
    "small working memory": (R / "overnight_wave/cells", "a_small_working_memory_and_an_archive"),
    "notes hidden": (R / "overnight_wave/cells", "prior_only_no_notes"),
}
ARMS_50 = {
    "last-seen": (R / "wave_the_second_illness/cells", "last_seen_no_model"),
    "claim store": (R / "wave_the_second_illness/cells", "incremental_edits"),
    "ACE-style": (R / "wave_the_second_illness_ACE_fixed/cells", "ACE_as_published"),
}


def apply_style():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": FONT, "axes.labelsize": FONT, "axes.titlesize": FONT,
        "xtick.labelsize": FONT, "ytick.labelsize": FONT, "legend.fontsize": FONT,
        "font.weight": "normal",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.0,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        # the page IS the figure size: no tight box, which would crop and change the size
        "savefig.bbox": None, "savefig.pad_inches": 0.0,
        "figure.constrained_layout.use": False,
    })


STALE_LOCKS = []


class Mismatch(AssertionError):
    pass


def check(label, got, want, tol=0.05):
    """The brief's value against the plotted one; tol covers one-decimal rounding."""
    if abs(got - want) > tol:
        raise Mismatch(f"{label}: plotted {got:.4f}, brief/paper says {want}")
    print(f"  ok  {label}: {got:.2f} (brief {want})")


def rows(cells, arm, home):
    """Search rows of one finished cell. No fallback to another wave: the path must exist."""
    path = pathlib.Path(cells) / arm / home / "searches.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    lock = path.parent / "RUNNING.lock"
    if lock.exists():
        # several finished September cells still carry a lock; it is stale when its process is gone
        # and the cell wrote its cell.json at the end of the run
        pid = json.load(open(lock)).get("pid")
        alive = pid and pathlib.Path(f"/proc/{pid}").exists()
        if alive or not (path.parent / "cell.json").exists():
            raise RuntimeError(f"{path.parent} is still running")
        STALE_LOCKS.append(str(path.parent))
    out = [json.loads(line) for line in path.open()]
    return [r for r in out if r.get("kind") == "search"]


def frr(r):
    return r.get("found_at_step") == 1


def pct(rs):
    return 100.0 * sum(frr(r) for r in rs) / len(rs) if rs else None


def by_day(rs_by_home, objects_of, days):
    """Per household share by day on the given objects, then the mean over households that were
    asked about those objects that day (the same rule as paper_figures.by_day)."""
    per_home = {}
    for h, rs in rs_by_home.items():
        keep = objects_of(h)
        day_rows = collections.defaultdict(list)
        for r in rs:
            if r["object_id"] in keep:
                day_rows[r["day"]].append(r)
        per_home[h] = {d: pct(day_rows.get(d, [])) for d in days}
    line, n_homes = [], []
    for d in days:
        vals = [per_home[h][d] for h in per_home if per_home[h][d] is not None]
        line.append(statistics.mean(vals) if vals else None)
        n_homes.append(len(vals))
    return line, per_home, n_homes


HEADROOM = 110      # y axis runs 0-100; the strip above 100 holds only the "ill" labels


def shade_ill(ax, spans):
    """Grey over the illness days, and "ill" inside the top of the axes, in a strip above 100
    where no line can run (at 0.975 of a 0-100 axis it sat on lines that reach 98)."""
    for first, last in spans:
        ax.axvspan(first - 0.5, last + 0.5, color=SHADE, lw=0, zorder=0)
        ax.text((first + last) / 2, HEADROOM - 1.2, "ill", ha="center", va="top",
                fontsize=SMALL, color=MUTED, zorder=1)


def y_0_100(ax, ticks=(0, 25, 50, 75, 100)):
    ax.set_ylim(0, HEADROOM)
    ax.set_yticks(list(ticks))
    ax.spines["left"].set_bounds(0, 100)


def plot_line(ax, days, values, name, lw=1.15, ms=2.6):
    xs = [d for d, v in zip(days, values) if v is not None]
    ys = [v for v in values if v is not None]
    face = "white" if name == "last-seen, updated nightly" else COLOR[name]
    # the palest colour gets a darker marker edge so its points hold up over the grey shading
    edge = "#6a9fc4" if name == "claim store + history" else COLOR[name]
    ax.plot(xs, ys, color=COLOR[name], linestyle=DASH[name], lw=lw, marker=MARK[name], ms=ms,
            markerfacecolor=face, markeredgecolor=edge, markeredgewidth=0.5, zorder=4,
            dash_capstyle="butt")


def handle(name, lw=1.15, ms=3.0):
    face = "white" if name == "last-seen, updated nightly" else COLOR[name]
    edge = "#6a9fc4" if name == "claim store + history" else COLOR[name]
    return plt.Line2D([], [], color=COLOR[name], linestyle=DASH[name], lw=lw, marker=MARK[name],
                      ms=ms, markerfacecolor=face, markeredgecolor=edge,
                      markeredgewidth=0.5, label=name)


def save(fig, stem, payload, size):
    OUT.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    assert abs(w - size[0]) < 1e-9 and abs(h - size[1]) < 1e-9, (stem, w, h)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    import re
    import subprocess
    text = subprocess.run(["pdftotext", "-raw", str(OUT / f"{stem}.pdf"), "-"], capture_output=True,
                          text=True, check=True).stdout
    flat = " ".join(text.split())
    for pat in FORBIDDEN:
        if re.search(pat, flat, re.I):
            (OUT / f"{stem}.pdf").rename(OUT / f"{stem}.REJECTED.pdf")
            raise Mismatch(f"{stem}.pdf contains forbidden text {pat!r}: {flat[:200]!r}")
    print(f"  ok  pdftotext: no {', '.join(FORBIDDEN)} in {stem}.pdf")
    payload = {"figure_file": f"{stem}.pdf", "size_in": list(size), **payload}
    (OUT / f"{stem}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"  wrote {OUT / stem}.pdf and .json")


def r1(x):
    return None if x is None else round(x, 2)


# ===================================================================== Figure 2 (figureA1)
def two_se_paired(a_by_home, b_by_home, homes):
    d = [b_by_home[h] - a_by_home[h] for h in homes]
    return statistics.mean(d), 2 * statistics.stdev(d) / len(d) ** 0.5, d


def figure_2():
    print("\nFigure 2: figureA1_first_illness.pdf")
    homes = D.HOMES_10
    banks = D.BANKS_10
    movers = {h: sd.the_movers(FrozenHousehold(banks / f"{h}.jsonl")) for h in homes}
    data = {n: {h: rows(c, a, h) for h in homes} for n, (c, a) in ARMS_10.items()}
    data_other = {n: {h: rows(c, a, h) for h in homes} for n, (c, a) in TABLE1_OTHERS.items()}

    # first room right is found_at_step == 1 (the paper's definition). item2_first_room_right.py
    # scores rooms_opened[0] == true_room instead, so check the two agree on every row plotted.
    for n, by_home in list(data.items()) + list(data_other.items()):
        for h, rs in by_home.items():
            for r in rs:
                alt = bool(r["rooms_opened"]) and r["rooms_opened"][0] == r["true_room"]
                if alt != frr(r):
                    raise Mismatch(f"{n} {h} {r['question_id']}: found_at_step and rooms_opened disagree")
            if len(rs) != 248:
                raise Mismatch(f"{n} {h}: {len(rs)} questions, expected 248 (8 a day x 31 days)")

    # ---- left panel: by day on the movers
    days = list(range(1, 32))
    left_names = ["last-seen", "claim store", "claim store + history"]
    curves = {}
    for n in left_names:
        line, per_home, n_homes = by_day(data[n], lambda h: movers[h], days)
        curves[n] = {"values": [r1(v) for v in line], "households_with_mover_questions": n_homes}

    # ---- right panel: whole-run first room right, pooled over 2,480 questions
    whole = {n: pct([r for h in homes for r in data[n][h]]) for n in ARMS_10}
    per_home = {n: {h: pct(data[n][h]) for h in homes} for n in ARMS_10}
    brief = {"claim store + history": 84.1, "claim store + today's sightings": 79.9,
             "claim store": 75.8, "last-seen": 82.1}
    for n, v in brief.items():
        check(f"whole-run first room right, {n}", whole[n], v)
    s1, s1se, _ = two_se_paired(per_home["claim store"], per_home["claim store + today's sightings"], homes)
    s2, s2se, _ = two_se_paired(per_home["claim store + today's sightings"],
                                per_home["claim store + history"], homes)
    check("step claim store -> + today's sightings, mean", s1, 4.1)
    check("step claim store -> + today's sightings, 2 SE", s1se, 2.8)
    check("step + today's sightings -> + history, mean", s2, 4.2)
    check("step + today's sightings -> + history, 2 SE", s2se, 3.0)
    nightly = whole["last-seen, updated nightly"]
    print(f"  REPORT last-seen, updated nightly, whole run: {nightly:.2f}")

    # ---- the day-14 range across Table 1's methods (reported, not drawn)
    def day14(by_home, drop_zero_baseline=False):
        vals = []
        for h in homes:
            m = [r for r in by_home[h] if r["object_id"] in movers[h]]
            base = pct([r for r in m if 8 <= r["day"] <= 13])
            at = pct([r for r in m if r["day"] == 14])
            if drop_zero_baseline and not base:   # day14_and_recovery.py's `if b:`
                continue
            vals.append(at)
        return statistics.mean(vals), len(vals)
    every = {"last-seen": data["last-seen"], **data_other,
             "last-seen, updated nightly": data["last-seen, updated nightly"],
             "claim store (current-code rerun)": data["claim store"],
             "claim store + today's sightings": data["claim store + today's sightings"],
             "claim store + history": data["claim store + history"]}
    d14 = {}
    for label, by_home in every.items():
        ten, _ = day14(by_home)
        skip, n_used = day14(by_home, drop_zero_baseline=True)
        d14[label] = {"ten_household_mean": r1(ten),
                      "day14_and_recovery_py_rule": r1(skip), "households_used_by_that_rule": n_used,
                      "dropped_from_paper": label in DROPPED_FROM_PAPER}
    # Table 1 as it stood: six rows (paper/table1_ten_homes.txt) plus history only (E3)
    table1_all = ["last-seen", "history and notes", "claim store (Table 1 older-code cell)",
                  "reduced ACE", "small working memory", "notes hidden", "history only"]
    table1_kept = [m for m in table1_all if m not in DROPPED_FROM_PAPER]
    kept_plus_new = table1_kept + ["last-seen, updated nightly", "claim store (current-code rerun)",
                                   "claim store + today's sightings", "claim store + history"]
    rng = lambda keys, k: [min(d14[x][k] for x in keys), max(d14[x][k] for x in keys)]
    day14_report = {
        "per_method": d14,
        "table1_methods_remaining": table1_kept,
        "table1_methods_all": table1_all,
        "range_remaining_ten_household_mean": rng(table1_kept, "ten_household_mean"),
        "range_remaining_day14_and_recovery_rule": rng(table1_kept, "day14_and_recovery_py_rule"),
        "range_all_ten_household_mean": rng(table1_all, "ten_household_mean"),
        "range_all_day14_and_recovery_rule": rng(table1_all, "day14_and_recovery_py_rule"),
        "range_remaining_plus_camera_ready_arms_ten_household_mean": rng(kept_plus_new, "ten_household_mean"),
        "note": ("28.8 and 37.2 come from paper/day14_and_recovery.txt, whose script skips a household "
                 "when its days 8-13 mover share is 0 (`if b:`): the no-memory arm is then a mean of 7 "
                 "households and the older claim store of 9. Over all ten households they are 31.0 and 36.0."),
    }
    print("  REPORT day 14 on movers, Table 1 methods:")
    for k, v in d14.items():
        print(f"     {k:38s} ten-household mean {v['ten_household_mean']:5.1f}   "
              f"day14_and_recovery rule {v['day14_and_recovery_py_rule']:5.1f} "
              f"({v['households_used_by_that_rule']} households)"
              + ("   [dropped from paper]" if v["dropped_from_paper"] else ""))
    for k, v in day14_report.items():
        if k.startswith("range"):
            print(f"     {k}: {v[0]:.1f} to {v[1]:.1f}")

    # ---- draw
    fig = plt.figure(figsize=(5.5, 2.2))
    # WIDTHS FROM MEASURED TEXT, in inches: from the right edge, the one-line step text (0.43),
    # the bracket (0.09), the value label of the 84.1 dot (0.20, starting 0.9 points right of it);
    # the 0.8 in axes for 70-90; then the longest method name (1.37) and a gap to the left panel.
    # One-line names and brackets beside the labels make the split about 50/50, not 60/40.
    axL = fig.add_axes([0.083, 0.165, 0.386, 0.81])
    axR = fig.add_axes([0.7425, 0.165, 0.1455, 0.81])
    shade_ill(axL, [(14, 23)])
    for n in left_names:
        plot_line(axL, days, curves[n]["values"], n)
    axL.set_xlim(0.5, 31.5)
    y_0_100(axL)
    axL.set_xticks([1, 14, 24, 31])
    axL.set_xlabel("day")
    axL.set_ylabel("first room right, moved objects (%)")
    axL.legend(handles=[handle(n) for n in left_names], loc="lower left",
               bbox_to_anchor=(0.0, 0.0), handlelength=2.6, handletextpad=0.5,
               labelspacing=0.25, borderaxespad=0.3, fontsize=FONT)

    order = ["claim store + history", "claim store + today's sightings", "claim store", None,
             "last-seen", "last-seen, updated nightly"]
    ypos = {n: i for i, n in enumerate(order) if n}
    labels = {}
    for n, y in ypos.items():
        axR.plot([70, whole[n]], [y, y], color="#e2e2e2", lw=0.6, zorder=1)
        face = "white" if n == "last-seen, updated nightly" else COLOR[n]
        axR.plot([whole[n]], [y], marker=MARK[n], ms=4.2, color=COLOR[n], markerfacecolor=face,
                 markeredgecolor=COLOR[n] if n != "claim store + history" else "#6a9fc4",
                 markeredgewidth=0.7, zorder=4, linestyle="none", clip_on=False)
        labels[n] = axR.text(whole[n] + 0.9, y, f"{whole[n]:.1f}", ha="left", va="center",
                             fontsize=FONT, clip_on=False)
    axR.set_ylim(len(order) - 0.45, -0.55)
    axR.set_yticks([ypos[n] for n in ypos])
    axR.set_yticklabels(list(ypos))
    axR.tick_params(axis="y", length=0)
    axR.set_xlim(70, 90)
    axR.set_xticks([70, 80, 90])
    axR.set_xticks([75, 85], minor=True)
    axR.tick_params(axis="x", which="minor", length=1.6, width=0.5)
    axR.set_xlabel("first room right, whole run (%)", ha="right", x=1.0)
    axR.spines["left"].set_visible(False)

    # STEP BRACKETS, TO THE RIGHT OF THE VALUE LABELS (Oliver: squeezed between the labels and
    # the dots they were confusing). Both share one x, just clear of the widest value label of the
    # three claim store rows; each spans the two rows it joins, its text on one line beside it.
    fig.canvas.draw()
    to_data = axR.transData.inverted()
    right_px = max(labels[n].get_window_extent().x1 for n in order[:3])
    bx = to_data.transform((right_px + 0.06 * fig.dpi, 0))[0]
    tick = to_data.transform((right_px + 0.11 * fig.dpi, 0))[0] - bx
    gap = 0.07                                   # keeps the two brackets apart at the shared row

    def bracket(y_a, y_b, text):
        top, bot = min(y_a, y_b), max(y_a, y_b)
        axR.plot([bx - tick, bx, bx, bx - tick], [top + gap, top + gap, bot - gap, bot - gap],
                 color=MUTED, lw=0.6, solid_capstyle="butt", zorder=3, clip_on=False)
        return axR.text(bx + tick * 0.8, (top + bot) / 2, text, ha="left", va="center",
                        fontsize=SMALL, color=INK, clip_on=False)
    steps = [bracket(ypos["claim store"], ypos["claim store + today's sightings"],
                     f"+{s1:.1f} \u00b1 {s1se:.1f}"),
             bracket(ypos["claim store + today's sightings"], ypos["claim store + history"],
                     f"+{s2:.1f} \u00b1 {s2se:.1f}")]
    fig.canvas.draw()
    for t in steps:
        if t.get_window_extent().x1 > fig.bbox.x1 - 0.01 * fig.dpi:
            raise Mismatch(f"step text {t.get_text()!r} runs off the right edge")
    left_edge = min(t.get_window_extent().x0 for t in axR.get_yticklabels())
    if left_edge < axL.get_window_extent().x1 + 0.08 * fig.dpi:
        raise Mismatch("right panel's method names run into the left panel")

    payload = {
        "source": TEN_HOMES,
        "measure": "first room right = found_at_step == 1",
        "left_panel": {
            "objects": "spell-1 movers, search_driven.the_movers on varied_homes/ten_homes/banks",
            "days": days, "aggregation": "per household share by day, mean over households asked about movers that day",
            "illness_days": [14, 23],
            "lines": {n: {"cells": str(ARMS_10[n][0] / ARMS_10[n][1]), **curves[n]} for n in left_names}},
        "right_panel": {
            "aggregation": "pooled over all 2,480 questions (248 per household)",
            "rows_top_to_bottom": [None if n is None else
                                   {"method": n, "value": r1(whole[n]),
                                    "cells": str(ARMS_10[n][0] / ARMS_10[n][1]),
                                    "per_household": {h: r1(per_home[n][h]) for h in homes}}
                                   for n in order],
            "steps": {"claim store -> claim store + today's sightings": {"mean": r1(s1), "two_se": r1(s1se)},
                      "claim store + today's sightings -> claim store + history": {"mean": r1(s2), "two_se": r1(s2se)}},
            "step_rule": "paired per-household difference, mean and 2 SE over ten households"},
        "reported_last_seen_updated_nightly_whole_run": r1(nightly),
        "day14_on_movers_table1": day14_report,
    }
    save(fig, "figureA1_first_illness", payload, (5.5, 2.2))
    return payload


# ===================================================================== Figure 3 (figure2)
def figure_3():
    print("\nFigure 3: figure2_both_illnesses.pdf")
    homes = D.HOMES_50
    both = {h: D.both_spell_movers(h) for h in homes}
    n_both = sum(len(v) for v in both.values())
    if n_both != 14:
        raise Mismatch(f"both-spell movers: {n_both}, the brief says 14")
    print(f"  ok  both-spell movers: {n_both}")
    data = {n: {h: rows(c, a, h) for h in homes} for n, (c, a) in ARMS_50.items()}
    for n, by_home in data.items():
        for h, rs in by_home.items():
            if max(r["day"] for r in rs) < 49:
                raise Mismatch(f"{n} {h}: ends on day {max(r['day'] for r in rs)}")
            per_day = collections.Counter(r["day"] for r in rs)
            if set(per_day.values()) != {24}:
                raise Mismatch(f"{n} {h}: questions a day {sorted(set(per_day.values()))}, expected 24")
    days = list(range(1, 50))
    curves, drops = {}, {}
    for n in ARMS_50:
        line, per_home, n_homes = by_day(data[n], lambda h: both[h], days)
        if min(n_homes) < 3:
            raise Mismatch(f"{n}: a day with mover questions in fewer than 3 households")
        curves[n] = {"values": [r1(v) for v in line]}
        drops[n] = {}
        for onset, before in ((14, range(10, 14)), (32, range(28, 32))):
            per = []
            for h in homes:
                base = statistics.mean(per_home[h][d] for d in before)
                per.append(base - per_home[h][onset])
            idx = {d: i for i, d in enumerate(days)}
            base_curve = statistics.mean(line[idx[d]] for d in before)
            drops[n][f"day{onset}"] = {
                "baseline_days": [before[0], before[-1]], "baseline_mean": r1(base_curve),
                "onset_value": r1(line[idx[onset]]), "drop": r1(statistics.mean(per)),
                "drop_per_household": [r1(x) for x in per],
                "two_se": r1(2 * statistics.stdev(per) / len(per) ** 0.5)}
            # the curve's own drop equals the household mean, since every day has all 3 households
            assert abs((base_curve - line[idx[onset]]) - statistics.mean(per)) < 1e-9
        print(f"  ok  {n:18s} drop day 14 {drops[n]['day14']['drop']:5.1f}  "
              f"day 32 {drops[n]['day32']['drop']:5.1f}")

    fig = plt.figure(figsize=(5.5, 1.8))
    ax = fig.add_axes([0.083, 0.2, 0.905, 0.685])
    shade_ill(ax, [(14, 23), (32, 41)])
    order = ["last-seen", "claim store", "ACE-style"]
    for n in order:
        plot_line(ax, days, curves[n]["values"], n, lw=1.0, ms=2.3)
    ax.set_xlim(0.5, 49.5)
    y_0_100(ax)
    ax.set_xticks([1, 14, 24, 32, 42, 49])
    ax.set_xlabel("day", labelpad=1.0)
    ax.set_ylabel("first room right (%)")
    fig.legend(handles=[handle(n) for n in order], loc="upper center", ncol=len(order),
               bbox_to_anchor=(0.535, 1.005), handlelength=2.6, handletextpad=0.5,
               columnspacing=1.6, borderaxespad=0.0, fontsize=FONT)
    payload = {
        "source": "three 50-day households (hh_s2, hh_s32, hh_s48 _t03), 24 questions a day, days 1-49, Qwen",
        "objects": "the 14 objects that change room in both illnesses (paper_data.both_spell_movers)",
        "measure": "first room right = found_at_step == 1; per household share by day, mean over the 3 households",
        "illness_days": [[14, 23], [32, 41]], "days": days,
        "lines": {n: {"cells": str(ARMS_50[n][0] / ARMS_50[n][1]), **curves[n]} for n in order},
        "drops": drops,
        "drop_rule": "mean over the 4 preceding days minus the onset day, per household, then the mean over "
                     "households",
    }
    save(fig, "figure2_both_illnesses", payload, (5.5, 1.8))
    return payload


# ===================================================================== Figure 4 (figureA2)
def figure_4():
    print("\nFigure 4: figureA2_what_the_notes_held.pdf")
    audit = json.load(open(os.environ.get("NIGHT31_AUDIT", R / "paper/night31_audit.json")))
    # the bars are the cells the current figure draws (the ten-household overnight_wave arms);
    # its "ACE-Style Playbook" bar is the reduced ACE arm, claim_store_told_if_it_was_right
    bars = [("claim store", "claim store"), ("ACE-style", "reduced ACE")]
    conds = [("always", "always true", "#3b3b3b", "white"),
             ("a time of day", "a time of day", "#7d7d7d", "white"),
             ("a person home or away", "a person home or away", "#b8b8b8", INK),
             ("other", "other", "#e0e0e0", INK)]
    counts = {m: {k: audit["by_arm"][key].get(k, 0) for k, *_ in conds} for m, key in bars}
    totals = {m: sum(v.values()) for m, v in counts.items()}
    want = {"claim store": 170, "ACE-style": 114}
    for m, v in want.items():
        check(f"night-31 entries, {m} total", totals[m], v, tol=0)
    check("night-31 entries, total over the two bars (the text says 284)", sum(totals.values()), 284, tol=0)
    # the segment counts the current figure prints
    for m, k, v in (("claim store", "always", 65), ("claim store", "a time of day", 77),
                    ("claim store", "other", 28), ("ACE-style", "always", 49),
                    ("ACE-style", "a time of day", 46), ("ACE-style", "a person home or away", 12)):
        check(f"night-31 entries, {m}, {k}", counts[m][k], v, tol=0)
    by_condition = {k: sum(counts[m][k] for m, _ in bars) for k, *_ in conds}
    print("  REPORT condition counts: " + "; ".join(
        f"{k}: " + " + ".join(f"{counts[m][k]} {m}" for m, _ in bars) + f" = {by_condition[k]}"
        for k in by_condition))

    fig = plt.figure(figsize=(3.4, 1.9))
    ax = fig.add_axes([0.135, 0.13, 0.85, 0.73])
    xs = list(range(len(bars)))
    ymax = 200
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    fig.canvas.draw()
    pt_per_note = ax.get_window_extent().height * 72 / fig.dpi / ymax
    bottoms = [0] * len(bars)
    printed = {m: {} for m, _ in bars}
    for key, label, colour, ink in conds:
        vals = [counts[m][key] for m, _ in bars]
        ax.bar(xs, vals, bottom=bottoms, color=colour, width=0.5, lw=0.4, edgecolor="white",
               label=label, zorder=3)
        for x, v, b, (m, _) in zip(xs, vals, bottoms, bars):
            # a 7 pt count needs about 7 x 1.2 = 8.4 pt of segment height
            if v and v * pt_per_note >= SMALL * 1.2:
                ax.text(x, b + v / 2, str(v), ha="center", va="center", fontsize=SMALL,
                        color=ink, zorder=4)
                printed[m][key] = v
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for x, (m, _) in zip(xs, bars):
        ax.text(x, totals[m] + 3, str(totals[m]), ha="center", va="bottom", fontsize=FONT)
    ax.set_xticks(xs)
    ax.set_xticklabels([m for m, _ in bars])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_ylabel("Entries (count)")
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=4,
               bbox_to_anchor=(0.52, 1.0), handlelength=0.9, handletextpad=0.35,
               columnspacing=0.75, borderaxespad=0.15, fontsize=SMALL)
    payload = {
        "source": "paper/night31_audit.json (by_arm), from night31_audit.py on overnight_wave/cells, ten households",
        "what": "entries held on night 31 that name an object and a room or spot it occupied on illness "
                "days 14-23, by the condition the entry attaches",
        "bars": {m: {"audit_key": key, "counts": counts[m], "total": totals[m],
                     "printed_inside": printed[m]}
                 for m, key in bars},
        "total_over_both_bars": sum(totals.values()),
        "by_condition_over_both_bars": by_condition,
        "count_printed_if_segment_height_pt_at_least": SMALL * 1.2,
        "note": "The 'ACE-style' bar is the ten-household reduced ACE arm (claim_store_told_if_it_was_right); "
                "the 'claim store' bar is the older-code incremental_edits cell, as in the current figure.",
    }
    # the audit file keys arms by paper name; record the arm directories explicitly
    payload["bars"]["claim store"]["cells"] = "overnight_wave/cells/incremental_edits"
    payload["bars"]["ACE-style"]["cells"] = "overnight_wave/cells/claim_store_told_if_it_was_right"
    save(fig, "figureA2_what_the_notes_held", payload, (3.4, 1.9))
    return payload


# ===================================================================== Figure 5 (figure3)
NIGHT13 = ("Tomas is unwell and staying home. Expect him in bedroom_1 or living room "
           "during day/evening, not office. His laptop/mug/charger likely in bedroom_1 "
           "or living, not office desk.")
NIGHT14 = ("Tomas works from the office during the day (seen 14:08)… The 'unwell' "
           "hypothesis from Day 13 is not supported by Day 14 activity.")
GLASS = "Tomas's glass is on the bedroom nightstand during the day (seen 08:07-15:12)."
DAY14 = ("11 of 11 questions about his moved objects answered in a room the prediction named: "
         "7 in bedroom_1, 4 in the living room.")
TOLD = R / "wave_the_family_reads_the_log/cells/ours_told_and_asked_what_changes/hh_s2_t03"


def _was(claims, claim_id, day):
    c = next(c for c in claims if c["claim_id"] == claim_id)
    rev = next(r for r in c["revision_history"] if r["day"] == day)
    return rev["was"]["statement"]


def figure_5_checks():
    claims = json.load(open(TOLD / "notes.json"))["claims"]
    if _was(claims, "claim_0022", 14) != NIGHT13:
        raise Mismatch("night 13 quote differs from claim_0022 as it stood on night 13")
    n14 = _was(claims, "claim_0022", 15)
    head, tail = NIGHT14.split("… ")
    if not (n14.startswith(head) and n14.endswith(tail)):
        raise Mismatch("night 14 quote differs from claim_0022 as it stood on night 14")
    omitted = n14[len(head):len(n14) - len(tail)].strip()
    if _was(claims, "claim_0024", 15) != GLASS:
        raise Mismatch("glass quote differs from claim_0024 as it stood on night 14")
    print("  ok  the three quotes match notes.json (claim_0022 nights 13 and 14, claim_0024 night 14)")
    movers = sd.the_movers(FrozenHousehold(D.BANKS_10 / "hh_s2_t03.jsonl"))
    tomas = {o for o in movers if o.endswith("_tomas")}
    day14 = [r for r in rows(TOLD.parent.parent, TOLD.parent.name, TOLD.name)
             if r["day"] == 14 and r["object_id"] in tomas]
    true_rooms = collections.Counter(r["true_room"] for r in day14)
    answer_rooms = collections.Counter(r["answer_room"] for r in day14)
    for what, c in (("true room", true_rooms), ("answer room", answer_rooms)):
        if len(day14) != 11 or c != collections.Counter({"bedroom_1": 7, "living": 4}):
            raise Mismatch(f"day 14 {what}s: {dict(c)} of {len(day14)}, expected 7 bedroom_1 + 4 living of 11")
    print("  ok  day 14: 11 questions on Tomas's moved objects, 7 bedroom_1 + 4 living (true and answered room)")
    return {"omitted_by_the_ellipsis": omitted, "day14_true_rooms": dict(true_rooms),
            "day14_answer_rooms": dict(answer_rooms), "day14_objects": sorted(tomas)}


def _wrap_to(fig, text, width_px, size, style="normal"):
    """Greedy wrap at spaces by measured width, so lines fill the box at the real font."""
    renderer = fig.canvas.get_renderer()
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        t = fig.text(0, 0, trial, fontsize=size, style=style)
        wpx = t.get_window_extent(renderer).width
        t.remove()
        if wpx <= width_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return "\n".join(lines)


def figure_5():
    print("\nFigure 5: figure3_told_the_cause.pdf")
    facts = figure_5_checks()
    W, H = 5.5, 1.5
    fig = plt.figure(figsize=(W, H))
    fig.canvas.draw()
    BODY = 8.0
    LS = 1.45
    # neutral: the arm this figure follows is the told variant of a method the paper no longer
    # names, so it must not borrow any method's colour
    accent = "#6f6f6f"
    # three boxes, widths in inches; the third holds two quotes and is widest
    gap, margin = 0.24, 0.02
    widths = [1.58, 1.56]
    widths.append(W - 2 * margin - 2 * gap - sum(widths))
    lefts = [margin, margin + widths[0] + gap, margin + widths[0] + widths[1] + 2 * gap]
    top, bottom = H - 0.02, 0.03
    pad = 0.06
    heads = ["Night 13: told one sentence", "Day 14: every prediction holds",
             "Night 14: the update removes it"]
    px = fig.dpi

    def box(i):
        fig.patches.append(FancyBboxPatch((lefts[i] / W, bottom / H), widths[i] / W,
                                          (top - bottom) / H, boxstyle="round,pad=0,rounding_size=0.012",
                                          transform=fig.transFigure, facecolor="#fbfbfa",
                                          edgecolor=accent, lw=0.9))
        fig.text((lefts[i] + pad) / W, (top - pad + 0.01) / H, heads[i], ha="left", va="top",
                 fontsize=FONT, fontweight="bold")

    for i in range(3):
        box(i)
    body_top = top - pad - 0.17
    inner = lambda i: (widths[i] - 2 * pad) * px
    fig.text((lefts[0] + pad) / W, body_top / H, _wrap_to(fig, NIGHT13, inner(0), BODY, "italic"),
             ha="left", va="top", fontsize=BODY, style="italic", linespacing=LS)
    fig.text((lefts[1] + pad) / W, body_top / H, _wrap_to(fig, DAY14, inner(1), BODY),
             ha="left", va="top", fontsize=BODY, linespacing=LS)
    t1 = fig.text((lefts[2] + pad) / W, body_top / H, _wrap_to(fig, NIGHT14, inner(2), BODY, "italic"),
                  ha="left", va="top", fontsize=BODY, style="italic", linespacing=LS)
    fig.canvas.draw()
    y_after = t1.get_window_extent().y0 / px          # inches from the bottom
    label_y = y_after - 0.045
    fig.text((lefts[2] + pad) / W, label_y / H, "same night", ha="left", va="top",
             fontsize=SMALL, color=MUTED)
    rule_y = label_y - 0.065
    fig.lines.append(plt.Line2D([(lefts[2] + pad + 0.52) / W, (lefts[2] + widths[2] - pad) / W],
                                [(label_y - 0.045) / H] * 2, transform=fig.transFigure,
                                color="#d6d6d6", lw=0.6))
    t2 = fig.text((lefts[2] + pad) / W, (rule_y - 0.05) / H,
                  _wrap_to(fig, GLASS, inner(2), BODY, "italic"),
                  ha="left", va="top", fontsize=BODY, style="italic", linespacing=LS)
    for i in range(2):
        x0 = (lefts[i] + widths[i] + 0.03) / W
        x1 = (lefts[i + 1] - 0.03) / W
        fig.patches.append(matplotlib.patches.FancyArrowPatch(
            (x0, 0.5), (x1, 0.5), transform=fig.transFigure, arrowstyle="-|>,head_width=2.2,head_length=3.6",
            mutation_scale=1.0, color="#4a4a48", lw=1.0))
    fig.canvas.draw()
    # every text item must sit inside its own box, with the padding kept on the right and below
    for t in fig.texts:
        bb = t.get_window_extent()
        x0, x1, y0 = bb.x0 / px, bb.x1 / px, bb.y0 / px
        i = max(j for j in range(3) if x0 >= lefts[j] - 1e-6)
        if x1 > lefts[i] + widths[i] - pad / 2 or y0 < bottom + pad / 2 or bb.y1 / px > top:
            raise Mismatch(f"text runs out of box {i + 1}: {t.get_text()[:40]!r}")
    lowest = min(t.get_window_extent().y0 / px for t in fig.texts)
    print(f"  box text fills {top - lowest:.2f} of {top - bottom:.2f} in; lowest text {lowest - bottom:.2f} in above the box floor")
    payload = {
        "source": str(TOLD),
        "arm": ("the told variant of the log-and-notes memory (cell ours_told_and_asked_what_changes, 24 a day); "
                "the paper no longer includes that method under any name"),
        "boxes": [{"heading": heads[0], "quote": NIGHT13,
                   "from": "notes.json claim_0022, revision of day 14, `was` (the text it held on night 13)"},
                  {"heading": heads[1], "text": DAY14,
                   "from": "searches.jsonl day 14, Tomas's spell-1 movers", **facts}],
        "box3": {"heading": heads[2],
                 "retraction": NIGHT14,
                 "retraction_from": "claim_0022, revision of day 15, `was`; the ellipsis omits one sentence",
                 "label": "same night", "glass": GLASS,
                 "glass_from": "claim_0024, revision of day 15, `was`"},
        "body_size_pt": BODY,
    }
    payload["boxes"].append(payload.pop("box3"))
    del t2
    save(fig, "figure3_told_the_cause", payload, (W, H))
    return payload


def main() -> int:
    apply_style()
    makers = {"figure2": figure_2, "figure3": figure_3, "figure4": figure_4, "figure5": figure_5}
    # `--only figure2 [figure3 ...]` draws just those (brief numbering); the default draws all four
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else list(makers)
    report = {k: makers[k]() for k in only}
    if STALE_LOCKS:
        print(f"\n  {len(STALE_LOCKS)} stale RUNNING.lock files on finished cells were ignored")
    if "figure3" not in report:
        return 0
    d = report["figure3"]["drops"]
    print("\nFigure 3 drops (points below the method's own mean over the 4 preceding days):")
    for n, v in d.items():
        print(f"  {n:18s} day 14 {v['day14']['drop']:5.1f} (2 SE {v['day14']['two_se']:4.1f})   "
              f"day 32 {v['day32']['drop']:5.1f} (2 SE {v['day32']['two_se']:4.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def pdf_text(pdf):
    import subprocess
    out = subprocess.run(["pdftotext", "-raw", str(pdf), "-"], capture_output=True, text=True,
                         check=True).stdout
    return " ".join(out.split())


def check_words(pdf, reject=True):
    """Fail loudly if the PDF's text holds a forbidden word; the PDF is renamed *.REJECTED.pdf."""
    import re
    flat = pdf_text(pdf)
    for pat in FORBIDDEN:
        hit = re.search(pat, flat, re.I)
        if hit:
            if reject:
                pdf.rename(pdf.with_suffix(".REJECTED.pdf"))
            raise Mismatch(f"{pdf.name} contains forbidden text {pat!r} ({hit.group(0)!r}) in: {flat[:200]!r}")
    print(f"  ok  pdftotext: none of {', '.join(FORBIDDEN)} in {pdf.name}")
    return flat


def save(fig, stem, payload, size):
    OUT.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    assert abs(w - size[0]) < 1e-9 and abs(h - size[1]) < 1e-9, (stem, w, h)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    check_words(OUT / f"{stem}.pdf")
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
# CONDITION COLOURS (Oliver, 2026-11-04: the greys were jarring). Okabe-Ito hues that no method
# uses now that the history arms are gone, and a light grey for the catch-all. The three hues pass
# the dataviz validator (light mode, all pairs: worst CVD dE 11.0 deutan, #D55E00 vs #009E73).
COUNT_INK = "#1d2422"


def _lum(h):
    c = [int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def count_ink(fill):
    """The in-segment count colour: white or dark ink, whichever has the higher WCAG contrast.
    (Measured: on #009E73 white 3.42 vs ink 4.62; on #D55E00 white 3.87 vs ink 4.09; on #E69F00
    white 2.25 vs ink 7.02; on #BBBBBB white 1.92 vs ink 8.24. So dark ink wins on all four.)"""
    return max(("#FFFFFF", COUNT_INK), key=lambda ink: contrast(fill, ink))


_FILLS = [("always", "always true", "#009E73"), ("a time of day", "a time of day", "#E69F00"),
          ("a person home or away", "a person home or away", "#D55E00"), ("other", "other", "#BBBBBB")]
CONDITIONS = [(k, label, fill, count_ink(fill)) for k, label, fill in _FILLS]
CONDITION_CONTRAST = {label: {"fill": fill, "count_ink": ink, "contrast": round(contrast(fill, ink), 2),
                              "white_would_be": round(contrast(fill, "#FFFFFF"), 2)}
                      for _, label, fill, ink in CONDITIONS}
def figure_4():
    print("\nFigure 4: figureA2_what_the_notes_held.pdf")
    audit = json.load(open(os.environ.get("NIGHT31_AUDIT", R / "paper/night31_audit.json")))
    # the bars are the cells the current figure draws (the ten-household overnight_wave arms);
    # its "ACE-Style Playbook" bar is the reduced ACE arm, claim_store_told_if_it_was_right
    bars = [("claim store", "claim store"), ("ACE-style", "reduced ACE")]
    conds = CONDITIONS
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
        "condition_colours": CONDITION_CONTRAST,
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
# Oliver, 2026-11-04: "the note named" became "the prediction named" (no "note" in any figure)
DAY14 = ("11 of 11 questions about his moved objects answered in a room the prediction named: "
         "7 in bedroom_1, 4 in the living room.")
DAY14_EXPECTED = ("11 of 11 questions about his moved objects answered in a room the prediction "
                  "named: 7 in bedroom_1, 4 in the living room.")
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
    if DAY14 != DAY14_EXPECTED:
        raise Mismatch(f"middle box text is {DAY14!r}, expected {DAY14_EXPECTED!r}")
    print("  ok  middle box text is the 'prediction named' wording")
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
    if " ".join(DAY14.split()) not in pdf_text(OUT / "figure3_told_the_cause.pdf"):
        raise Mismatch("the new middle-box wording is not in the written PDF")
    print("  ok  the PDF prints the 'prediction named' wording")
    return payload


# ===================================================================== POSTER SET
# The same four figures laid out again at poster size (Oliver, 2026-11-04): figures 2, 3 and 5 at
# 11.5 in wide and figure 4 at 7.1 in, 24 pt body text and nothing below 20 pt, lines about 3 pt.
# Laid out from measured text, not scaled: Figure 2's two panels are stacked, ticks are thinned,
# Figure 5's boxes are sized to their wrapped text. The plotted values are read back from the
# paper set's JSON files, and every value the brief lists is asserted again on what is drawn.
# The whole-run dot values are recomputed from the cells, because the JSON keeps two decimals and
# 76.85 would print as 76.8 where the paper prints 76.9 (76.855).
POSTER = R / "paper/figures/poster"


class Poster:
    def __init__(self, width, width_fig4, font):
        self.W, self.W4, self.F = width, width_fig4, font
        self.k = font / 24.0
        self.S = 20.0 * self.k                 # the smallest text: 20 pt at a 24 pt body
        self.LW = 3.0 * self.k
        self.MS = 8.0 * self.k
        self.DOT = 13.0 * self.k
        self.TICK = 7.0 * self.k
        self.THIN = 1.3 * self.k               # spines, ticks, brackets, box outlines

    def describe(self):
        return {"body_pt": self.F, "smallest_pt": self.S, "line_pt": self.LW, "marker_pt": self.MS,
                "width_in_figs_2_3_5": self.W, "width_in_fig_4": self.W4}


def apply_poster_style(P):
    apply_style()
    plt.rcParams.update({
        "font.size": P.F, "axes.labelsize": P.F, "axes.titlesize": P.F,
        "xtick.labelsize": P.F, "ytick.labelsize": P.F, "legend.fontsize": P.F,
        "axes.linewidth": P.THIN, "xtick.major.width": P.THIN, "ytick.major.width": P.THIN,
        "xtick.major.size": P.TICK, "ytick.major.size": P.TICK,
        "xtick.major.pad": 4 * P.k, "ytick.major.pad": 4 * P.k, "axes.labelpad": 6 * P.k,
    })


_RULER = {}


def measure(text, size, weight="normal", style="normal", linespacing=1.2):
    """(width, height) in inches of `text` as drawn, from a scratch canvas."""
    if "fig" not in _RULER:
        _RULER["fig"] = plt.figure(figsize=(4, 4))
    fig = _RULER["fig"]
    t = fig.text(0, 0, text, fontsize=size, fontweight=weight, style=style, linespacing=linespacing)
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def wrap_measured(text, width_in, size, weight="normal", style="normal"):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if measure(trial, size, weight, style)[0] <= width_in or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return "\n".join(lines)


def inch_axes(fig, x0, y0, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


def layout_check(fig, name, min_size):
    """Nothing clipped at the canvas edge, no two text items overlapping, no text below min_size."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.width == 0 or bb.height == 0:
            continue
        if t.get_fontsize() < min_size - 1e-6:
            raise Mismatch(f"{name}: {t.get_text()[:30]!r} is {t.get_fontsize():.1f} pt, below {min_size}")
        items.append((t.get_text(), bb))
    fb = fig.bbox
    for text, bb in items:
        if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5:
            raise Mismatch(f"{name}: {text[:30]!r} is clipped by the canvas edge")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if (min(a.x1, b.x1) - max(a.x0, b.x0) > 1.0 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1.0):
                raise Mismatch(f"{name}: {items[i][0][:30]!r} overlaps {items[j][0][:30]!r}")
    print(f"  ok  {name}: {len(items)} text items, none clipped, none overlapping, none below {min_size:g} pt")


def poster_save(fig, stem, payload, P, min_size):
    layout_check(fig, stem, min_size)
    POSTER.mkdir(parents=True, exist_ok=True)
    (POSTER / "png_150dpi").mkdir(exist_ok=True)
    pdf = POSTER / f"{stem}_poster.pdf"
    w, h = fig.get_size_inches()
    fig.savefig(pdf)
    plt.close(fig)
    check_words(pdf)
    import subprocess
    subprocess.run(["pdftoppm", "-r", "150", "-png", "-singlefile", str(pdf),
                    str(POSTER / "png_150dpi" / f"{stem}_poster")], check=True)
    payload = {"figure_file": pdf.name, "size_in": [round(w, 3), round(h, 3)], "poster": P.describe(),
               **payload}
    (POSTER / f"{stem}_poster.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"  wrote {pdf} ({w:.2f} x {h:.2f} in), .json and png_150dpi/{stem}_poster.png")


def paper_json(stem):
    path = OUT / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: draw the paper set first")
    return json.load(open(path))


def poster_line(ax, days, values, name, P):
    xs = [d for d, v in zip(days, values) if v is not None]
    ys = [v for v in values if v is not None]
    face = "white" if name == "last-seen, updated nightly" else COLOR[name]
    edge = "#6a9fc4" if name == "claim store + history" else COLOR[name]
    # dash patterns are in units of the line width, so they scale with it
    ax.plot(xs, ys, color=COLOR[name], linestyle=DASH[name], lw=P.LW, marker=MARK[name], ms=P.MS,
            markerfacecolor=face, markeredgecolor=edge, markeredgewidth=1.0 * P.k, zorder=4,
            dash_capstyle="butt")


def poster_handle(name, P):
    face = "white" if name == "last-seen, updated nightly" else COLOR[name]
    edge = "#6a9fc4" if name == "claim store + history" else COLOR[name]
    return plt.Line2D([], [], color=COLOR[name], linestyle=DASH[name], lw=P.LW, marker=MARK[name],
                      ms=P.MS, markerfacecolor=face, markeredgecolor=edge,
                      markeredgewidth=1.0 * P.k, label=name)


POSTER_HEADROOM = 116


def poster_ill(ax, spans, P):
    for first, last in spans:
        ax.axvspan(first - 0.5, last + 0.5, color=SHADE, lw=0, zorder=0)
        ax.text((first + last) / 2, POSTER_HEADROOM - 1.5, "ill", ha="center", va="top",
                fontsize=P.S, color=MUTED, zorder=1)
    ax.set_ylim(0, POSTER_HEADROOM)
    ax.spines["left"].set_bounds(0, 100)


def poster_figure_2(P):
    print("\nPoster Figure 2: figureA1_first_illness_poster.pdf")
    J = paper_json("figureA1_first_illness")
    homes = D.HOMES_10
    rows_named = [r for r in J["right_panel"]["rows_top_to_bottom"]]
    order = [None if r is None else r["method"] for r in rows_named]
    if order != ["claim store + history", "claim store + today's sightings", "claim store", None,
                 "last-seen", "last-seen, updated nightly"]:
        raise Mismatch(f"row order in the paper JSON is {order}")
    whole = {}
    for n in [o for o in order if o]:
        c, a = ARMS_10[n]
        whole[n] = pct([r for h in homes for r in rows(c, a, h)])
        if abs(round(whole[n], 2) - rows_named[order.index(n)]["value"]) > 0.006:
            raise Mismatch(f"{n}: recomputed {whole[n]} differs from the paper JSON")
    for n, v in {"claim store + history": 84.1, "claim store + today's sightings": 79.9,
                 "claim store": 75.8, "last-seen": 82.1}.items():
        check(f"whole-run first room right, {n}", whole[n], v)
    st = J["right_panel"]["steps"]
    s1 = st["claim store -> claim store + today's sightings"]
    s2 = st["claim store + today's sightings -> claim store + history"]
    for label, got, want in (("step 1 mean", s1["mean"], 4.1), ("step 1 2 SE", s1["two_se"], 2.8),
                             ("step 2 mean", s2["mean"], 4.2), ("step 2 2 SE", s2["two_se"], 3.0)):
        check(label, got, want)
    left = J["left_panel"]
    left_names = ["last-seen", "claim store", "claim store + history"]
    if list(left["lines"]) != left_names:
        raise Mismatch(f"left lines in the paper JSON are {list(left['lines'])}")

    F, S, W = P.F, P.S, P.W
    pad = 0.12 * P.k
    # --- vertical budget, in inches from the top
    leg_h = measure("Ag", F)[1] * 1.6
    ylab = "first room right,\nmoved objects (%)"
    ylab_w = measure(ylab, F)[1]                       # rotated: its height is its width on the page
    ytick_w = measure("100", F)[0]
    xaxis_h = measure("31", F)[1] + measure("day", F)[1] + P.TICK / 72 + 0.22 * P.k
    top_h = 3.6 * P.k
    row_pitch = measure("Ag", F)[1] * 1.75
    dot_h = row_pitch * len(order)
    gap = 0.45 * P.k
    H = pad + leg_h + top_h + xaxis_h + gap + dot_h + xaxis_h + pad
    fig = plt.figure(figsize=(W, H))

    # --- top: by day on the moved objects
    lm = pad + ylab_w + ytick_w + 0.25 * P.k
    axT = inch_axes(fig, lm, H - pad - leg_h - top_h, W - lm - 0.3 * P.k, top_h)
    poster_ill(axT, [(14, 23)], P)
    days = left["days"]
    for n in left_names:
        poster_line(axT, days, left["lines"][n]["values"], n, P)
    axT.set_xlim(0.5, 31.5)
    axT.set_yticks([0, 50, 100])
    axT.set_xticks([1, 14, 24, 31])
    axT.set_xlabel("day")
    axT.set_ylabel(ylab)
    fig.legend(handles=[poster_handle(n, P) for n in left_names], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5 + (lm - 0.3 * P.k) / W / 2, 1 - pad / H), handlelength=2.4,
               handletextpad=0.5, columnspacing=1.6, borderaxespad=0.0, fontsize=F)

    # --- bottom: the dot plot, its own width budget from measured text
    names = [o for o in order if o]
    label_w = max(measure(n, F)[0] for n in names)
    value_w = max(measure(f"{whole[n]:.1f}", F)[0] for n in names)
    step_txt = [f"+{s1['mean']:.1f} ± {s1['two_se']:.1f}", f"+{s2['mean']:.1f} ± {s2['two_se']:.1f}"]
    step_w = max(measure(t, S)[0] for t in step_txt)
    x0 = pad + label_w + 0.2 * P.k
    off = 0.12 * P.k                                   # dot to its value label
    bgap, btick, tgap = 0.14 * P.k, 0.16 * P.k, 0.1 * P.k
    right_need = off + value_w + bgap + tgap + step_w + pad + 0.05
    frac = (max(whole[n] for n in names[:3]) - 70) / 20
    A = min((W - x0 - right_need) / frac, W - x0 - 0.3 * P.k)
    axB = inch_axes(fig, x0, pad + xaxis_h, A, dot_h)
    ypos = {n: i for i, n in enumerate(order) if n}
    labels = {}
    for n, y in ypos.items():
        axB.plot([70, whole[n]], [y, y], color="#e2e2e2", lw=P.THIN, zorder=1)
        face = "white" if n == "last-seen, updated nightly" else COLOR[n]
        axB.plot([whole[n]], [y], marker=MARK[n], ms=P.DOT, color=COLOR[n], markerfacecolor=face,
                 markeredgecolor=COLOR[n] if n != "claim store + history" else "#6a9fc4",
                 markeredgewidth=1.6 * P.k, zorder=4, linestyle="none", clip_on=False)
        labels[n] = axB.annotate(f"{whole[n]:.1f}", (whole[n], y), xytext=(off * 72 + P.DOT / 2, 0),
                                 textcoords="offset points", ha="left", va="center", fontsize=F,
                                 annotation_clip=False)
    axB.set_ylim(len(order) - 0.5, -0.5)
    axB.set_yticks([ypos[n] for n in ypos])
    axB.set_yticklabels(names)
    axB.tick_params(axis="y", length=0)
    axB.set_xlim(70, 90)
    axB.set_xticks([70, 80, 90])
    axB.set_xticks([75, 85], minor=True)
    axB.tick_params(axis="x", which="minor", length=P.TICK * 0.6, width=P.THIN)
    axB.set_xlabel("first room right, whole run (%)")
    axB.spines["left"].set_visible(False)
    fig.canvas.draw()
    inv = axB.transData.inverted()
    right_px = max(labels[n].get_window_extent().x1 for n in names[:3])
    bx = inv.transform((right_px + bgap * fig.dpi, 0))[0]
    tick = bx - inv.transform((right_px + (bgap - btick) * fig.dpi, 0))[0]
    g = 0.08

    def bracket(y_a, y_b, text):
        top, bot = min(y_a, y_b), max(y_a, y_b)
        axB.plot([bx - tick, bx, bx, bx - tick], [top + g, top + g, bot - g, bot - g], color=MUTED,
                 lw=P.THIN, solid_capstyle="butt", zorder=3, clip_on=False)
        axB.annotate(text, (bx, (top + bot) / 2), xytext=(tgap * 72, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=S, annotation_clip=False)
    bracket(ypos["claim store"], ypos["claim store + today's sightings"], step_txt[0])
    bracket(ypos["claim store + today's sightings"], ypos["claim store + history"], step_txt[1])

    payload = {"layout": "the paper's two panels stacked: by-day lines above, the dot plot below",
               "values_from": str(OUT / "figureA1_first_illness.json"),
               "left_panel": left,
               "right_panel": {"rows_top_to_bottom": [None if n is None else
                                                      {"method": n, "value": round(whole[n], 3),
                                                       "printed": f"{whole[n]:.1f}"} for n in order],
                               "steps_printed": step_txt}}
    poster_save(fig, "figureA1_first_illness", payload, P, S)


def poster_figure_3(P):
    print("\nPoster Figure 3: figure2_both_illnesses_poster.pdf")
    J = paper_json("figure2_both_illnesses")
    order = ["last-seen", "claim store", "ACE-style"]
    if list(J["lines"]) != order:
        raise Mismatch(f"lines in the paper JSON are {list(J['lines'])}")
    # the drops, recomputed from the drawn curves against the paper JSON
    days = J["days"]
    for n in order:
        line = J["lines"][n]["values"]
        for onset, before in ((14, range(10, 14)), (32, range(28, 32))):
            base = statistics.mean(line[days.index(d)] for d in before)
            drop = base - line[days.index(onset)]
            if abs(drop - J["drops"][n][f"day{onset}"]["drop"]) > 0.02:
                raise Mismatch(f"{n} day {onset}: drop {drop:.2f} on the drawn curve, JSON "
                               f"{J['drops'][n][f'day{onset}']['drop']}")
        print(f"  ok  {n:12s} drops on the drawn curve: day 14 {J['drops'][n]['day14']['drop']:.1f}, "
              f"day 32 {J['drops'][n]['day32']['drop']:.1f}")
    F, S, W = P.F, P.S, P.W
    pad = 0.12 * P.k
    leg_h = measure("Ag", F)[1] * 1.6
    ylab = "first room right (%)"
    lm = pad + measure(ylab, F)[1] + measure("100", F)[0] + 0.25 * P.k
    xaxis_h = measure("49", F)[1] + measure("day", F)[1] + P.TICK / 72 + 0.22 * P.k
    ax_h = 3.4 * P.k
    H = pad + leg_h + ax_h + xaxis_h + pad
    fig = plt.figure(figsize=(W, H))
    ax = inch_axes(fig, lm, pad + xaxis_h, W - lm - 0.3 * P.k, ax_h)
    poster_ill(ax, [(14, 23), (32, 41)], P)
    for n in order:
        poster_line(ax, days, J["lines"][n]["values"], n, P)
    ax.set_xlim(0.5, 49.5)
    ax.set_xticks([1, 14, 24, 32, 42, 49])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("day")
    ax.set_ylabel(ylab)
    fig.legend(handles=[poster_handle(n, P) for n in order], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5 + (lm - 0.3 * P.k) / W / 2, 1 - pad / H), handlelength=2.4,
               handletextpad=0.5, columnspacing=1.8, borderaxespad=0.0, fontsize=F)
    payload = {"values_from": str(OUT / "figure2_both_illnesses.json"),
               "days": days, "lines": J["lines"], "drops": J["drops"]}
    poster_save(fig, "figure2_both_illnesses", payload, P, S)


def poster_figure_4(P):
    print("\nPoster Figure 4: figureA2_what_the_notes_held_poster.pdf")
    J = paper_json("figureA2_what_the_notes_held")
    bars = list(J["bars"])
    if bars != ["claim store", "ACE-style"]:
        raise Mismatch(f"bars in the paper JSON are {bars}")
    counts = {m: J["bars"][m]["counts"] for m in bars}
    totals = {m: sum(counts[m].values()) for m in bars}
    check("claim store total", totals["claim store"], 170, tol=0)
    check("ACE-style total", totals["ACE-style"], 114, tol=0)
    check("total over the two bars", sum(totals.values()), 284, tol=0)
    for m, k, v in (("claim store", "always", 65), ("claim store", "a time of day", 77),
                    ("claim store", "other", 28), ("ACE-style", "always", 49),
                    ("ACE-style", "a time of day", 46), ("ACE-style", "a person home or away", 12)):
        check(f"{m}, {k}", counts[m][k], v, tol=0)
    conds = CONDITIONS
    F, S, W = P.F, P.S, P.W4
    pad = 0.12 * P.k
    leg_h = measure("Ag", S)[1] * 1.5 * 2 + 0.1 * P.k          # two rows
    ylab = "Entries (count)"
    lm = pad + measure(ylab, F)[1] + measure("200", F)[0] + 0.25 * P.k
    xaxis_h = measure("ACE-style", F)[1] + 0.3 * P.k
    ax_h = 4.2 * P.k
    H = pad + leg_h + 0.15 * P.k + ax_h + xaxis_h + pad
    fig = plt.figure(figsize=(W, H))
    ax = inch_axes(fig, lm, pad + xaxis_h, W - lm - 0.2 * P.k, ax_h)
    ymax = 200
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    pt_per = ax_h * 72 / ymax
    xs = list(range(len(bars)))
    bottoms = [0] * len(bars)
    printed = {m: {} for m in bars}
    for key, label, colour, ink in conds:
        vals = [counts[m][key] for m in bars]
        ax.bar(xs, vals, bottom=bottoms, color=colour, width=0.5, lw=1.0 * P.k, edgecolor="white",
               label=label, zorder=3)
        for x, v, b, m in zip(xs, vals, bottoms, bars):
            if v and v * pt_per >= S * 1.25:
                ax.text(x, b + v / 2, str(v), ha="center", va="center", fontsize=S, color=ink, zorder=4)
                printed[m][key] = v
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for x, m in zip(xs, bars):
        ax.annotate(str(totals[m]), (x, totals[m]), xytext=(0, 5 * P.k), textcoords="offset points",
                    ha="center", va="bottom", fontsize=F)
    ax.set_xticks(xs)
    ax.set_xticklabels(bars)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([0, 100, 200])
    ax.set_ylabel(ylab)
    ax.yaxis.grid(True, color="#e6e6e6", lw=P.THIN, zorder=0)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5 + (lm - 0.2 * P.k) / W / 2, 1 - pad / H), handlelength=1.0,
               handletextpad=0.4, columnspacing=1.2, labelspacing=0.3, borderaxespad=0.0, fontsize=S)
    payload = {"values_from": str(OUT / "figureA2_what_the_notes_held.json"),
               "bars": {m: {"counts": counts[m], "total": totals[m], "printed_inside": printed[m]}
                        for m in bars},
               "total_over_both_bars": sum(totals.values()),
               "count_printed_if_segment_height_pt_at_least": S * 1.25,
               "condition_colours": CONDITION_CONTRAST}
    poster_save(fig, "figureA2_what_the_notes_held", payload, P, S)


def poster_figure_5(P):
    print("\nPoster Figure 5: figure3_told_the_cause_poster.pdf")
    J = paper_json("figure3_told_the_cause")
    heads = [b["heading"] for b in J["boxes"]]
    q13, day14, box3 = J["boxes"][0]["quote"], J["boxes"][1]["text"], J["boxes"][2]
    for label, got, want in (("night 13 quote", q13, NIGHT13), ("day 14 text", day14, DAY14_EXPECTED),
                             ("retraction", box3["retraction"], NIGHT14), ("glass", box3["glass"], GLASS)):
        if got != want:
            raise Mismatch(f"poster {label} differs from the paper figure's")
    print("  ok  the three quotes and the middle text are the paper figure's, character for character")
    F, S, W = P.F, P.S, P.W
    LS = 1.3
    margin, gap, pad = 0.04, 0.5 * P.k, 0.16 * P.k
    avail = W - 2 * margin - 2 * gap

    def content(i, w):
        """Wrapped pieces of box i at box width w, and the height they need (inches)."""
        inner = w - 2 * pad
        # each heading on two lines, broken at its colon ("Night 14:" / "the update removes it");
        # a width that cannot hold either half on one line is ruled out (no orphaned words)
        first, rest = heads[i].split(": ", 1)
        if max(measure(first + ":", F, "bold")[0], measure(rest, F, "bold")[0]) > inner:
            return None, None, None, float("inf")
        head = first + ":\n" + rest
        parts = [("head", head, F, "bold", "normal")]
        if i == 0:
            parts.append(("body", wrap_measured(q13, inner, F, style="italic"), F, "normal", "italic"))
        elif i == 1:
            parts.append(("body", wrap_measured(day14, inner, F), F, "normal", "normal"))
        else:
            parts.append(("body", wrap_measured(box3["retraction"], inner, F, style="italic"), F, "normal", "italic"))
            parts.append(("label", box3["label"], S, "normal", "normal"))
            parts.append(("body", wrap_measured(box3["glass"], inner, F, style="italic"), F, "normal", "italic"))
        hs = [measure(t, sz, wt, st, LS)[1] for _, t, sz, wt, st in parts]
        sep = [0.12 * P.k] + [0.1 * P.k] * (len(parts) - 2)
        return parts, hs, sep, sum(hs) + sum(sep) + 2 * pad

    # box widths that make the tallest box as short as possible (a 0.1 in grid)
    best = None
    step = 0.1
    w1 = 2.0
    while w1 < avail - 4.0:
        w2 = 2.0
        while w1 + w2 < avail - 2.0:
            w3 = avail - w1 - w2
            hmax = max(content(0, w1)[3], content(1, w2)[3], content(2, w3)[3])
            if hmax < float("inf") and (best is None or hmax < best[0] - 1e-9):
                best = (hmax, [w1, w2, w3])
            w2 += step
        w1 += step
    if best is None:
        raise Mismatch("poster figure 5: no box widths fit the headings on two lines")
    box_h, widths = best
    H = box_h + 2 * margin
    fig = plt.figure(figsize=(W, H))
    T = fig.dpi_scale_trans
    lefts = [margin, margin + widths[0] + gap, margin + widths[0] + widths[1] + 2 * gap]
    accent = "#6f6f6f"
    placed = []
    for i in range(3):
        fig.patches.append(FancyBboxPatch((lefts[i], margin), widths[i], box_h, transform=T,
                                          boxstyle=f"round,pad=0,rounding_size={0.05 * P.k}",
                                          facecolor="#fbfbfa", edgecolor=accent, lw=1.8 * P.k))
        parts, hs, sep, _ = content(i, widths[i])
        y = H - margin - pad
        for j, ((kind, text, sz, wt, st), h) in enumerate(zip(parts, hs)):
            colour = MUTED if kind == "label" else INK
            fig.text(lefts[i] + pad, y, text, transform=T, ha="left", va="top", fontsize=sz,
                     fontweight=wt, style=st, color=colour, linespacing=LS)
            if kind == "label":
                lw_ = measure(text, sz)[0]
                fig.lines.append(plt.Line2D([lefts[i] + pad + lw_ + 0.12 * P.k, lefts[i] + widths[i] - pad],
                                            [y - h / 2] * 2, transform=T, color="#d6d6d6", lw=P.THIN))
            y -= h + (sep[j] if j < len(sep) else 0)
        placed.append(i)
    for i in range(2):
        fig.patches.append(matplotlib.patches.FancyArrowPatch(
            (lefts[i] + widths[i] + 0.07 * P.k, margin + box_h / 2),
            (lefts[i + 1] - 0.07 * P.k, margin + box_h / 2), transform=T,
            arrowstyle=f"-|>,head_width={9 * P.k},head_length={14 * P.k}", mutation_scale=1.0,
            color="#4a4a48", lw=3.0 * P.k))
    fig.canvas.draw()
    px = fig.dpi
    for t in fig.texts:
        bb = t.get_window_extent()
        x0, x1, y0, y1 = bb.x0 / px, bb.x1 / px, bb.y0 / px, bb.y1 / px
        i = max(j for j in range(3) if x0 >= lefts[j] - 1e-6)
        if x1 > lefts[i] + widths[i] - pad / 2 or y0 < margin + pad / 2 or y1 > margin + box_h:
            raise Mismatch(f"poster figure 5: text runs out of box {i + 1}: {t.get_text()[:40]!r}")
    payload = {"values_from": str(OUT / "figure3_told_the_cause.json"),
               "box_widths_in": [round(w, 2) for w in widths], "boxes": J["boxes"]}
    poster_save(fig, "figure3_told_the_cause", payload, P, S)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", choices=["figure2", "figure3", "figure4", "figure5"],
                    help="draw only these (brief numbering); default all four")
    ap.add_argument("--poster", action="store_true",
                    help="draw the poster set into figures/poster/ (from the paper set's JSON)")
    ap.add_argument("--poster-only", action="store_true", help="the poster set only, not the paper set")
    ap.add_argument("--poster-width", type=float, default=11.5, help="poster width of figures 2, 3, 5 (in)")
    ap.add_argument("--poster-width-fig4", type=float, default=7.1, help="poster width of figure 4 (in)")
    ap.add_argument("--poster-font", type=float, default=24.0,
                    help="poster body text (pt); the smallest text is 20/24 of it, lines 3/24 of it")
    ap.add_argument("--check-words", action="store_true",
                    help="only run the forbidden-word check on every PDF in camera_ready/ and poster/")
    args = ap.parse_args()
    if args.check_words:
        pdfs = sorted(OUT.glob("*.pdf")) + sorted(POSTER.glob("*.pdf"))
        for pdf in pdfs:
            check_words(pdf, reject=False)
        print(f"  {len(pdfs)} PDFs checked")
        return 0
    only = args.only or ["figure2", "figure3", "figure4", "figure5"]
    report = {}
    if not args.poster_only:
        apply_style()
        makers = {"figure2": figure_2, "figure3": figure_3, "figure4": figure_4, "figure5": figure_5}
        report = {k: makers[k]() for k in only}
        if STALE_LOCKS:
            print(f"\n  {len(STALE_LOCKS)} stale RUNNING.lock files on finished cells were ignored")
    if args.poster or args.poster_only:
        P = Poster(args.poster_width, args.poster_width_fig4, args.poster_font)
        apply_poster_style(P)
        posters = {"figure2": poster_figure_2, "figure3": poster_figure_3,
                   "figure4": poster_figure_4, "figure5": poster_figure_5}
        for k in only:
            posters[k](P)
    if "figure3" in report:
        d = report["figure3"]["drops"]
        print("\nFigure 3 drops (points below the method's own mean over the 4 preceding days):")
        for n, v in d.items():
            print(f"  {n:18s} day 14 {v['day14']['drop']:5.1f} (2 SE {v['day14']['two_se']:4.1f})   "
                  f"day 32 {v['day32']['drop']:5.1f} (2 SE {v['day32']['two_se']:4.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

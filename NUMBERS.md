# Where each number comes from

Every command runs from the repository root on the included data, CPU only, with `PYTHONPATH=src`.
`./reproduce.sh` runs all of them and compares each output with the paper's copy in
`results/self_improve/paper/` (the "expected" column). Cell paths below are relative to `results/self_improve/`.

Measures. **First room right**: the first room the robot opened held the object (`found_at_step == 1` in
`searches.jsonl`, the same as `rooms_opened[0] == true_room`). **Found within 3**: the object was found within
the 3-room budget (`found_it`). "±" is 2 standard errors of the per-household paired difference
(2 · sd / √households). Qwen is Qwen3.8-27B; Sonnet is claude-sonnet-5-5.

**NOT IN THE CAMERA-READY PAPER** marks numbers for history only (`log_only_no_notes`), history and notes
(`the_log_and_notes_about_the_routine`), notes hidden (`prior_only_no_notes`) and small working memory (`a_small_working_memory_and_an_archive`). The camera-ready paper's
methods are last-seen, last-seen updated nightly, claim store, claim store + today's sightings, claim store +
history and ACE-style. Those three arms' cells and scripts are kept because they are still cited during the
revision and several checks pair against them.

## Table 1: ten homes, 8 questions a day, 31 days (`overnight_wave/cells/`)

Command: `python3 analysis/table1_ten_homes.py`. Expected: `paper/table1_ten_homes.txt`.

| method | cell dir | first room right | found within 3 |
|---|---|---|---|
| last-seen | `last_seen_no_model` | 82.1 | 95.0 |
| claim store (older code, see README caveat) | `incremental_edits` | 77.0 | 86.9 |
| reduced ACE (ACE-style) | `claim_store_told_if_it_was_right` | 76.0 | 85.3 |
| small working memory: NOT IN THE CAMERA-READY PAPER | `a_small_working_memory_and_an_archive` | 71.0 | 82.1 |
| log and notes: NOT IN THE CAMERA-READY PAPER | `the_log_and_notes_about_the_routine` | 85.6 | 96.6 |
| notes hidden: NOT IN THE CAMERA-READY PAPER | `prior_only_no_notes` | 53.5 | 65.8 |

Paired differences, same file: last-seen − claim store +5.1 ± 5.7 / +8.2 ± 4.3 (7 / 10 of 10 homes);
last-seen − reduced ACE +6.0 ± 6.1 / +9.7 ± 6.1; last-seen − small working memory +11.1 ± 7.1 / +12.9 ± 6.9 (NOT IN THE CAMERA-READY PAPER);
log and notes − last-seen +3.5 ± 4.2 / +1.5 ± 2.0 (NOT IN THE CAMERA-READY PAPER).

## E0. The sighting history

Constants, not computed: `src/self_improve/the_log_the_robot_reads.py`, `what_you_have_seen_of`:
`HOW_MANY_SIGHTINGS_TO_SHOW = 40`, `HOW_MANY_THINGS_BESIDE_IT = 6`. The real rendered prompts are in
`paper/prompts_appendix.md` (needs the response cache to regenerate; see README).

How often the 40-sighting window binds (NOT IN THE CAMERA-READY PAPER as computed: it is measured on the history-and-notes
arm's record): `python3 analysis/session_length_and_window_share.py`, expected `paper/session_length_and_window_share.txt`.
8 a day: the window cuts on 1,224 of 2,480 questions (49.4%); 24 a day (31 days, `overnight_wave_24_questions`): 78.3%.

## E1. Last-seen updated each night (retention vs same-day access)

Command: `python3 analysis/camera_ready_report.py`. Expected: `paper/camera_ready_report.txt`
(the original `REPORT_numbers.txt`). First room right, `last_seen_updated_each_night` minus:

| setting | vs last-seen | vs claim store | report line |
|---|---|---|---|
| ten homes, 8 a day (`overnight_wave`) | −5.2 ± 4.0 | −0.2 ± 6.5 | 23, 32 |
| three 50-day homes, 24 a day, days 1-49 (`wave_the_second_illness`) | −8.6 ± 3.2 | +1.0 ± 6.6 | 207, 220 |
| five homes, 4 / 8 / 24 a day (`wave_the_budget_sweep_q4`, `_q8`, `wave_wider_five`) | −2.7 / −2.1 / −8.1 | (not run) | 339, 404, 469 |

Absolute level on the ten homes: 76.9% (1,906 of 2,480), same report, "days 1-31" column of the
`last_seen_updated_each_night` row.

## E2. Claim store + history (`camera_ready/qwen_ten_homes_8_a_day/cells/`)

Command: `python3 analysis/item2_first_room_right.py`. Expected: `paper/item2_first_room_right_qwen.txt`.
Claim store (current-code rerun, `incremental_edits`) 75.8 → claim store + history (`claim_store_plus_the_log`) 84.1,
+8.3 ± 4.6 (9 of 10). By period: settled +5.8 ± 4.1, disrupted +12.6 ± 8.1, back to normal +6.9 ± 4.4.
The comparison with history only (85.2) and history and notes (85.6) is NOT IN THE CAMERA-READY PAPER.

## E2b. Claim store + today's sightings

Commands, expected `paper/item2b_today_sightings_qwen.txt` (the two outputs, in this order, separated by a blank line):

    python3 analysis/item2_first_room_right.py "claim_store_plus_today's_sightings"
    python3 analysis/item2_first_room_right.py claim_store_plus_the_log "claim_store_plus_today's_sightings"

(1) plus today's − claim store +4.1 ± 2.8 (7 of 10); settled +2.3 ± 3.2, disrupted +6.9 ± 6.2, back to normal +3.4 ± 4.4.
(2) plus history − plus today's +4.2 ± 3.0 (7 of 10); settled +3.5 ± 3.0, disrupted +5.7 ± 4.7, back to normal +3.4 ± 2.7.
Pooled 75.8 / 79.9 / 84.1 over 2,480 questions each; per home in the same file. "About half": 4.1 of 8.3 = 49%.
The hh_s63 d3q22 example: `camera_ready/qwen_ten_homes_8_a_day/cells/claim_store_plus_today's_sightings/hh_s63_t03/`,
`searches.jsonl` (question d3q22) and `notes.json` (claim_0011).

## E3 and E3b. History only: NOT IN THE CAMERA-READY PAPER

- Ten homes: `python3 analysis/log_only_row.py`, expected `paper/log_only_row.txt`. 85.2 first room right;
  −0.4 ± 1.9 vs history and notes, +8.2 ± 3.9 vs claim store (9 of 10), +3.1 ± 3.4 vs last-seen; found within 3 95.9,
  +9.0 ± 3.7 vs claim store.
- Five homes, 4 a day: `python3 analysis/history_only_q4.py`, expected `paper/history_only_q4.txt`.
  +1.6 ± 2.9 vs history and notes, +6.0 ± 2.1 vs last-seen, +8.7 ± 2.9 vs last-seen updated each night.
- Three 50-day homes: `python3 analysis/history_only_50day.py`, expected `paper/history_only_50day.txt`.
  +0.1 ± 1.0 vs history and notes, +14.0 ± 4.3 vs claim store, +10.0 ± 7.0 vs ACE as published,
  +13.0 ± 3.2 vs last-seen updated each night, +4.3 ± 4.3 vs last-seen.
- Budget sweep from the earlier draft (log and notes − last-seen, five homes, first room right / found within 3):
  4 a day +4.4 / +5.0 from `python3 analysis/budget_sweep.py`, expected `paper/budget_sweep.txt` (written for this
  release; it matches the draft). The draft's 8 a day (+3.5 / +2.6) and 24 a day (+1.2 / +0.1) values need the
  history-and-notes cells of `wave_the_budget_sweep_q8` and `wave_wider_five`, which were left out for size.

## E4. Prompts and household setup

- Prompts appendix: `paper/prompts_appendix.md`. Regenerating it replays every arm's hh_s2_t03 cell to day 14
  from the response cache (`analysis/replay/prompts_appendix.py`), which is not in this release.
- Household shapes: `python3 analysis/what_a_household_is_made_of.py`, expected `paper/what_a_household_is_made_of.txt`.
  Ten homes: 2-3 residents, 8-9 rooms, 39-46 places, 76-104 objects (52-73 ever move), 5-9 movers, 744 questions
  over 31 days in each bank (743 in hh_s93).
- The new arms' prompts differ from the claim store's only in the added block:
  `python3 analysis/replay/prompt_diff.py "claim store plus the log" out/diff.md` (no model; expected
  `paper/prompt_diffs/`).

## E5. Sonnet one-household pilot (`camera_ready/sonnet_pilot_v2/`, hh_s2_t03, 8 a day, 31 days)

Command: `python3 analysis/sonnet_pilot_summary.py`, expected `paper/sonnet_pilot_summary.txt` (written for this
release). First room right: ACE as published 0.859, claim store 0.786; history only 0.867 and history and notes
0.823 are NOT IN THE CAMERA-READY PAPER. Room choices still refused after the retry (fell back to the fixed room
order): history and notes 14 of 308, history only 7 of 293, claim store 1 of 321, ACE 1 of 303.
Calls, rewordings and cost: `python3 analysis/report_sonnet_v2.py`, expected `camera_ready/sonnet_pilot_v2/REPORT.txt`
($2.72 + $2.39 + $9.68 + $5.43 = $20.22 for the four finished arms; the paper's $22.85 also counts a stopped ACE attempt
(about $2.62; its files are not included).

## E6. Why summaries fail

- Claims citing contradicting sightings, and duplicate claims: `python3 analysis/claims_contradicting_and_duplicates.py`,
  expected `paper/claims_contradicting_and_duplicates.txt`. Qwen, 20 runs (10 homes × `incremental_edits`,
  `claim_store_plus_the_log`): 913 claims (459 + 454), 0 citing a contradicting sighting; 811 standing (401 + 410),
  116 extra copies (46 + 70). Sonnet: 7 of 30 (claim store), 4 of 45 (ACE), and 5 of 59 (history and notes, NOT IN THE
  CAMERA-READY PAPER). Table 1's older-code hh_s2 cell: 0 of 50, 8 copies. `--all` adds the
  claim store + today's sightings arm (0 of 471), which the paper does not count.
- The mug example: `camera_ready/qwen_ten_homes_8_a_day/cells/incremental_edits/hh_s2_t03/notes.json`, claim_0028;
  searches d14q13, d16q22, d20q19 in the same cell.
- Category-prior misses (E6 addition): `python3 analysis/category_prior_misses.py`, expected
  `paper/category_prior/category_prior_misses.txt`. Strict count: claim store 56 of 599 (9.3%), plus today's 23 of 498
  (4.6%), plus history 0.5%; history only 1.1% and history and notes 0 are NOT IN THE CAMERA-READY PAPER.
- Readers misjudge absences (40 hand-coded wrong first choices): NOT IN THE CAMERA-READY PAPER (both arms are
  history arms). Sample rebuilt by `python3 analysis/absence_misreads_sample.py` (expected
  `paper/absence_misreads/sample.json`); tally `python3 analysis/absence_misreads_tally.py` (expected
  `paper/absence_misreads_tally.txt`): 14 of 40 misread absences (7 + 7), 25 sound but wrong, 1 other misread.
  The coding (`coding.json`) was done by hand and is data, not code.

## E8. Schedule-knowing observer

`python3 analysis/schedule_ceiling.py`, expected `paper/schedule_ceiling.txt`: knows the schedule 82.7 overall
(settled 79.7, disrupted 88.8, back to normal 79.8); knows only the normal routine 72.3; claim store 75.8.

## E9. Day 14 on the movers

- Old rule (drops a household whose days 8-13 mover share is 0): `python3 analysis/day14_and_recovery.py`,
  expected `paper/day14_and_recovery.txt` (range 28.8 to 37.2).
- Over all ten households: printed by the figure script (`REPORT day 14 on movers`), see Figures. With the
  final figure script: 31.5 to 36.8 over the Table 1 methods still in the paper (`range_remaining_ten_household_mean`),
  and 31.0 to 36.8 when history and notes and notes hidden (NOT IN THE CAMERA-READY PAPER) are included
  (`range_all_ten_household_mean`). Output: `out/camera_ready_figures.log`.
- Last-seen updated nightly, ten homes, 76.9 (1,906 of 2,480): see E1.

## Figures

Final figures, v2 of the revised brief (2026-11-04): Figure 2 `figureA1_first_illness`, Figure 3
`figure2_both_illnesses`, Figure 4 `figureA2_what_the_notes_held`, Figure 5 `figure3_told_the_cause`.
The script asserts the values the figure brief lists against the plotted data before it writes a figure.


Figure data and checks come from `analysis/camera_ready_figures.py`, which writes each PDF and a JSON of every
plotted value to `out/figures/camera_ready/`. Expected: `paper/figures/camera_ready/`. Figure 4's bars read
`night31_audit.json` from `python3 analysis/night31_audit.py` (expected `paper/night31_audit.txt` and `.json`).

#!/usr/bin/env bash
# Recompute every paper number and figure from the included data, CPU only, and compare each output with
# the paper's copy in results/self_improve/paper/. No model is called: the only harness runs are the two
# arms that use no model (last-seen and last-seen updated each night), which must reproduce their shipped
# cells search for search. Outputs go to $OUT_DIR (default out/). Exit status is non-zero on any FAIL.
#
#     ./reproduce.sh            # about a minute
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src MPLBACKEND=Agg
# Nothing below calls a model. Point the client at a closed local port so an accidental call fails at once.
export PATROL_LLM_ENDPOINT=http://127.0.0.1:9
export OUT_DIR=${OUT_DIR:-out}
OUT=$OUT_DIR
E=results/self_improve/paper
mkdir -p "$OUT"
fails=0
note() { printf '\n== %s\n' "$*"; }
run() {  # run OUTPUT_NAME COMMAND...: stdout to $OUT/OUTPUT_NAME
  local name=$1; shift
  if ! "$@" > "$OUT/$name" 2> "$OUT/$name.stderr"; then
    echo "FAIL  $name: the command failed"; sed 's/^/      /' "$OUT/$name.stderr" | tail -5; fails=$((fails+1))
  fi
}
cmp_out() { python3 analysis/compare_outputs.py "$1" "$2" || fails=$((fails+1)); }

note "Table 1, ten homes, 8 a day"
run table1_ten_homes.txt python3 analysis/table1_ten_homes.py;           cmp_out $E/table1_ten_homes.txt "$OUT/table1_ten_homes.txt"
run log_only_row.txt python3 analysis/log_only_row.py;                   cmp_out $E/log_only_row.txt "$OUT/log_only_row.txt"
run day14_and_recovery.txt python3 analysis/day14_and_recovery.py;       cmp_out $E/day14_and_recovery.txt "$OUT/day14_and_recovery.txt"

note "E1 (last-seen updated each night) and the new arms, every setting"
run camera_ready_report.txt python3 analysis/camera_ready_report.py;     cmp_out $E/camera_ready_report.txt "$OUT/camera_ready_report.txt"

note "E2 / E2b, claim store plus history and plus today's sightings"
run item2_first_room_right_qwen.txt python3 analysis/item2_first_room_right.py
cmp_out $E/item2_first_room_right_qwen.txt "$OUT/item2_first_room_right_qwen.txt"
{ python3 analysis/item2_first_room_right.py "claim_store_plus_today's_sightings"; echo
  python3 analysis/item2_first_room_right.py claim_store_plus_the_log "claim_store_plus_today's_sightings"; } \
  > "$OUT/item2b_today_sightings_qwen.txt"
cmp_out $E/item2b_today_sightings_qwen.txt "$OUT/item2b_today_sightings_qwen.txt"

note "E3b, five homes at 4 a day and the 50-day homes"
run history_only_q4.txt python3 analysis/history_only_q4.py;             cmp_out $E/history_only_q4.txt "$OUT/history_only_q4.txt"
run history_only_50day.txt python3 analysis/history_only_50day.py;       cmp_out $E/history_only_50day.txt "$OUT/history_only_50day.txt"
run budget_sweep.txt python3 analysis/budget_sweep.py;                   cmp_out $E/budget_sweep.txt "$OUT/budget_sweep.txt"

note "E4, household shapes"
run what_a_household_is_made_of.txt python3 analysis/what_a_household_is_made_of.py
cmp_out $E/what_a_household_is_made_of.txt "$OUT/what_a_household_is_made_of.txt"

note "E5, Sonnet one-household pilot"
run sonnet_pilot_summary.txt python3 analysis/sonnet_pilot_summary.py;   cmp_out $E/sonnet_pilot_summary.txt "$OUT/sonnet_pilot_summary.txt"
run sonnet_pilot_v2_REPORT.txt python3 analysis/report_sonnet_v2.py
cmp_out results/self_improve/camera_ready/sonnet_pilot_v2/REPORT.txt "$OUT/sonnet_pilot_v2_REPORT.txt"

note "E6, why summaries fail"
run claims_contradicting_and_duplicates.txt python3 analysis/claims_contradicting_and_duplicates.py
cmp_out $E/claims_contradicting_and_duplicates.txt "$OUT/claims_contradicting_and_duplicates.txt"
run category_prior_misses.txt python3 analysis/category_prior_misses.py
cmp_out $E/category_prior/category_prior_misses.txt "$OUT/category_prior_misses.txt"
run absence_misreads_sample.log python3 analysis/absence_misreads_sample.py
cmp_out $E/absence_misreads/sample.json "$OUT/absence_misreads/sample.json"
run absence_misreads_tally.txt python3 analysis/absence_misreads_tally.py; cmp_out $E/absence_misreads_tally.txt "$OUT/absence_misreads_tally.txt"
run session_length_and_window_share.txt python3 analysis/session_length_and_window_share.py
cmp_out $E/session_length_and_window_share.txt "$OUT/session_length_and_window_share.txt"

note "E8, schedule-knowing observer"
run schedule_ceiling.txt python3 analysis/schedule_ceiling.py;           cmp_out $E/schedule_ceiling.txt "$OUT/schedule_ceiling.txt"

note "Figure data: night-31 notes audit"
run night31_audit.txt python3 analysis/night31_audit.py;                 cmp_out $E/night31_audit.txt "$OUT/night31_audit.txt"
cmp_out $E/night31_audit.json "$OUT/night31_audit.json"

if [ -f analysis/camera_ready_figures.py ]; then
  note "Figures (camera_ready_figures.py; reads the night-31 audit just written)"
  FIG_OUT="$OUT/figures/camera_ready" NIGHT31_AUDIT="$OUT/night31_audit.json" \
    run camera_ready_figures.log python3 analysis/camera_ready_figures.py
  for f in $E/figures/camera_ready/*.json $E/figures/camera_ready/*.pdf; do
    cmp_out "$f" "$OUT/figures/camera_ready/$(basename "$f")"
  done
fi

note "Harness: the arms' prompts (no model) and two no-model arms rerun from scratch"
for arm in "claim store plus the log" "claim store plus today's sightings"; do
  safe=$(echo "$arm" | sed "s/[ ']/_/g")
  run "prompt_diff_$safe.log" python3 analysis/replay/prompt_diff.py "$arm" "$OUT/prompt_diff_$safe.md"
  cmp_out "$E/prompt_diffs/prompt_diff_$safe.md" "$OUT/prompt_diff_$safe.md"
done
HOMES="hh_s2_t03 hh_s19_t03 hh_s20_t03 hh_s32_t03 hh_s48_t03 hh_s63_t03 hh_s93_t03 hh_s109_t03 hh_s123_t03 hh_s151_t03"
for arm in "last seen, no model" "last seen, updated each night"; do
  dir=$(echo "$arm" | sed 's/, /_/; s/ /_/g')
  for h in $HOMES; do
    python3 -m self_improve.overnight_wave --arm "$arm" --household "$h" \
      --banks results/self_improve/varied_homes/ten_homes/banks --questions-per-day 8 --last-day 31 \
      --out "$OUT/rerun_overnight_wave" --cache "$OUT/llm_cache" > "$OUT/rerun_${dir}_$h.log" 2>&1 \
      || { echo "FAIL  rerun $arm $h (see $OUT/rerun_${dir}_$h.log)"; fails=$((fails+1)); }
  done
  python3 analysis/compare_cells.py "$OUT/rerun_overnight_wave/cells" results/self_improve/overnight_wave/cells "$dir" $HOMES \
    || fails=$((fails+1))
done

echo
if [ "$fails" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "$fails CHECK(S) FAILED"; fi
exit $(( fails > 0 ))

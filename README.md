# Self-improving robot memory

Code and data for the CoRL 2026 workshop paper on self-improving memory for a household robot
(CSIR, "Continually Self-Improving Robots"). Authors and citation: to be added.

## What the paper studies

A robot lives in a simulated household for 31 days (50 in a second setting) and is asked where things are, several
times a day. For each question it may open up to three rooms. Every night it can update its memory from what it saw
that day. On day 14 one resident falls ill, and for ten days their things move to different rooms. On day 24 the
routine returns (in the 50-day homes a second illness starts on day 32). The question is whether a memory that a
language model writes about the household's routine helps the robot find things, especially when the routine
changes. The main measure is **first room right**: the share of searches whose first room held the object.

## Methods

- **last-seen**: no model. Opens the room where the object was last seen, using every sighting up to the moment
  of the question.
- **last-seen, updated nightly**: the same rule, but it only sees sightings up to the previous night, as the
  language-model memories do.
- **claim store**: each night the model (Qwen3.8-27B) edits a list of claims about the household, each with the
  sightings that support or contradict it. The room chooser reads the claims.
- **claim store + today's sightings**: the claim store, plus the sightings of the asked object made earlier today.
- **claim store + history**: the claim store, plus the object's sighting history (its last 40 sightings, failed
  looks folded per room). Nothing else in the prompt changes.
- **ACE-style**: our reimplementation of Agentic Context Engineering (Zhang et al., arXiv 2510.04618): each night the
  writer is told which of the day's searches worked and which claims helped or misled, and may merge claims. "ACE as
  published" is used on the 50-day homes and in the Sonnet pilot; a reduced version ("claim store told if it was
  right") is the ACE-style row of Table 1.

History only, history and notes, notes hidden and small working memory are also in the code and data, because they are
still cited during the revision. They are **not in the camera-ready paper**.

## Names in the code

The paper calls the robot's record the **sighting history**. The code, arm keys and cell directories keep their old
names, because the response cache and the replay checks are keyed on them.

| In the paper | Arm key (`--arm`) | Cell directory |
|---|---|---|
| last-seen | `last seen, no model` | `last_seen_no_model` |
| last-seen, updated nightly | `last seen, updated each night` | `last_seen_updated_each_night` |
| claim store | `incremental edits` | `incremental_edits` |
| claim store + today's sightings | `claim store plus today's sightings` | `claim_store_plus_today's_sightings` |
| claim store + history | `claim store plus the log` | `claim_store_plus_the_log` |
| ACE-style (as published) | `ACE as published` | `ACE_as_published` |
| ACE-style (reduced, Table 1) | `claim store told if it was right` | `claim_store_told_if_it_was_right` |
| small working memory (not in the camera-ready paper) | `a small working memory and an archive` | `a_small_working_memory_and_an_archive` |
| history only (not in the camera-ready paper) | `log only, no notes` | `log_only_no_notes` (waves `wave_log_only*`) |
| history and notes (not in the camera-ready paper) | `the log and notes about the routine` | `the_log_and_notes_about_the_routine` |
| notes hidden (not in the camera-ready paper) | `prior only, no notes` | `prior_only_no_notes` |

"The log", "the raw log" and "the record" in code comments and older files all mean the sighting history.

## Layout

    src/self_improve/        the harness: question loop, room chooser, nightly note writers, arms (overnight_wave.py)
      the_log_the_robot_reads.py   how the sighting history is rendered into the prompt
    src/situation_sim/       the household simulator the banks were generated with
    src/baselines/           shared types, bank loader, and the LLM clients (baselines/patrol/llm.py)
    analysis/                every script behind a paper number or figure; compare_outputs.py checks them
    analysis/replay/         scripts that need the response cache (not included) or no model at all
    results/self_improve/
      varied_homes/*/banks/  the frozen question banks (ten homes; three 50-day homes; five homes, v3)
      <wave>/cells/<arm>/<household>/   one run of one arm in one household
      paper/                 the paper's own copies of every output, for comparison
    NUMBERS.md               each paper number, its command and its expected output
    reproduce.sh             runs everything and compares

Each cell holds `searches.jsonl` (one row per question: rooms opened, reasons, true room, answer), `notes.json`
(the memory at the end of the run, with each claim's revision history), `cell.json` (run settings and per-night
records; its copy of the searches is removed because it equals `searches.jsonl`) and `arm.json`. The 53 cells that
an analysis script reads looks from (the three claim-store arms on the ten homes, history only, and history and
notes) also hold `looks_without_absences.jsonl.gz`: every look and sighting the robot made, with the per-look list
of objects not seen (about 95% of the original file) dropped.

## Reproducing the numbers and figures

Python 3 (tested with 3.13), CPU only, under a minute:

    pip install -r requirements.txt
    ./reproduce.sh

It runs every analysis script on the included cells, writes to `out/`, and compares each output with the paper's
copy (same numbers, in order; for figures, every plotted value in the figure's JSON and the PDF's text layer). It
also reruns the two arms that need no model (last-seen and last-seen updated nightly) on all ten homes from
scratch and checks that every search matches the shipped cell, and checks that the new arms' prompts differ from the
claim store's only in their added block. `NUMBERS.md` maps each number in the paper to its command.

## Rerunning an arm with a model

The language-model arms need an OpenAI-compatible chat-completions server. The paper used vLLM 0.25 serving
Qwen3.8-27B with structured outputs, temperature 0 and seed 0:

    vllm serve Qwen/Qwen3.8-27B --port 8000 --max-model-len 98304 --reasoning-parser qwen3 \
      --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'

Then, from the repository root:

    export PYTHONPATH=src
    export PATROL_LLM_ENDPOINT=http://localhost:8000      # your server
    python3 -m self_improve.overnight_wave --arm "claim store plus the log" --household hh_s2_t03 \
      --banks results/self_improve/varied_homes/ten_homes/banks --questions-per-day 8 --last-day 31 \
      --out runs/qwen_ten_homes_8_a_day --cache llm_cache/self_improve

Always pass `--banks`. Ten homes: `varied_homes/ten_homes/banks`, 8 a day, 31 days. 50-day homes:
`varied_homes/all_generated_v2_twice/banks`, `--questions-per-day 24 --last-day 49`. Five homes:
`varied_homes/all_generated_v3/banks`, 4, 8 or 24 a day, 31 days (some cells record this path as
`varied_homes/headline_five/banks`, which held links to the same five files). Responses are cached by a hash of the request,
so rerunning an interrupted cell takes its earlier calls from the cache.

The response cache from the paper's runs is not included (it is large). Without it, rerunning a language-model arm
makes new model calls and is a new run, not a replay. The scripts in `analysis/replay/` (`replay_check.py`,
`prompts_appendix.py`) need that cache; point `LLM_CACHE` at it. The ACE-style arms also load a small local model
(`meta-llama/Llama-3.2-1B`, via `torch` and `transformers`) to find claims worth merging.

The Sonnet pilot used `--backend anthropic` (or `PATROL_LLM_BACKEND=anthropic`). The `anthropic` client reads
`ANTHROPIC_API_KEY` from the environment; no key is stored in this repository.

## Caveats

**Table 1's claim-store cells do not replay.** The ten-home claim store in Table 1 (`overnight_wave/cells/incremental_edits`,
77.0 first room right) was run with older code, before the reasons-first answer schemas. With the current code it
would need new model calls rather than replaying from the cache. So the claim store + today's sightings and claim
store + history arms are compared with a fresh current-code claim store run on the same ten homes and questions
(`camera_ready/qwen_ten_homes_8_a_day/cells/incremental_edits`, 75.8), not with the Table 1 cell.

**Sonnet refusal retry (pilot only).** Sonnet (claude-sonnet-5-5, thinking off, to match Qwen) refused some calls.
When a call was refused, it was retried once with a few phrases reworded; for example the answer field "reasoning"
became "why" and "think about" became "consider". The full list is `REWORDINGS` in `src/baselines/patrol/llm.py`.
Calls that were not refused were sent unchanged. The retry recovered every refused from-memory answer. A few room
choices stayed refused and fell back to the fixed room order: claim store 1 of 321, ACE 1 of 303 (and history and
notes 14 of 308, history only 7 of 293). Text fields longer than the schema limit were cut to the limit, as Qwen's
constrained decoding does. ACE's nightly judge calls got twice the token limit, because Sonnet wrote longer
judgements that otherwise ended mid-answer. The pilot cost $22.85. It is one household and cannot support a ranking.

## Not included

- The LLM response cache, so language-model arms cannot be replayed offline (see above).
- The full `looks.jsonl` files (13-66 MB per cell). `looks_without_absences.jsonl.gz` (every sighting) is shipped
  only for the 53 cells a script reads, to keep the release under 150 MB.
- The bank generation pipeline (household selection and gating). The banks are shipped frozen.
- The history-and-notes cells at 8 and 24 a day on the five homes (only the draft's budget sweep read them).
- Cells that no paper number reads, earlier exploratory waves, and the first Sonnet pilot, whose refusals were not
  yet retried and whose numbers are not used.

## License

MIT. See `LICENSE`.

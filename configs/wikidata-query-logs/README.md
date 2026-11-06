# Wikidata query logs (WDQL) runs

The WDQL dataset was generated incrementally since November 2025. Each run
used the GRASP code from the `main` branch as it was when the run's `grasp
serve` process was (re)started; this branch only adds the configs and scripts
for the runs.

Tags: `wdql-start` (first run) and `wdql-end` (last run, added once
generation is finished).

| Run | Output period | GRASP commit at start | GRASP commits at likely restarts |
|---|---|---|---|
| qwen3-next-80b-a3b | 2025-11-21 – 2026-03-17 | `dc1cc0c` | – |
| qwen35-27b | 2026-03-19 – 2026-06-01 | `8c65868` | `e11264e` (03-31), `dd098e5` (06-01) |
| qwen36-35b-a3b | 2026-06-02 – 2026-06-19 | `9d30f1a` | – |
| qwen36-27b | 2026-06-19 – 2026-07-21 | `c116acd` | `6fc3d3b` (07-12) |
| qwen35-397b | 2026-07-24 – 2026-09-30 | `90b815f` | `5ca846a` (07-28), `3a33d1e` (08-17), `a8f2067` (09-17) |
| deepseek-v41-flash | 2026-10-06 – 2026-11-02 | `a03b0ba` | `f28403b` (10-21) |
| qwen38-flash-next | 2026-11-03 – | `f28403b` | – |

The qwen35-397b restart on 2026-09-17 picked up `477e1b0`, which changed the
generation prompt to avoid IRIs in questions. Outputs of that run before and
after this date used different prompts.

deepseek-v41-flash was stopped on 2026-11-03 after it hit the daily
university API budget, to keep that budget for other GRASP work. It continued
as qwen38-flash-next (`ufr/coding-fast`, which is not counted against the
budget), with all earlier output directories as skip dirs.

The deepseek-v41-flash restart and the qwen38-flash-next run are taken from
run logs and image build dates. The rest of the table was reconstructed after
the fact:

- Output periods are from file modification times of the outputs. Files
  rewritten later (e.g. by `scripts/fix_prefixes.py`) carry the rewrite date.
- Restarts are inferred from multi-day gaps in the outputs, not from logs.
- Commits are from the git reflog, resolved per day; a commit made on the
  same day as a restart may be missing.
- Uncommitted local changes at the time are not captured.
- The Wikidata endpoint and search indices were rebuilt during this period;
  which versions each run used is not recorded.

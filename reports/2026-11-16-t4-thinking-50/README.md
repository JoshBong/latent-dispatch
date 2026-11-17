# Completed Qwen3-4B thinking continuation: 50 per cell

Completed successfully at 2026-11-17 04:45:13 UTC (November 16, 11:45 p.m. EST).
Verified 2,100 main responses: 350 layouts in each of six conditions, with no
missing or duplicate selected IDs. Qwen/Qwen3-4B, FP16, NVIDIA T4, seed 0;
runtime commit `133b213`. The VM is stopped.

## Scope and cache reuse

Seven cells have 50 layouts each: two-fact toy AB/BA, three-fact toy AB/BA,
and MuSiQue 2-hop AB/BA/BB. There are 150 unique source questions overall;
AB/BA and BB reuse sources and are not independent observations.
B1 reads jointly; K0 stitches independently encoded caches; T1 passes worker
text reports. Each runs with thinking off or forced up to 1,024 tokens.
Final-answer and worker-report caps are also 1,024 tokens. T1 thinking applies
to the orchestrator, not the worker reports.

All 12 saved main/smoke cache prefixes match the pre-continuation byte hashes.
The 1,150 existing main responses were retained unchanged; 950 were added.
The completed 25-per-cell report is preserved separately.

## Exact-match accuracy (%)

| Group | B1 off | B1 on | K0 off | K0 on | T1 off | T1 on |
|---|---:|---:|---:|---:|---:|---:|
| Toy two-fact AB+BA | 100 | 100 | 29 | 74 | 45 | 90 |
| Toy three-fact AB+BA | 100 | 100 | 21 | 60 | 51 | 74 |
| MuSiQue 2-hop AB+BA | 35 | 46 | 17 | 36 | 22 | 21 |
| MuSiQue BB control | 26 | 42 | 34 | 50 | 46 | 48 |

Thinking substantially improves K0 on the toy chains, but joint reading still
leads by 26 points on two-fact and 40 on three-fact chains. Paired, source-cluster
bootstrap 95% intervals for these residual gaps are 19–33 and 31–49 points.
Text reports also benefit on toys, reaching 90% and 74%, respectively.

On split MuSiQue, K0 improves 17→36% and B1 improves 35→46%. The B1–K0 gap
shrinks 18→10 points; its residual interval is 2–18 points, while the gap-closure
interval is −5 to +20. T1 remains approximately flat at 22→21%.
These are descriptive pilot estimates from 50 sources per task group; no
multiple-testing adjustment is applied.

## Output checks and limits

K0 has four truncated final answers: three with thinking on and one with it
off. These four also lack boxed answers. They remain in the denominator and
are scored from the saved text; no outputs were discarded or regenerated.
Other conditions have zero final-answer truncations or missing boxes. Worker
reports have zero truncations. All forced-thinking outputs have a closing
`</think>` marker.

| Method | Mean thinking tokens | Hit 1,024-token thinking cap |
|---|---:|---:|
| B1 | 400 | 41/350 (11.7%) |
| K0 | 651 | 127/350 (36.3%) |
| T1 | 690 | 145/350 (41.4%) |

K0 and T1 use more reasoning tokens than B1, so the comparison is not at equal
realized compute. Results cover one model, seed and maximum thinking budget.
MuSiQue numbers use the existing exact-match scorer and aliases, without a
new manual entity-equivalence review. Answer-form and source-quality issues
may affect those numbers; do not present them as adjudicated semantic accuracy.
Per-response token F1 is also saved in scored-responses.csv.

## Artifacts and reproduction

- results-and-logs.tar.gz: raw responses, runtime scripts, logs/status, launch
  manifest, benchmark checksums and selected question metadata. SHA256:
  `89bc44a2e4648d8f64d677e10b6de2e8dc6ba4f224c7246d352c1ce82b30b7b3`.
- analyze.py: validates IDs/options, preserved cache prefixes and benchmark
  checksums, rescores all rows and computes 10,000 source-cluster bootstrap
  resamples. Run `python reports/2026-11-16-t4-thinking-50/analyze.py` with NumPy.
- analysis.json and scored-responses.csv: reproducible numerical results.
- question-metadata.json: selected layouts, gold aliases and source IDs.
- thinking-experiment.status/log and thinking-pilot-launch.json: runtime records.
- resume-baseline.json: pre-continuation hashes/counts/options.
- SHA256SUMS: checksums of report files.

The automatic publisher did not retrieve this outcome before shutdown. Recovery
used a snapshot of the stopped disk, a disposable copy mounted read-only with
journal replay disabled, and a CPU-only helper. The downloaded archive matched
the helper's checksum. The original stopped VM and disk were unchanged.
The custom archive is not directly compatible with results.load_archive.

# Completed Qwen3-4B thinking pilot — November 16, 2026

The run completed successfully at **2026-11-16 10:38:27 UTC (5:38 a.m. EST)**.
Validated **1,050 selected responses**, 175 per condition, with no duplicate or
missing selected IDs. The six conditions have zero answer truncations and zero
missing boxed answers. T1 has zero truncated worker reports.

## Scope and provenance

- Qwen/Qwen3-4B, FP16, NVIDIA T4; seed 0.
- 25 per cell: toy two-fact and three-fact AB/BA; MuSiQue 2-hop AB/BA and BB control.
- 175 question layouts from **75 unique source questions**. AB/BA and the MuSiQue
  BB control reuse the same source questions; they are not independent samples.
- B1 joint reading, K0 independently encoded/stitched caches, T1 worker text reports;
  each with thinking off or forced up to 1,024 tokens. T1 thinking applies to the
  orchestrator, not worker reports. All final-answer caps are 1,024 tokens; each
  worker report has a 1,024-token cap.
- Source commit: `70640bd` (25-per-cell checkpoint branch).
- Thinking uses fix/think-eos: EOS during reasoning closes thinking and continues
  to a final answer. All 18 FP16 real-model checks passed before inference.
- The original 50-per-cell run was reduced after a frozen, hashed backup.
  **All 232 main responses saved at that point remain byte-identical** in the
  final archive. B1 has 275 raw main rows: 175 selected plus 100 retained extras.
  The other conditions each have 175 main rows. The tables below exclude extras
  and smoke outputs. Old failed smoke artifacts are preserved in the bundle.

## Exact-match accuracy (%)

AB and BA are pooled, with 25 source questions / 50 layouts per split group.

| Group | B1 off | B1 on | K0 off | K0 on | T1 off | T1 on |
|---|---:|---:|---:|---:|---:|---:|
| Toy two-fact AB+BA | 100 | 100 | 28 | 74 | 40 | 94 |
| Toy three-fact AB+BA | 100 | 100 | 26 | 70 | 52 | 70 |
| MuSiQue 2-hop AB+BA | 30 | 38 | 12 | 32 | 18 | 12 |
| MuSiQue 2-hop BB control | 20 | 32 | 28 | 40 | 28 | 40 |

## Brief interpretation

**Reasoning materially reduces the separate-cache deficit.** On two-fact toy
chains K0 rises 28→74%; on three-fact chains, 26→70%, while B1 remains 100%.
The corresponding B1–K0 gaps shrink 72→26 and 74→30 percentage points. Paired
source-cluster bootstrap intervals for gap closure are +28 to +64 and +28 to +60
points, respectively. A substantial residual toy gap remains under this budget.

On split MuSiQue, K0 rises 12→32% and B1 rises 30→38%. The gap shrinks from 18 to
6 points. The residual gap's paired 95% bootstrap interval is −4 to +16 points;
the gap-closure interval is −2 to +26. This small pilot does not establish either
that the gap persists or that it disappears on MuSiQue.

**Text passing is mixed.** T1 improves 40→94% on two-fact toy chains and 52→70%
on three-fact chains, but falls 18→12% on split MuSiQue. This is not evidence of
a general text-passing improvement. In the same-worker MuSiQue control, B1,
K0 and T1 rise 20→32%, 28→40%, and 28→40%, respectively.

## Thinking length and limitations

| Method | Mean thinking tokens | Hit 1,024-token thinking cap |
|---|---:|---:|
| B1 | 408 | 22/175 (12.6%) |
| K0 | 663 | 63/175 (36.0%) |
| T1 | 717 | 78/175 (44.6%) |

Thinking caps are experimental budgets, not truncated final answers. Actual
reasoning compute differs between methods despite equal maximum budgets: K0
and T1 think longer than B1. Comparisons therefore do not isolate architectural
changes at equal realized compute. Results are one model, one seed, 25 source
questions per task group, and a fixed thinking budget; no claim that unlimited
reasoning cannot recover the gap is supported.

Intervals use 10,000 paired bootstrap resamples over source questions, keeping
AB/BA together. They are descriptive pilot intervals, without multiple-testing
correction. Full per-response scores, audit counts, and paired comparisons are
in `scored-responses.csv` and `analysis.json`.

## Artifacts and reproduction

- `results-and-logs.tar.gz`: final raw rows (including retained extras), manifests,
  runtime scripts, full log/status, and benchmark checksums. SHA256:
  `5b43ad2a8d7a1c2f8ca589da1629f2144397ca9f6345f2a1d78698226220153d`.
- `question-metadata.json`: selected question IDs, answers, source IDs and layout metadata.
- `analyze.py`: deterministic validation, rescoring and clustered bootstrap.
  From the repo root, run `python3 reports/2026-11-16-t4-thinking-25/analyze.py`
  with NumPy available. This does not load model weights or require a GPU.
- `thinking-pilot-launch.json`, `thinking-experiment.status`, and
  `thinking-experiment.log`: direct copies of the final runtime records.
- `SHA256SUMS`: checksums of this report's files.

The custom archive contains logs and code and is not accepted directly by
`results.load_archive`. Read its members explicitly, or extract only the named
`latent-dispatch-thinking/results/` members with one leading component removed.

The automatic publisher did not retrieve the final bundle before shutdown.
Recovery used a stopped-disk copy on a temporary CPU VM because restarting the
T4 encountered a GPU capacity shortage. The downloaded archive checksum matches
the recovery host's checksum; the original stopped VM and disk were unchanged.

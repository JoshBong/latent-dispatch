# Completed Qwen3-4B FP16 same-worker control

Completed successfully at **3:56 p.m. EST on November 18, 2026** (20:56:24 UTC),
about 46 minutes after the observed wrapper start. All **800 control answers**
are present: the same 200 sources, AA and BB layouts, blocked and full access.
All four longest-layout smoke checks passed first. No sources or layouts were
dropped. Results were pushed automatically, and the L4 is stopped.

**The control supports a layout-dependent masking effect on the primary F1
metric.** Full access helps by +6.18 F1 points with facts split across workers,
but averages −0.44 points with both support paragraphs in the same worker.
The matched-source interaction is **+6.62 points [95% CI +1.97, +11.28]**.
This is stronger evidence than merely noting that the control interval covers zero.

| Supporting facts | Blocked F1 | Full F1 | Full − blocked | 95% paired interval |
|---|---:|---:|---:|---:|
| Split across workers: AB/BA main run | 33.49 | 39.67 | +6.18 | +2.57 to +9.85 |
| Together in one worker: AA/BB control | 42.80 | 42.36 | −0.44 | −3.62 to +2.75 |
| Split gap minus together gap | | | **+6.62** | **+1.97 to +11.28** |

F1 is on a 0–100 scale; differences are percentage points. Analysis averages
the two layouts within each source, then bootstraps the same 200 paired sources
20,000 times, seed 20261116. Intervals are unadjusted percentile intervals.
Fixed accepted answers and the scorer are unchanged from the main run.

A defensible paper statement is: *On these sources, cross-worker attention has
a larger benefit when the supporting facts are split across workers than when
they are co-located.* This is consistent with the main penalty involving
cross-worker fact integration, rather than an undifferentiated masking cost.
It does not uniquely identify the internal mechanism: moving paragraphs also
changes their positions and surrounding distractors.

**Do not equate the pooled control mean with proof of no effect.** Its AA and BB
point estimates point in opposite directions:

| Control layout | Blocked F1 | Full F1 | Full − blocked | 95% interval |
|---|---:|---:|---:|---:|
| AA: both supports in A | 43.01 | 45.90 | +2.89 | −1.39 to +7.30 |
| BB: both supports in B | 42.59 | 38.82 | −3.78 | −8.75 to +1.25 |

The pooled near-zero result partly reflects that cancellation. Neither layout
has a tight equivalence bound; “blocking costs exactly nothing” is too strong.
The EM interaction is +4.00 points, interval −0.75 to +8.75, so the primary F1
result is clearer than the secondary exact-match result. All detailed F1/EM
levels and contrasts are in FINAL.md and analysis.json.

**Relation to the other model runs.** This task ran only Qwen3-4B. Re-scoring
the pre-existing Qwen3-8B and Llama archives gives the same direction for the
split-minus-together F1 interaction:

| Model | Interaction | 95% interval |
|---|---:|---:|
| Qwen3-4B, FP16/L4 (this run) | +6.62 | +1.97 to +11.28 |
| Qwen3-8B, BF16/A100 (existing archive) | +5.95 | +1.21 to +10.80 |
| Llama 3.1 8B, BF16/A100 (existing archive) | +6.37 | +1.03 to +11.70 |

A secondary equal-model average, formed within source before bootstrapping,
is +6.31 points [+3.19, +9.44]. These are the same 200 sources, not 600
independent observations. Precision/hardware vary across model settings, so
this does not isolate model size or establish universality. compare_models.py
and cross-model.json reproduce this supplemental comparison; the repository's
existing two-model paper-table output was not silently changed.

**Output quality and data limits.** Two blocked answers hit the 1,024-token
answer cap; three blocked answers lack a box. Full access has no capped or
unboxed answers. All rows remain in the fixed-score analysis. Annotation
ambiguities identified in the main-run analysis still apply here: the control
uses the same questions and accepted answers, without manual adjudication.
The report does not claim human-verified semantic accuracy.

**Execution and provenance.** Qwen/Qwen3-4B revision
1cfa9a7208912126459214e8b04321603b3df60c, float16, NVIDIA L4, torch 2.6.0+cu124,
transformers 4.57.6, SDPA, greedy decoding, thinking off. No optional B1 arm,
position follow-up, quantization, or chunked-prefill experiment was run.
The exact control script was extracted unchanged from colab_wmac_together.ipynb
at 9747d6f, SHA256
e1f9b72fc3b3b5ead0b54e42373e0057ab487f5a9d5009f0c1cb51a1dbe08de2.
Core model/masking code matches 64044d8 byte-for-byte. Runtime checkout commit:
848fca7. The notebook's outer setup was updated to allow MODELS=["4b"] on an
L4 and explicitly pass --dtype float16; 8B defaults remain BF16/A100.

The unchanged script selftest passed. The 4B tokenizer matches the frozen main
bundle; all 400 layouts fit (1,165–4,646 tokens, median 2,406). AA/BB preserve
the full paragraph multiset and each worker's paragraph count, placing both
support paragraphs in the chosen worker's final two slots, in hop order.
Every saved question hash, plan, option set, source ID, model revision, dtype,
and complete unique ID/condition set was checked against the main bundle.
Prepared SHA256:
8abf29425611b1f3b65f5016fd40651ef0281bdb6de9852e1df2940f33953058.

**Artifacts.** results-and-logs.tar.gz preserves all raw rows, plan/status,
smoke results, execution log, package freeze, wrapper and runtime library code.
Its SHA256 is
364e67317d101cc3ff652700274c56a2900ecb85536ec7a413cd6b2985682985.
The same rows/plan/status are also supplied in the conventional compact archive:
../../archive/wmac_gpu_v3/l4-4b-together-2026-11-18.tar.gz, under
wmac_gpu_v3_together/4b/together/.

```bash
# From the repository root, with its Python dependencies installed:
python reports/wmac-4b-together/analyze.py
python reports/wmac-4b-together/compare_models.py
# Verify original report files before regenerating them:
(cd reports/wmac-4b-together && sha256sum -c SHA256SUMS)
```

The publisher collected the archive by SSH, verified its hash, analyzed it,
pushed it, and stopped the VM. An independent post-shutdown check reconstructed
the byte-identical archive from Cloud Logging; see serial-backup-verification.json.
The decoder was corrected to include continuation records and handle escaped
CR/LF plus interleaved syslog messages. This transport correction changes no
experimental code or output. Both checksum and missing-chunk checks are mandatory.

VM: latent-wmac-l4-v3, us-central1-b. Unit: latent-wmac-4b-together.
Remote checkout: /home/elkalt/latent-dispatch-wmac-together.
Status/log: /home/elkalt/wmac-4b-together.status and wmac-4b-together.log.
The consumed reservation and unused snapshot were removed. No GPU remains running.

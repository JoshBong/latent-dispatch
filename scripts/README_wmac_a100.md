# WMAC follow-ups for Josh's A100

Three experiments, four jobs. These are inference/evaluation jobs, not training.
This package was written and statically reviewed; **its tests, preparation,
tokenizer checks, smoke runs, and GPU experiments have not been executed for
this revision**. Run the supplied checks and smoke commands before full runs.

## Checkout and environment

Use a clean checkout of branch `codex/wmac-a100-followups`. Record its commit ID
and keep that checkout unchanged while jobs run or resume. It includes the
earlier paper runner dependency, so PR #2 need not be merged separately.

Use Python 3.11+ and CUDA-enabled PyTorch on an A100 40 GB or larger:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_wmac_a100.py tests/test_masking.py tests/test_paper_ablations.py
```

The tests use tiny random CPU models and a tokenizer; they do not run the 4B/8B
experiments. Do not interpret this document as a record that these checks passed.

## Prepare once, before inference

```bash
python -m latent_dispatch.data toy musique
python scripts/run_wmac_a100.py prepare
```

If the benchmark files already exist, skip the builder: it refuses to overwrite
them, and preparation verifies the repository's SHA256 checksums. Preparation
uses the pinned 8B tokenizer but does not download model weights or run inference.

Preparation writes `results/wmac_a100_v1/prepared.json` plus
`prepared-review.json`. Both model sizes use this same frozen set of 200 MuSiQue
sources. The bundle includes all questions, hashes, the historical exclusion
list, input-only screening decisions, token preflight, and the full protocol.
Re-running preparation loads the existing bundle; it does not select new data.

The review packet includes questions, both supporting paragraphs, decompositions,
and accepted answers. Automatic grounding checks do **not** establish semantic
answerability or eliminate ambiguous annotations. Review inputs before inspecting
main outputs. Human exclusions or aliases must be frozen prospectively in a new
bundle, using a JSON object keyed by source ID, with `reviewer`, `reason`, and
optional `exclude: true` or `aliases: ["..."]`:

```bash
python scripts/run_wmac_a100.py prepare --reviews reviewed_sources.json --run-name wmac_reviewed
```

Then use `--run-name wmac_reviewed` for every job. Do not silently retrofit new
aliases or exclusions to already-observed experimental results.

## Four A100 commands

Run validation first and inspect its diagnostic outputs. Completion means
diagnostics were collected, not that numerical equivalence was established.

```bash
python scripts/run_wmac_a100.py validation
python scripts/run_wmac_a100.py musique-8b
python scripts/run_wmac_a100.py musique-4b
python scripts/run_wmac_a100.py toy-8b
```

Run sequentially on one GPU in a durable session such as tmux, saving to durable
storage. Add `--smoke-only` to any command to stop after its smoke pass, then
repeat without it to continue. Smoke prints raw answers. Wrong answers, malformed
boxes, and cap hits are outcomes; the smoke check does not filter those examples.
Empty outputs or unclosed forced-thinking output stop the run for inspection.

`--plan-only` validates the prepared inputs and writes the job manifest without
loading weights. `--summarize` rebuilds tables and the archive without weights.
The default soft time limit is 24 hours per job (`--max-hours` changes it).
Signals and time limits stop between evaluations and return exit code 2 when
incomplete. Repeat the exact command to resume. Do not change code, precision,
backend, model, data, or package versions inside an existing run.

## Counts and fixed settings

| Job | Main answers | Additional work |
| --- | ---: | --- |
| validation | 24 layouts × 4 paths = 96 | 48 shared-prefix diagnostic pairs; 16 smoke answers |
| musique-8b | 200 sources × 2 orders × 4 masks = 1,600 | 600 shortcut probes; 8 smoke answers |
| musique-4b | Same source list and conditions = 1,600 | 600 shortcut probes; 8 smoke answers |
| toy-8b | 50 chains × 2 orders × 2 placements × 2 headings × 2 masks × 2 thinking modes = 1,600 | 32 smoke answers |

Total: **4,896 main answers + 1,200 shortcut-probe answers + 64 smoke answers**.
Teacher-forced diagnostic pairs reuse fixed continuations and are not additional
generated answers. Keep smoke/probe outcomes out of the primary sample counts.

BF16, SDPA, greedy decoding, one sequence at a time. Main answer cap: 1,024
tokens. Forced-thinking toy cells get 1,024 thinking tokens separately, then an
answer budget of 1,024. Validation and shortcut probes use 256 answer tokens.
No 4,096-token thinking extension. Every cap hit and forced thinking closure is
recorded; no capped main answer is discarded.
Thinking off closes the dedicated thinking channel; it does not prevent reasoning
in ordinary answer text.

8B is pinned to revision `b968826d9c46dd6066d109eabc6255188de91218` from the
archived run. The first 4B execution records its resolved revision and pins
resumes to it. All jobs save hardware, package versions, tokenizer/config hashes,
attention backend and precision. TF32 is disabled. The runner refuses contexts
over 8,192 tokens; it does not truncate or silently replace selected questions.

## Experiment 1: implementation correspondence

Use the previous 24 deliberately selected diagnostic layouts. Compare joint
reading versus masked-full, and true-position separate caches versus
masked-blocked. Save full first-answer logits and free-generation traces. Then
feed both implementations the **same reference-generated continuation**, including
after their free answers diverge. Save per-position argmax, maximum/mean logit
differences, KL, top-two margins, and complete logit vectors at fixed prefix
positions and the first teacher-forced disagreement.

Numerical differences are evidence to inspect, not an automatic bug/no-bug verdict.
For a separate precision check, if memory permits:

```bash
python scripts/run_wmac_a100.py validation --dtype float32 --validation-limit 8 --prepared results/wmac_a100_v1/prepared.json --run-name wmac_fp32_diagnostic
```

This uses the eight shortest frozen diagnostic layouts. FP32 8B may exceed
40 GB; an OOM is a failed diagnostic, not evidence of parity. Precision and
backend never silently fall back or mix with the BF16 results.

## Experiment 2: fresh MuSiQue confirmation, 8B and 4B

Select 200 new source questions deterministically. Exclude previously used IDs,
reused supporting paragraph text (ignoring titles), and duplicate questions.
Require a literal bridge in both supports and an accepted answer in the second
support but not the first. Token feasibility is checked before selection is
frozen. These are **automatic checks, not human answerability certification**.

All 200 remain in the primary cohort. Closed-book and single-support probes
(retaining all distractors) run on AB inputs. The strict sensitivity cohort
requires F1 < 0.5 in all three probes with no probe truncation. Probe results
never replace questions or filter the primary cohort. F1 is primary; alias-aware
EM uses the same frozen dataset/review answers for every condition.

Four masks: blocked, gold, full, and equal-token irrelevant access. Put the
support last in each worker under **all** masks so earlier irrelevant keys cannot
already have attended to the annotated support. This intentional layout change
from the historical sample must be stated in the paper.

The irrelevant control is a deterministic window over eligible non-support token
indices before A's support. It may cover parts of multiple paragraphs. Exclude
paragraphs containing literal answers/bridges, and record potential earlier
prefix exposure. It matches token count, **not** key positions/distances or all
semantic information. Inputs, positions, and final-answer access are identical
across the four masks. Do not call this a fully isolated semantic intervention,
a learned selector, or an efficient deployed communication protocol.

## Experiment 3: controlled paragraph confusion, 8B

Keep both gold facts split. Gold People remains in slot 0 of its worker;
distractor People stays in slot 0 of the other; gold Cities stays in slot 1 of
that other worker. Only distractor Cities and a neutral filler exchange slots.
Each worker has three paragraphs. Freeze token-balanced fillers and assert equal
worker token counts and gold positions across all factorial variants.

Cities headings are identical or distinct (`Cities A`/`Cities B`), counterbalanced
across sources. Test blocked/gold masks with thinking off/forced. These new
stimuli should be compared internally; their absolute scores are not a matched
comparison to the historical toy. A heading benefit tests an addressing-cue
explanation. Its absence does not establish a different internal mechanism.
Neutral fillers remain artificial; equal length does not imply equal semantics.

## Return complete bundles and inspect raw answers

Each command writes a `.tar.gz` beside its job directory under
`results/wmac_a100_v1/`. It includes the prepared questions, manifest/runtime,
raw smoke/probe/main outputs, numerical artifacts, status, and summary tables.
Keep the whole validation archive, including the logit files missing from the
previous handoff. No tool sends the archives or messages to teammates.

`main/summary.json`, `cells.csv`, and `answers.csv` contain F1/EM, timing/token/cap
metrics, order splits, and paired source-bootstrap intervals (20,000 resamples).
F1 and EM are stored as fractions from 0 to 1; multiply by 100 for percentage tables.
MuSiQue adds strict-probe sensitivity and fixed support-length buckets. Toy adds
heading-by-placement interactions and thinking fix/break counts. Intervals are
unadjusted; exploratory slices are not multiple-testing-corrected significance.
Cap-hit subsets are descriptive, not causal. Partial jobs report their incomplete
status and only fully paired sources enter paired estimates.

The scripts prepare analysis artifacts; human input review, raw-answer inspection,
mechanistic interpretation, and the four-versus-eight-page decision remain research work.

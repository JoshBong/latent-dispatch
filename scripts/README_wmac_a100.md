# WMAC follow-ups for Josh's A100

V2 has four primary jobs and two implementation-validation jobs. These are
inference/evaluation jobs, not training. The primary jobs produce **8,000 answers**.
This package was written and statically reviewed; **its tests, preparation,
tokenizer checks, smoke runs, and GPU experiments have not been executed for
this revision**. Run the supplied checks and smoke commands before full runs.

## Checkout and environment

Use a clean checkout of branch `codex/wmac-a100-followups`. Record its commit ID
and keep that checkout unchanged while jobs run or resume. It includes the
earlier paper runner dependency, so PR #2 need not be merged separately.

The branch is published through PR #3 from the `ajit-mallav` fork. For a separate
checkout that leaves any existing V1 run alone:

```bash
git clone --single-branch --branch codex/wmac-a100-followups https://github.com/ajit-mallav/latent-dispatch.git latent-dispatch-wmac-v2
cd latent-dispatch-wmac-v2
git rev-parse HEAD
```

Use Python 3.11+ and CUDA-enabled PyTorch on an A100 40 GB or larger:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_wmac_a100.py tests/test_masking.py tests/test_paper_ablations.py
```

The tests use tiny random CPU models and a tokenizer; they do not run the 4B/8B
experiments. Do not interpret this document as a record that these checks passed.

Llama uses `meta-llama/Llama-3.1-8B-Instruct`. Before preparation, the Hugging Face
account on the A100 must have accepted access to that gated repository and be
authenticated. There is no automatic model substitution if access is unavailable.

## Prepare once, before inference

```bash
python -m latent_dispatch.data toy musique
python scripts/run_wmac_a100.py prepare
```

If the benchmark files already exist, skip the builder: it refuses to overwrite
them, and preparation verifies the repository's SHA256 checksums. Preparation
resolves immutable model revisions and uses all three tokenizers, but does not
download model weights or run inference.

Preparation writes `results/wmac_a100_v2/prepared.json` plus
`prepared-review.json`. All three models use this same frozen set of 200 MuSiQue
sources. The bundle includes all questions, hashes, the historical exclusion
list, input-only screening decisions, per-model token preflight, and the full protocol.
Re-running preparation loads the existing bundle; it does not select new data.

If V1 preparation has already selected the 200 questions, use this command
**instead of** plain V2 preparation, before the V2 destination exists:

```bash
python scripts/run_wmac_a100.py prepare --reuse-prepared results/wmac_a100_v1/prepared.json
```

This preserves the exact source texts, orders, and accepted answers. It checks
feasibility with all three tokenizers and stops if any selected question fails;
it never replaces a source. Without reuse, shared-tokenizer feasibility can select
a different cohort from V1. V2 never imports V1 generated answers. Existing V1
jobs must continue on their original pinned checkout; do not change a running
checkout or point V2 inference directly at a V1 bundle.

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

## Six A100 commands

Run both validation jobs first and inspect their diagnostic outputs before the
primary jobs. Completion means diagnostics were collected, not that numerical
equivalence was established.

```bash
python scripts/run_wmac_a100.py validation
python scripts/run_wmac_a100.py validation-llama
python scripts/run_wmac_a100.py musique-8b
python scripts/run_wmac_a100.py musique-llama
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

Prioritize the fresh MuSiQue jobs. If time is tight, defer the entire toy
diagnostic rather than dropping factorial control cells. Timings for the new
Llama and actual K0 paths are not established; use their smoke timings to estimate
remaining work, and report actual per-condition timing from the final bundles.

## Counts and fixed settings

| Job | Main answers | Additional work |
| --- | ---: | --- |
| musique-8b | 200 sources × 2 orders × 6 conditions = 2,400 | 600 shortcut probes; 12 smoke answers |
| musique-llama | Same source list and conditions = 2,400 | 600 shortcut probes; 12 smoke answers |
| musique-4b | Same source list and conditions = 2,400 | 600 shortcut probes; 12 smoke answers |
| toy-8b | 25 chains × 2 orders × 2 placements × 2 headings × 2 masks × 2 thinking modes = 800 | 32 smoke answers |
| validation | 24 layouts × 4 paths = 96 | 48 shared-prefix diagnostic pairs; 16 smoke answers |
| validation-llama | Same diagnostic layouts and paths = 96 | 48 shared-prefix diagnostic pairs; 16 smoke answers |

Total: **8,000 primary answers + 192 validation answers + 1,800 shortcut-probe
answers + 100 smoke answers = 10,092 generated answers**.
Teacher-forced diagnostic pairs reuse fixed continuations and are not additional
generated answers. Small forward-only RoPE checks also generate no answers.
Keep validation/smoke/probe outcomes out of the primary sample counts.

BF16, SDPA, greedy decoding, one sequence at a time. Main answer cap: 1,024
tokens. Forced-thinking toy cells get 1,024 thinking tokens separately, then an
answer budget of 1,024. Validation and shortcut probes use 256 answer tokens.
No 4,096-token thinking extension. Every cap hit and forced thinking closure is
recorded; no capped main answer is discarded.
For Qwen3, thinking off closes the dedicated thinking channel. Llama uses its own
native template and no Qwen thinking tags. Neither prevents reasoning in ordinary
answer text; these are not matched reasoning-model architectures.

Qwen3-8B is pinned to revision `b968826d9c46dd6066d109eabc6255188de91218` from the
archived run. Preparation resolves and freezes Qwen3-4B and Llama revisions before
any outputs. All jobs save hardware, package versions, tokenizer/config hashes,
attention backend and precision. TF32 is disabled. The runner refuses contexts
over 8,192 tokens; it does not truncate or silently replace selected questions.

## Experiment 1: implementation correspondence

Run on Qwen3-8B and Llama using the previous 24 deliberately selected diagnostic
layouts. Compare joint reading versus masked-full, and true-position separate caches versus
masked-blocked. Save full first-answer logits and free-generation traces. Then
feed both implementations the **same reference-generated continuation**, including
after their free answers diverge. Save per-position argmax, maximum/mean logit
differences, KL, top-two margins, and complete logit vectors at fixed prefix
positions and the first teacher-forced disagreement.

`b1_mask` in this validation job means **true-position separate caches**. It is
not original K0. Original K0 encodes every worker at local positions after the
shared prefix, then shifts keys and stitches caches. It is evaluated directly
on all 200 fresh sources in each MuSiQue job below.

Preparation checks native chat templates and segmented tokenization for each
model. Every inference job repeats the relevant checks and tests shifted
layer-0 RoPE keys against native encoding at shifted positions. That small
forward check tests key rotation, not equivalence of full K0 and blocked-mask
computation. All six primary MuSiQue conditions also check final cache length
and answer position.

Numerical differences are evidence to inspect, not an automatic bug/no-bug verdict.
For a separate precision check, if memory permits:

```bash
python scripts/run_wmac_a100.py validation --dtype float32 --validation-limit 8 --prepared results/wmac_a100_v2/prepared.json --run-name wmac_fp32_diagnostic
```

This uses the eight shortest frozen diagnostic layouts. FP32 8B may exceed
40 GB; an OOM is a failed diagnostic, not evidence of parity. Precision and
backend never silently fall back or mix with the BF16 results.

## Experiment 2: fresh MuSiQue confirmation, Qwen3-8B, Llama-3.1-8B, Qwen3-4B

Select 200 new source questions deterministically. Exclude previously used IDs,
reused supporting paragraph text (ignoring titles), and duplicate questions.
Require a literal bridge in both supports and an accepted answer in the second
support but not the first. Feasibility with all three tokenizers is checked
before selection is frozen. These are **automatic checks, not human
answerability certification**.

All 200 remain in the primary cohort. Closed-book and single-support probes
(retaining all distractors) run on AB inputs. The strict sensitivity cohort
requires F1 < 0.5 in all three probes with no probe truncation. Probe results
never replace questions or filter the primary cohort. F1 is primary; alias-aware
EM uses the same frozen dataset/review answers for every condition and model.
Strict cohorts are model-specific sensitivity analyses; compare families on
the complete shared cohort rather than treating their strict subsets as matched.

Six conditions: blocked, gold, full, equal-token irrelevant access, actual B1
joint reading, and original K0 with local positions and shifted-key stitching.
The latter two each add 400 answers per model. Compare B1 minus K0 directly;
B1 minus full-mask and blocked-mask minus K0 are implementation comparisons,
not extra semantic interventions. Blocked minus K0 includes position/stitching
effects as well as numerical differences.

Put the support last in each worker under **all six** conditions so earlier
irrelevant keys cannot already have attended to the annotated support. This intentional layout change
from the historical sample must be stated in the paper.

The irrelevant control is a deterministic window over eligible non-support token
indices before A's support. It may cover parts of multiple paragraphs. Exclude
paragraphs containing literal answers/bridges, and record potential earlier
prefix exposure. It matches token count, **not** key positions/distances or all
semantic information. Inputs, positions, and final-answer access are identical
across the four masks. Do not call this a fully isolated semantic intervention,
a learned selector, or an efficient deployed communication protocol.

Exposed token counts and selected irrelevant-token spans can differ between
tokenizers. Interpret within-model paired effects first; model-family contrasts
also differ in training, tokenizer, and native prompting.

## Experiment 3: controlled paragraph confusion, 8B

Select 25 fresh chains before inspecting outputs; retain every factorial cell.
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
`results/wmac_a100_v2/`. It includes the prepared questions, manifest/runtime,
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

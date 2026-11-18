# Three experiments to complete the four-page paper

These runners implement the agreed 96 baseline generations, 600 masked-thinking
generations, and 255 likelihood evaluations. **No experiment results are included
in this branch.** Planning, CPU unit tests, and inspection of existing archives
do not start an 8B run.

Use an A100-class GPU with at least 30 GiB for the default Qwen3-8B/bfloat16
protocol. The code does not provision a VM, send messages, or shut a machine down.

## Setup

Use a separate checkout so existing notebooks/results stay intact:

```bash
git clone --branch codex/paper-completion-ablations https://github.com/ajit-mallav/latent-dispatch.git latent-dispatch-paper
cd latent-dispatch-paper
python -m pip install -e '.[dev]'
python -m latent_dispatch.data toy musique
```

The last command builds the deterministic benchmark files, not model outputs.
If those files already exist, skip that command. Every runner verifies both
files against the committed checksums. `--benchmark-dir /existing/benchmark`
can reuse an existing verified copy instead of rebuilding/downloading data.

The frozen selection is in `experiments/paper_ablations_v1.json`: question IDs,
question hashes, archived endpoint labels, candidate answers, budgets, and the
SHA256 of the reviewed 8B archive. Samples use selection seed 20261116 and the
original seed-0 benchmark; decoding is greedy. No new response selects a sample.

Preview all three jobs without importing torch or loading weights:

```bash
python scripts/run_paper_ablations.py validation --plan-only
python scripts/run_paper_ablations.py thinking --plan-only
python scripts/run_paper_ablations.py likelihood --plan-only
```

## 1. Baseline validation: run first

```bash
python scripts/run_paper_ablations.py validation
```

24 question layouts: 12 with an archived correctness disagreement and 12 with
agreement, selected across toy/MuSiQue and AB/BA. Four implementations each:
original B1, masked-full, original B1-mask at true positions, masked-blocked.
Thinking is off and the answer cap is 256, matching the earlier baseline setup.
The same model and explicit SDPA backend execute all four implementations.

Outputs include full first-answer-token logits (`main/logits/*.npz`), context
and position hashes, generated token IDs, and top-five logits at each generation
step (including the stopping decision). `main/summary.json` compares endpoints,
reports first divergence and local traces, distribution KL, logit differences,
top-two margins, and correctness differences. This is a purposively selected
diagnostic sample, **not an accuracy estimate or an automatic parity approval**.

Inspect these diagnostics before the thinking experiment. A numerical near-tie
is a possible explanation, not a conclusion the code assumes. If needed, a
separate diagnostic can request `--dtype float32` or `--attention eager` with a
new `--run-name`; those runs are outside the agreed 951 evaluations. Float32
needs more GPU memory. Do not change precision/backend inside an existing run.

## 2. Matched masked-thinking comparison

```bash
python scripts/run_paper_ablations.py thinking
```

25 synthetic and 25 MuSiQue source questions, drawn from the original 8B
composition-filtered cohort, each in AB and BA: 100 layouts. Each layout has
blocked/gold-only/full attention with thinking off/forced: 600 generations.
All conditions use a 1,024-token answer cap. Forced thinking allows up to 1,024
thinking tokens before answering; naturally shorter thinking is retained.
The EOS-inside-thinking fix is included. Workers and evidence are unchanged.
This run does not generate worker text reports or resample the cohort under
thinking on.

The mask likelihood function's thinking-off restriction is avoided by calling
masked context construction and the shared decoder directly. Every row records
thinking/answer token counts, forced budget closure, EOS repair, answer caps,
and latency. Conditions are interleaved within each question; AB/BA for each
source are adjacent. Smoke answers are kept separately and must contain a
complete boxed answer after `</think>` when thinking is forced.

Primary contrasts are the change in full-minus-blocked and gold-minus-blocked
EM gaps when thinking is enabled, separately for toy and MuSiQue. Their four
paired tests receive a Holm correction. All CIs use 20,000 source-bootstrap
resamples, averaging AB/BA within each source. Only sources completed in all
six conditions and both placements enter paired comparisons. Partial tables
remain explicitly partial. Cell EM/F1 and paired differences are fractions,
not percentages; multiply by 100 for percentage points.

Budgets are matched; **actual compute is not**. Report lengths and latency.
A wide interval is inconclusive, not proof of no thinking effect. Gold access
is an oracle paragraph intervention; it is not an efficient repair method.

## 3. Answer-format-controlled likelihood

```bash
python scripts/run_paper_ablations.py likelihood
```

The exact existing 42 AB and 43 BA toy layouts whose selected distractor is
before all gold paragraphs in A. Each has blocked/pre-gold-distractor/gold-only
attention: 255 context/example evaluations, **no free answer generation**.

Each evaluation scores the correct mayor and the two incorrect mayors in the
supporting cities paragraph, with that candidate set frozen across conditions.
It preserves the canonical original `\boxed{answer}` tokenization, refuses a
boundary-changing prefix, and partitions the teacher-forced token probabilities
into the common `\boxed{` prefix, answer tokens conditional on that prefix, and
closing brace. The correct boxed probability is therefore also recomputed.

Report answer-only log probability, gold-minus-best-incorrect score, and their
length-normalized versions as secondary checks. Token counts and per-token
scores are saved; unequal candidate lengths must not be hidden. Original
archived generation correctness and boxed likelihood are retained as references,
with new-minus-archived likelihood differences to expose runtime drift. The
archive's generated accuracy is not a new generation measurement.

255 evaluations contain three candidate-scoring forwards each (765 total)
after 255 context encodings. The 951 total counts example/condition evaluations,
not low-level forward passes. Control subsets are retrospective; they do not
establish a randomized effect of distractor position.

## Resume, output, and stopping

All commands run a small separate smoke check before main inference. Thinking
smoke outputs must have an uncapped complete boxed answer after thinking closes;
validation cap hits are retained as diagnostics at the archived 256-token budget.
Add
`--smoke-only` to stop after that check. Smoke adds 16 validation generations,
24 thinking generations, or 6 likelihood evaluations, excluded from main counts.

Rerun exactly the same command to resume. Completed rows are skipped; each new
row is flushed and fsynced. A torn final JSON line is preserved in a sidecar and
removed before appending; interior corruption, duplicates, changed samples,
changed code/settings, or changed runtime metadata are refused. A new protocol
needs a new `--run-name`. Missing runtime metadata must be restored from the
original bundle rather than guessed.

The default soft limit is eight hours, checked between evaluations. Increase
it with `--max-hours 12` without changing the experiment. SIGTERM/Ctrl-C request
a checkpoint after the current evaluation. Exit 2 means incomplete; exit 1
means an error; exit 0 means the requested phase finished. No GPU keep-alive
workloads or automatic VM shutdowns are installed.

Outputs are under `results/paper_v1/<job>/`: frozen `manifest.json`, resolved
`runtime.json`, `smoke/`, `main/rows.jsonl`, `main/cells.csv`, `main/summary.json`,
and `status.json`. The manifest fingerprints generation code and settings;
the runtime freezes resolved model revision, tokenizer, libraries, GPU, and
attention backend. Each job is bundled next to its directory as `<job>.tar.gz`
on completion, graceful interruption, or an inference error.

Rebuild a saved summary without loading model weights:

```bash
python scripts/run_paper_ablations.py thinking --summarize
```

The existing 4B B1/K0/T1 thinking pilot remains a separate experiment. These
runners neither resume nor overwrite it. Text-report controls are also separate.

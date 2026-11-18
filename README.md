# latent-dispatch

CS 5787 Deep Learning (Cornell Tech, Fall 2026) final project.
Josh Huang, Elliott Kalt, Ajit Mallavarapu.

Research journal (what was run, what was decided, dead ends, open questions): [`JOURNAL.md`](JOURNAL.md). People and agents reviewing or extending this repo: read its newest entry first.

## The question

An orchestrator splits a multi-hop question's paragraphs across parallel workers, which are copies of one model. Each worker reads only its share. The orchestrator then answers from what the workers hand back: here, their key-value caches, joined into one context.

**When the two facts a question needs sit in different workers, what is lost, and what would a worker have had to see to keep it?**

We answer it with attention masks. Two workers, A and B, each hold 10 paragraphs of a 2-hop question (one supporting paragraph each, 18 distractors). The model reads `[question, A's block, B's block]` and answers. A mask changes one thing: which of A's tokens B's tokens may attend to while B is being read. The answer step always attends to everything.

## What we found

Fresh-source run, 2026-11-17 and 11-18: 200 MuSiQue 2-hop questions fixed before any output, each in both orders (AB, BA), thinking off, three models. Exact match / F1:

| | Qwen3-8B | Llama 3.1 8B | Qwen3-4B |
|---|---|---|---|
| joint reading | 40.2 / 49.5 | 35.2 / 43.4 | 31.8 / 39.8 |
| full mask (B may attend to all of A) | 39.8 / 48.8 | 35.5 / 42.8 | 31.8 / 39.7 |
| gold (B may attend to A's supporting paragraph only) | 40.0 / 49.8 | 36.0 / 44.2 | 34.8 / 43.3 |
| equal-token window (as many other tokens of A) | 34.0 / 42.7 | 29.5 / 37.1 | 25.2 / 32.0 |
| blocked (B may attend to none of A) | 34.0 / 42.3 | 31.0 / 38.5 | 26.8 / 33.5 |
| K0 (blocks encoded separately, caches joined) | 34.5 / 42.9 | 30.5 / 37.9 | 29.0 / 35.5 |

Paired F1 differences over the 200 questions, 95% bootstrap intervals, pooled over the three models unless said:

- **Reading separately costs about 6 F1 points, and one paragraph accounts for it.** Gold − blocked is +7.6 (5.1, 10.2). Letting B read A's supporting paragraph recovers everything that full access recovers.
- **It is specific.** The same number of other tokens recovers nothing: window − blocked is −0.8 (−2.6, 1.0).
- **It is not position handling.** Separately encoded caches score the same as the blocked mask (blocked − K0: −0.7 (−2.5, 1.2)). Under the blocked mask, swapping A's text for other text leaves B's keys and values bit-identical at every layer.
- **It does not depend on the paragraph sitting next to B.** With A's supporting paragraph moved to the front of A and the control window placed next to B, gold − blocked is +9.6 (6.4, 12.9) and the window still adds nothing (two models).
- **It appears only when a fact has to cross between workers.** With both supporting paragraphs in the same worker, B's access to A is worth nothing: full − blocked is −0.8 (−3.3, 1.7), against +5.4 (2.0, 8.8) on the same questions with one in each worker (two models).
- **Gold against full access is small and depends on the layout.** +2.0 (0.1, 3.8) with the supporting paragraph last in A, −2.1 (−4.4, 0.2) with it first.
- **An earlier set of 79 questions, kept because the model failed their shortcut probes, showed the same direction at three times the size** (gold − blocked +23). Quote the fresh-source sizes.

Every number above and in the 2026-11-18 journal entries is rebuilt from the saved rows, with no GPU, by:

```
python scripts/paper_tables.py
```

## Words used in this repo

| Word | Meaning |
|---|---|
| worker A, worker B | The two halves of the context, read in that order. Both follow a shared prefix S (system prompt and question), so both know the question. |
| supporting paragraph | One of the two paragraphs a 2-hop question needs. "Gold" in file and condition names. |
| AB, BA, AA, BB | Where the two supporting paragraphs sit. AB: hop 1 in A, hop 2 in B. BA: the reverse. AA and BB: both in one worker. |
| joint reading, `b1` | The whole context read in one pass. The reference. |
| blocked, `b1_mask` | B is read without access to A. Positions are the true ones. |
| gold, full, distractor, except-gold, `matched_irrelevant`, `matched_late` | Mask conditions: what of A's block B may attend to. The last two are controls with as many tokens as the supporting paragraph, taken from other text of A before it or nearest B. |
| K0 | Each block encoded on its own at local positions, keys rotated to their true positions, caches joined. What a multi-worker system would do. |
| K1 | K0 plus m latent "thinking" steps per worker. |
| T1 | Each worker writes a text report and the orchestrator answers from the reports. |
| scan, repair | The orchestrator reads one worker's cache at a time and keeps a short note between reads, for 1 or 2 rounds. |
| B0 | The question alone. Closed-book floor. |
| probe cohort | Questions on which the closed-book probe and both single-supporting-paragraph probes fail (F1 below 0.5) for that model. A stricter set, reported next to all questions. |
| toy | Synthetic two-fact chains ("who is the mayor of X's favorite city") with look-alike paragraphs. |

## Index: each experiment, its code and its rows

One row per experiment. "Code" is what ran it. Where the code is not on `main`, the tag marks the exact commit the rows are stamped with. "Role" is how the result is used as of 2026-11-18; update this table when a run is added or its role changes.

| Experiment | Models | Code | Rows | Journal | Role |
|---|---|---|---|---|---|
| Fresh-source masks: six conditions, 200 new questions | Qwen3-8B, Llama 3.1 8B | `scripts/run_wmac_gpu.py a100` (guide: `scripts/README_wmac_a100.md`, plan: `experiments/wmac_gpu_v3.json`); tag `run/2026-11-17-fresh-sources` | `archive/wmac_gpu_v3/a100-no-logits-2026-11-17.tar.gz` | 11-18 | main result |
| Same, third model | Qwen3-4B | same code plus a memory-saving prefill, tag `run/2026-11-18-fresh-sources-4b` (branch `codex/wmac-l4-v3`) | `archive/wmac_gpu_v3/l4-4b-no-logits-2026-11-18.tar.gz` | 11-18, later | main result |
| Supporting paragraph first: position run | Qwen3-8B, Llama 3.1 8B | `colab_wmac_checks.ipynb` | `archive/wmac_gpu_v3/a100-checks-2026-11-18.tar.gz`, `<model>/position/` | 11-18 | robustness |
| Control layout: both supporting paragraphs in one worker (AA, BB), blocked and full | Qwen3-8B, Llama 3.1 8B | `colab_wmac_together.ipynb` | `archive/wmac_gpu_v3/a100-together-2026-11-18.tar.gz` | 11-18, afternoon | control |
| Mask checks: swap test, kernel comparison | Qwen3-8B, Llama 3.1 8B | `colab_wmac_checks.ipynb` | same tarball, `<model>/verify/` | 11-18 | implementation check |
| Endpoint validation on 24 picked layouts | all three | `scripts/run_wmac_gpu.py` (`validation` jobs) | inside the two no-logits tarballs | 11-18 | implementation check |
| Toy, new layout, thinking off and forced | Qwen3-8B | `scripts/run_wmac_gpu.py a100` (`toy-8b`) | `archive/wmac_gpu_v3/a100-no-logits-2026-11-17.tar.gz` | 11-18 | toy result |
| Masks on 79 picked questions and the toy | Qwen3-8B | `notebooks/h1_masking.ipynb` at tag `run/2026-11-15-masking`, launcher `colab_mask_8b.ipynb`; executed notebook inside the tarball | `archive/Qwen3-8B-bfloat16-masking-2026-11-15.tar.gz`; labels in `archive/screening-2026-11-16/` | 11-15 night, 11-16 evening | earlier cohort; same direction, size not reproduced |
| Same masks, smaller model | Qwen3-4B | same notebook | branch `results/t4-masking-2026-11-15` | 11-15 night, 11-16 evening | earlier cohort |
| Masked thinking, answer likelihood, endpoint validation | Qwen3-8B | `scripts/run_paper_ablations.py` (guide: `scripts/README_paper_ablations.md`, plan: `experiments/paper_ablations_v1.json`); tag `run/2026-11-16-ablations` | `archive/Qwen3-8B-bfloat16-paper_v1-{thinking,likelihood,validation}-2026-11-16.tar.gz` | 11-16 (three entries) | exploratory; 25 questions |
| Thinking with joint reading, K0 and text reports | Qwen3-4B | branch `codex/results-t4-thinking-25-2026-11-16` | same branch | 11-16, later still | exploratory |
| Joint reading against K0 and the block mask, by model size | Qwen3-1.7B, 4B, 8B | `notebooks/part0_ruler.ipynb`, `notebooks/h1_encoding.ipynb` | `archive/Qwen3-4B-bfloat16-2026-11-13.tar.gz`, `Qwen3-1.7B-bfloat16-2026-11-14.tar.gz`, `Qwen3-8B-bfloat16-2026-11-14.tar.gz` | 11-13, 11-14 (two entries) | background: where the loss is and how it scales |
| Latent steps per worker (K1) | Qwen3-4B | `notebooks/h2_thinking.ipynb` | `archive/Qwen3-4B-bfloat16-2026-11-13.tar.gz` | 11-13 | background |
| Text reports (T1) and sequential repair | Qwen3-4B, 8B | `notebooks/h5_rounds.ipynb` (repair), `notebooks/h2_thinking.ipynb` (its T1 control) | `archive/Qwen3-4B-bfloat16-2026-11-14.tar.gz`, `Qwen3-8B-bfloat16-2026-11-15.tar.gz` | 11-14, 11-15 | background |

`notebooks/h3_compression.ipynb`, `notebooks/h4_composition.ipynb` and `notebooks/h7_content_use.ipynb` were built for the original plan and have no reported run.

## Reproducing

**The tables, from saved rows (no GPU).** `python scripts/paper_tables.py` prints the main, position, control-layout, mask-check, validation and toy tables. `pytest tests/test_paper_tables.py` checks them against the summaries the runner saved at run time.

**A rerun (GPU).**
- Fresh-source protocol: `scripts/README_wmac_a100.md`. `python scripts/run_wmac_a100.py prepare` rebuilds the frozen question bundle; `python scripts/run_wmac_gpu.py a100` runs every job. `--plan-only` previews a job without loading a model.
- Position run and mask checks: open `colab_wmac_checks.ipynb` on an A100.
- Control layout: open `colab_wmac_together.ipynb` on an A100.
- Ablations: `scripts/README_paper_ablations.md`.
- Earlier notebook experiments: `colab.ipynb` clones this repo at a commit, builds the benchmarks, runs the chosen notebooks as a smoke and then in full. `LOAD` starts a run from a tarball in `archive/`.

**The exact code of a run.** Every saved row records the commit that wrote it. Four of those commits are tagged `run/<date>-<what>` (`git tag -l 'run/*'`).

## Layout

```
latent_dispatch/   shared code, plain functions
  arms.py            every arm as one function: b0 b1 b1_mask k0 k1 t1 scan
  masking.py         the attention masks: what of A's block B may attend to
  prompts.py         the one prompt layout every arm uses, with each family's chat markup
  model.py           the model matrix and architecture check; explicit-position forwards, greedy decode
  kv.py              copy / slice / concat / RoPE key shift
  data.py, toy.py    benchmark questions, placements, the synthetic toy
  results.py         run an arm over questions -> results/<model>/<name>.jsonl; load and re-score
  analysis.py        accuracy tables, paired gaps with intervals
  score.py           answer extraction and normalized exact match / F1 (one scorer for everything)
  stats.py           sign-flip test, McNemar, paired bootstrap, interaction, Holm
  wmac_*.py          the fresh-source protocol: question selection, the experiments, the runner's summary
  paper_*.py         the 11-16 ablations: plan, experiments, summary
scripts/           runners (run_wmac_gpu.py, run_wmac_a100.py, run_paper_ablations.py), their guides, paper_tables.py
experiments/       frozen plans of the scripted runs (JSON)
notebooks/         the earlier experiments, one notebook each: prediction -> questions and arms -> run -> tables
colab*.ipynb       Colab launchers
archive/           saved rows, one tarball per run (see the index)
benchmark/         SHA256SUMS (committed); question files are rebuilt locally and checked against it
tests/             fast: tiny random models of each family and each family's chat template; slow: real weights
JOURNAL.md         research journal
```

## Rules for `archive/`

- **Nothing new at the top level.** The fresh-source planner treats every tarball directly in `archive/` as an earlier run and excludes its questions; the frozen bundle records the nine it saw, and a test holds that set fixed. New runs go in a subfolder.
- **Rows without logit arrays.** The tarballs here hold rows, summaries and the question bundle. Raw first-token and shared-prefix logit arrays are hundreds of megabytes and stay out of `main`. For Qwen3-4B they are on branch `codex/wmac-l4-v3` as `reports/wmac-l4-v3/raw-results.part-000` to `part-006` (`cat raw-results.part-* > raw-results.tar.gz`, sha256 `244f3e1e…509f85`); the tarball in `archive/wmac_gpu_v3/` is that archive minus its `.npz` files.
- Each tarball unpacks into its own folder. The older ones unpack at the repo root into `results/` and `notebooks/runs/`, both ignored by git.

## Setup

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
pytest                # fast tests: no GPU, downloads tokenizers
pytest -m slow        # real Qwen3-0.6B weights end to end, and the real datasets
```

transformers is pinned to 4.57.6. Qwen3 needs 4.51 or newer, and the cache code is tested against this exact version.

Benchmark files are built once and checked against the committed hashes:

```
python -m latent_dispatch.data toy toy_d0 musique hotpotqa
cd benchmark && shasum -a 256 -c SHA256SUMS
```

Rebuilding the tables needs none of this: the questions, layouts and accepted answers of the fresh-source runs are in `prepared.json` inside the two `no-logits` tarballs of `archive/wmac_gpu_v3/`; the other runs there read the same file.

## Datasets and models

- **Questions.** MuSiQue-Ans dev, linear 2-hop chains, for every mask experiment. The earlier notebook runs also used 3-hop and 4-hop MuSiQue chains and HotpotQA bridge questions. The toy is generated.
- **Models run.** Qwen3-1.7B, 4B and 8B, and Llama 3.1 8B Instruct (gated: accept its license on huggingface.co). `model.MODELS` also lists Qwen3-14B, Llama 3.2 3B, Mistral-Nemo and phi-4, which pass the architecture check and were not run.
- **Models that cannot be used.** Qwen3.5 and newer Qwen: most of their layers are linear-attention layers that keep a fixed-size state instead of a per-token cache, so two workers' states cannot be joined.

## Rules for every run

- Models from `model.MODELS` only: full-attention models with plain RoPE, because every arm needs caches that can be joined and shifted. `model.check_architecture` refuses anything else before downloading weights.
- Greedy decoding. The answer format and the thinking mode are forced by prefill, the same way in every condition.
- One scorer for everything: normalized exact match and token F1. Report both.
- Every condition runs on the same questions with the same input tokens; the rows of the scripted runs record a hash of the input and the answer position so this can be checked.
- The unit of analysis is the source question. Its layouts are averaged first, then paired bootstrap intervals over questions.
- Every row records its options, whether a token cap cut it off, and the commit that wrote it.
- Say how each claim was checked, in the journal entry that makes it.

## Borrowed code

Code ported from [JoshBong/LatentMAS](https://github.com/JoshBong/LatentMAS) (a fork of [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS), Apache-2.0) keeps a link and license note at the top of each file, as the course requires. Code taken from any other source gets the same.

## References

- LatentMAS, arXiv 2511.20639
- RecursiveMAS / RecursiveLink, arXiv 2604.25917
- CacheBack, arXiv 2609.32046 (code: maxr0ssi/cacheback)
- Min et al. 2019, "Compositional Questions Do Not Necessitate Multi-hop Reasoning"
- MuSiQue: Trivedi et al. 2022, "MuSiQue: Multihop Questions via Single-hop Question Composition"

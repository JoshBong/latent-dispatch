# latent-dispatch

CS 5787 Deep Learning (Cornell Tech, Fall 2026) final project.
Josh Huang, Elliott Kalt, Ajit Mallavarapu.

Research journal (what is running, what was decided, dead ends, open questions): [`JOURNAL.md`](JOURNAL.md). People and agents reviewing or extending this repo: read its newest entry first.

## The question

An orchestrator model splits a multi-hop question's paragraphs across N parallel workers (copies of the same model). Each worker reads only its share and sends something back: its KV cache, a compressed latent payload, or text. The orchestrator answers.

**When the evidence a question needs is split across workers, at which stage does this lose the answer, and does the loss grow with the number of crossings?**

A *crossing* is one jump in the reasoning chain from one worker's evidence to another's. Hop 1 on worker A and hop 2 on worker B is one crossing. A, B, A is two.

## Pipeline and where each hypothesis sits

```
split paragraphs -> workers encode -> workers think -> send payload -> orchestrator composes -> [re-query rounds]
                        H1                H2               H3                 H4                      H5
H6 = a trained fix for H3      H7 = does the orchestrator use what was sent (checks all of the above)
```

Payloads sit on two axes. They have to be kept apart: in our pilots, lossy text reports beat lossless KV. Text gets re-read jointly in the orchestrator's prompt, while the KV is used as the workers encoded it.

|            | orchestrator re-encodes it jointly | orchestrator uses worker KV as-is |
|------------|------------------------------------|-----------------------------------|
| lossless   | B1                                 | K0, K1                            |
| compressed | T1, T2, L1/L2, K4 re-prefilled     | K3, K4 as KV                      |

## Arms

| Arm | What the orchestrator gets |
|---|---|
| B0 | The question only. Closed-book floor. |
| B1 | All paragraphs as text in one context. Ceiling. |
| B1-mask | B1 with a block-diagonal attention mask (paragraph blocks can't attend to each other) and true positions. |
| K0 | Each worker's paragraph KV with no latent steps, stitched at disjoint positions (keys RoPE-shifted). |
| K1 | K0 plus m latent thinking steps per worker. |
| K2 | K1 with LatentMAS's W_a realignment. Models with tied embeddings (Qwen3-4B and smaller, Llama-3.2-3B) have W_a ≈ I, so K2 ≈ K1 there; run it only as a check. Qwen3-8B/14B, Llama-3.1-8B, Mistral-Nemo and phi-4 don't tie them, so K2 is a real arm on those. |
| K3 | Only the m latent-thought positions. |
| K4 | Top-B positions per worker, chosen by the question's attention (CacheBack-style). Sent as KV, or re-prefilled as tokens. |
| T1 | The worker's text report, capped at B tokens. |
| T2 | The worker's copied supporting sentences, capped at B tokens. |
| L1 / L2 | Trained RecursiveLink: worker hidden states → link → orchestrator input embeddings. L2 shares weights across workers and works for any N. (The framework doc calls these H1/H2; renamed here so they don't collide with the hypotheses.) |

Every worker continues from a shared prefix S (system prompt + question, prefilled once), so every worker knows the question.

Placements: **P-colo** = all gold paragraphs on one worker. **P-split** = gold paragraphs on different workers. **P-rand** = random. **P-plan** = the orchestrator assigns by title.

## Main design: crossings set by placement on the same question

Use N = 2 workers, and set the number of crossings by where the gold paragraphs go:
- A 3-hop MuSiQue question can be placed BBB (0 crossings), AAB (1) or ABA (2).
- A 4-hop question can go up to ABAB (3).
- The 0-crossing control is BB-type: all gold in the second block, which is encoded separately and RoPE-shifted. AA-type placements put all gold in the first block, which K0 encodes exactly like B1 (a causal first block never sees later blocks anyway). Only the distractor block differs, so they serve as a reference, not as the control.

The question, N and the distractors stay fixed, so crossings is the only thing that changes. Distractors pad both workers to the same paragraph count (off by one when the total is odd, as on 9 MuSiQue questions). Token counts are not balanced: on MuSiQue, one worker's block can be 900 tokens and the other's 1,300. Every row logs `worker_tokens`, so analyses can control for it.

The synthetic toy gives exact control over 0/1/2 crossings, with unlimited n. Shard A holds people facts, shard B holds city facts, and both include same-relation distractors. The chain is A→B or A→B→A.

## Hypotheses

**Part 0: the data has to need composition.**
- Per question, run three probes: closed-book; each gold paragraph with the other gold removed (distractors kept); and all paragraphs.
- A question requires composition if closed-book is wrong and every single-paragraph probe is wrong, under every data seed it ran on.
- Report every result on the full set and on that subset.
- Don't filter on B1 being right; that inflates every gap.

**H1 Encoding.** Encoding paragraphs separately costs accuracy only when the chain crosses workers, and the cost grows with crossings, even though the orchestrator holds every paragraph's KV.
- Test: B1 vs B1-mask vs K0, by crossings.
  - B1 − B1-mask = missing cross-paragraph attention.
  - B1-mask − K0 = encoding each block at local positions and shifting the keys afterward.
- Prediction: B1 − K0 ≈ 0 at 0 crossings (BB-type), then grows with crossings.
- Mechanism: recompute a fraction of each block with full attention, bridge-entity tokens vs random tokens.
- Sanity gates before any result:
  - K0 with N = 1 reproduces B1's logits.
  - K0 built through the cache path matches a single forward pass with the same mask and positions.

**H2 Worker thinking.** Latent steps help only when one worker holds both hops.
- Test: K1 at m ∈ {0, 8, 16, 32} × {P-colo, P-split}, plus K3.
- Prediction: a gain under P-colo only, and equivalence to 0 under P-split.
- Controls:
  - T1 under P-colo should come out ≈ B1, which shows pre-composition is possible.
  - The worker answers from its own cache after m steps, to check the latent steps aren't corrupting it.

**H3 Compression.** With a compressed payload, split questions fail because a worker can't tell which of its facts another worker's hop needs.
- Example: the question asks for the nationality of Inception's director.
  - Worker A has "Inception was directed by Christopher Nolan."
  - Worker B has "Nolan is British," but the question never mentions Nolan, so B has no reason to keep it.
- Test: T1, T2, K3, K4-as-KV and K4-reprefilled at budget B ∈ {16, 64, 256}, each against K0, by crossings.
- Retention: measure how many gold facts each payload keeps, split into hop-1 facts (entities in the question) and hop-2+ facts (reachable only through the bridge).
- Matched pairs:
  - K4-as-KV vs K4-reprefilled (same spans, only the encoding differs).
  - K4-reprefilled vs T2.
  - K3 vs T1 at an equal number of positions.

**H4 Composition.** Orchestrator reasoning makes up for separate encoding at 1 crossing, but not at 2.
- Test: {B1, K0} × thinking {off, forced, budget 256, budget 1024} × crossings.
- Thinking is forced by prefilling `<think>` in every arm. Asking for it gets ignored in long prompts, which would make reasoning depend on the channel.
- Diagnostic: does the trace name the gold bridge entity?

**H5 Rounds.** Re-querying the saved worker caches with the orchestrator's partial findings repairs split failures, and the rounds needed ≈ crossings.
- Run on a compressed channel, because under K0 the payload is already lossless.
- Test: rounds ∈ {1, 2, 3} × crossings.
- Oracle arm: the round-2 query is the gold bridge entity.
- Matched-budget control: one round at 2B vs two rounds at B.
- The orchestrator re-reading the K0 caches runs as a separate arm.

**H6 Trained link.** A RecursiveLink trained for dispatch closes the H3 gap.
- Test at equal B: L1/L2 vs a T2 extractor fine-tuned on the same supporting-fact labels (matched params and steps), T2 prompted, and K4.
- If L ≈ fine-tuned T2, the gain comes from the labels, not the latent channel.
- Generalization: train on 2-hop with N = 2, test on 3–4-hop with N = 4.

**H7 Content use.** Answers depend on what the workers sent. When the orchestrator has both text and cache, it follows the text.
- Use counterfactuals, not deletions. Deleting the payload always craters accuracy in dispatch, because it's the only evidence.
- Four tests:
  - Edit the answer entity in the gold paragraph, split by whether closed-book gets the question right.
  - Swap one worker's payload with its payload from another question.
  - Swap K1's thought positions while keeping the paragraph KV.
  - Give the orchestrator conflicting text and KV, and see which one it follows.

## Datasets

Benchmark files are built once, deterministically, and checked against `benchmark/SHA256SUMS` (committed). The files themselves are gitignored; rebuild them with the commands under Running.

| File | Dataset | Questions | Placements | Seeds | Instances |
|---|---|---|---|---|---|
| `toy.jsonl` | `toy_d2`: toy with 2 distractor shards | 500 per hop (hop0–2) | AB, BA, AA, BB | 3 | 18,000 |
| `toy_d0.jsonl` | `toy_d0`: the pilots' toy, no distractors | 500 per hop | AB, BA | 3 | 9,000 |
| `musique.jsonl` | MuSiQue-Ans dev, linear chains only | 1,252 2-hop, 568 3-hop, 246 4-hop | 2-hop AA/BB/AB/BA; 3-hop AAA/BBB/AAB/ABA; 4-hop AAAA/BBBB/AABB/ABBA/ABAB | 3 | 25,530 |
| `hotpotqa.jsonl` | HotpotQA distractor dev, 10-paragraph items | 5,899 bridge, 1,446 comparison | AA, BB, AB | 3 | 66,105 |

- A seed is a data seed: which distractors go to which worker, and the paragraph order inside each worker. On MuSiQue and HotpotQA the 3 seeds are the same source question in 3 layouts; on the toy they are different generated questions.
- Within one seed, every placement of a question uses the same random draw, so AB and BB differ only in where the gold sits (and the one distractor it displaces), not in a fresh shuffle.
- MuSiQue's non-linear types (3hop2, 4hop2, 4hop3) are skipped because their hops branch, so there's no single chain to count crossings on.
- HotpotQA doesn't mark which gold paragraph is hop 1, so chain order is random, drawn once per question (the same under every seed). Use it only after Part 0 has found the questions that need both paragraphs. Comparison questions are scored against the majority-class rate.

## Models

Every arm needs the worker caches to be stitched (concatenated) and shifted to new positions. That only works when every layer keeps a per-token KV cache with plain RoPE, so the matrix is limited to full-attention models. `model.check_architecture` refuses anything else before downloading weights.

| Role | Models | Notes |
|---|---|---|
| Main family | Qwen3-4B, Qwen3-8B, Qwen3-14B | The backbone in LatentMAS and almost every 2026 latent-MAS paper. Milestone runs on 4B. |
| Second family | Llama-3.1-8B-Instruct (Llama-3.2-3B-Instruct for cheap checks) | The second family in KITE, MaSRead and Cache Merging, so our numbers compare with theirs. Gated: accept the license on huggingface.co. |
| Optional third | Mistral-Nemo-Instruct-2407 (12B), phi-4 (14B) | Used by Cheng, Das & Ramnath (2608.04893). |

- **Forced thinking (H4's forced arms) runs on Qwen3 only.** The other families have no thinking mode, and `thinking="forced"` raises on them.
- **K1 on Mistral is slightly off-template.** Mistral's template moves the system prompt to the last user turn. K1 keeps it in S so that every worker shares the prefix. B1, K0, T1 and scan match the template exactly on every family (`tests/test_formats.py`).
- **Check K1 on Llama before relying on it.** LatentMAS latent steps made Llama-3.2-3B degenerate in 2608.04893 (hidden states shrink about 83x at the rescaling step). The slow test `test_k1_latent_steps_do_not_wreck_the_output` catches this.
- **Qwen3.5 and newer Qwen are out.** Qwen3.5-4B has 24 Gated DeltaNet (linear-attention) layers and 8 attention layers. A DeltaNet layer keeps a fixed-size state instead of a per-token cache, so two workers' states can't be concatenated or shifted. HyLaMAS (2610.37779) runs Qwen3.5 only by handing the full state from one agent to the next, which is a chain, not parallel workers. Write this up as a limitation.
- Memory in bf16: 4B ≈ 8 GB, 8B ≈ 16 GB, 12–14B ≈ 24–28 GB (A100 40 GB; too big for an L4).
- Adding a model: put it in `model.MODELS` with its chat format (`prompts.FORMATS`), then run `pytest tests/test_formats.py tests/test_architecture.py` and `MODEL=<repo> pytest -m slow tests/test_real_model.py` on a GPU.

## Rules for every run

- Models from the matrix above, bf16, greedy decoding.
- One scorer for every arm: normalized EM/F1, with number words normalized ("five" = "5").
- The answer format and thinking mode are forced by prefill, the same way in every arm.
- Token budgets come in two kinds, and neither is picked at random:
  - **Caps that must never bind**: the answer (256 tokens), scan scratch notes (1024), the 8192 hard stop. Every row records `cut`, and `analysis.audit` reports per arm how often a cap stopped the output. It should be ~0; if the smoke shows cuts, raise the cap and rerun that arm.
  - **Budgets that are the variable**: T1's report budget (H3), the thinking budget (H4), latent steps m (H2). These are swept over a grid fixed before the run and reported as a curve, never tuned on the results.
- B1's paragraph order matches K0's stitched order.
- Every arm runs on the same questions, with 3 data seeds (`SEEDS` in each notebook's settings cell).
- Statistics:
  - The unit is the source question. Its rows (3 seeds; several placements at one crossing level) are averaged first, because they aren't independent. Then: paired bootstrap CIs over source questions, an exact sign-flip test (equal to exact McNemar when each question has one row), Holm correction across the family.
  - "Grows with crossings" is tested as an interaction.
  - "No effect" claims use TOST at ±2 EM.
- The prediction and decision rule go in the first cell of the experiment's notebook and get committed before the run.

## Milestone (Mon Nov 16)

Part 0 + H1 + H2 on Qwen3-4B, with arms B0, B1, B1-mask, K0, K1 and K3.

| Part | Owner |
|---|---|
| TBD | TBD |

The course requires every member to own a part.

## Prior pilots (to rerun)

These came from Qwen2.5-3B/7B at n = 50–200 and are pilots only.

- Toy, 7B, forced reasoning:
  - 1 crossing: B1 99, K0 85, T1 100.
  - 2 crossings: B1 97, K0 32, T1 49.
- MuSiQue, 7B, n = 200, F1:
  - 2-step: B1 .414, K0 .294, T1 .387.
  - 3-step: B1 .415, K0 .313, T1 .314.
- The orchestrator re-reading the worker caches once per crossing recovers the toy (2 crossings, reversed order: 1 pass 14, 2 passes 62). It doesn't recover MuSiQue.
- HotpotQA bridge questions are mostly answerable from one paragraph: P-split accuracy sits flat at 34–38 across every arm. That's why Part 0 exists.

## Layout

```
latent_dispatch/   shared code, plain functions, imported by every notebook
  arms.py          every arm as one function: b0 b1 b1_mask k0 k1 t1 scan
  prompts.py       the one prompt layout every arm uses (S / blocks / tail), with each family's chat markup
  model.py         the model matrix + architecture check; explicit-position forwards, latent steps, greedy decode
  kv.py            copy / slice / concat / RoPE key shift              (ported from the fork, tested)
  data.py          build / load / filter benchmark questions; placement; MuSiQue + HotpotQA loaders
  toy.py           synthetic toy generator
  results.py       run an arm over questions -> results/<model>/<name>.jsonl; load_results -> scored table
  analysis.py      accuracy tables, paired gaps with CIs, TOST, crossing trend
  score.py         answer extraction + normalized EM/F1 (one scorer for every arm)
  stats.py         sign-flip test, McNemar, paired bootstrap, interaction, TOST, Holm
notebooks/         one per hypothesis: prediction -> questions + arms -> run -> tables and plots -> findings
benchmark/         SHA256SUMS (committed); question files (gitignored, rebuilt locally)
results/           <model>-<dtype>/<arm name>.jsonl, one row per question (gitignored)
tests/             fast: tiny random models of each family (H1 gates) + each family's real chat template;
                   slow: real weights end to end (MODEL=...) + the real datasets
archive/           finished runs, one tarball each (rows + executed notebooks); unpack at the repo root
colab.ipynb        the whole pipeline on one Colab GPU; carries its own copy of the code (see Running)
JOURNAL.md         research journal: what is running, decisions, dead ends, open questions
```

Each notebook reads top to bottom as the experiment. The first cell is the prediction and decision rule, frozen before the run. Then the questions and the arms, a run cell, the tables, and a findings cell to fill in. Notebooks never define prompt, KV or scoring code; they import it, so every experiment uses the same arms.

Built so far: Part 0, H1, H2 (K1 only), H3 (T1 only), H4, H5 (orchestrator scan only), H7 (select-one + notes audit). These are the experiments the pilots already ran. Each notebook's first cell lists what is still missing for that hypothesis.

## Setup

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

transformers is pinned to 4.57.6. Qwen3 needs ≥ 4.51, and the KV code is tested against this exact version. Notebooks use the `.venv` kernel (Jupyter Lab: `.venv/bin/jupyter lab`; VS Code: pick `.venv` as the kernel).

## Running

**On Colab:** open `colab.ipynb` in Colab (its first cell has the link; uploading the file works too), pick an L4 GPU and run all. The VS Code Colab extension's kernel works too, but ours kept disconnecting on Nov 12. The notebook carries its own copy of the code (one commit, as a base64 git bundle in the `BUNDLE_B64` cell), so Colab needs no GitHub access. **It runs that commit, not your checkout**, and every results row records the commit it came from. To repack after changing code: commit, point a branch named `colab-run` at that commit, run `git bundle create latent-dispatch.bundle colab-run`, put that file's base64 into `BUNDLE_B64`, and update the commit named in that cell's comment.

The notebook installs the code, builds and checks the benchmarks, and runs the tests, including the real-weights check for `MODEL` (`RUN_TESTS = False` skips them on a rerun of the same code). It then runs every chosen notebook as a smoke. An arm whose smoke output is broken is left out of the full run. The run stops there if B1 is broken or if a notebook fails during its smoke. In the full run a notebook that fails doesn't stop the next one, and `results/` is packed into a tarball after every notebook. Results stay on the Colab machine, not Drive, so download the tarball before the runtime ends. Settings: `MODEL`, `SEEDS`, `N_PER_CELL` (e.g. 100 for a quick preliminary), `NOTEBOOKS`, `SKIP` (arm names to leave out). One model per run; check its audit and tables before the next model.

The same steps by hand:

1. **Tests.** Both must pass before any run:
   ```
   pytest                # fast: tiny random models of each family (H1 gates) + chat templates (downloads tokenizers)
   pytest -m slow        # real Qwen3-0.6B weights (every arm end to end) + the real datasets
   MODEL=Qwen/Qwen3-4B DTYPE=bfloat16 pytest -m slow tests/test_real_model.py   # each model, before its first run (GPU; DEVICE=cpu by default without CUDA)
   ```
2. **Benchmarks.** Built once, then checked against everyone else's:
   ```
   python -m latent_dispatch.data toy toy_d0 musique hotpotqa
   cd benchmark && shasum -a 256 -c SHA256SUMS
   ```
3. **Smoke.** Open the notebook and run it with `SMOKE = True`: 2 questions per cell, and the raw outputs get printed. Read them before step 4.
4. **Full run.** Set `SMOKE = False` and run all. Pick the model with `MODEL` and the data seeds with `SEEDS`. For long runs, run it headless so the results land in a copy of the notebook:
   ```
   mkdir -p notebooks/runs
   papermill notebooks/h1_encoding.ipynb notebooks/runs/h1_encoding.ipynb -p SMOKE False -p MODEL Qwen/Qwen3-8B -y "SEEDS: [0]"
   ```
   Lists go through `-y` (YAML); `-p SEEDS "[0]"` would pass a string and select nothing. Three seeds cost 3x one seed: Part 0 and H3–H7 grow from about 4–11k to 13–33k questions each, so for the milestone `SEEDS: [0]` is the cheap option.
   Runs resume: questions already saved are skipped, so stopping and restarting loses nothing. On Colab, keep the repo (or at least `results/`) on Google Drive: a disconnected runtime wipes its disk, and with it every saved row.

How results are shared and kept clean:
- Each arm name (`k0`, `k1_m8`, `t1_256`, …) is one file in `results/<model>-<dtype>/`. Notebooks that ask for the same name share its rows, so H3 doesn't rerun H1's K0.
- Each row stores the arm's options and the git commit. Asking for a name with different options stops with an error instead of mixing.
- If you change code in `latent_dispatch/` that changes what an arm generates, delete that arm's results file, or give the new version a new name.
- Scoring changes need no rerun: `load_results` re-scores every row from its raw text.

## Borrowed code

Code ported from [JoshBong/LatentMAS](https://github.com/JoshBong/LatentMAS) (a fork of [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS), Apache-2.0) keeps a link and license note at the top of each file, as the course requires. Code taken from any other source, such as CacheBack or RecursiveMAS, gets the same.

## References

- LatentMAS, arXiv 2511.20639
- RecursiveMAS / RecursiveLink, arXiv 2604.25917
- CacheBack, arXiv 2609.32046 (code: maxr0ssi/cacheback)
- Min et al. 2019, "Compositional Questions Do Not Necessitate Multi-hop Reasoning"

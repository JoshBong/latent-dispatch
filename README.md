# latent-dispatch

CS 5787 Deep Learning (Cornell Tech, Fall 2026) final project.
Josh Huang, Elliott Kalt, Ajit Mallavarapu.

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
| K2 | K1 with LatentMAS's W_a realignment. Qwen3-4B has tied embeddings, so W_a ≈ I and K2 ≈ K1; run only as a check. |
| K3 | Only the m latent-thought positions. |
| K4 | Top-B positions per worker, chosen by the question's attention (CacheBack-style). Sent as KV, or re-prefilled as tokens. |
| T1 | The worker's text report, capped at B tokens. |
| T2 | The worker's copied supporting sentences, capped at B tokens. |
| L1 / L2 | Trained RecursiveLink: worker hidden states → link → orchestrator input embeddings. L2 shares weights across workers and works for any N. (The framework doc calls these H1/H2; renamed here so they don't collide with the hypotheses.) |

Every worker continues from a shared prefix S (system prompt + question, prefilled once), so every worker knows the question.

Placements: **P-colo** = all gold paragraphs on one worker. **P-split** = gold paragraphs on different workers. **P-rand** = random. **P-plan** = the orchestrator assigns by title.

## Main design: crossings set by placement on the same question

Use N = 2 workers, and set the number of crossings by where the gold paragraphs go:
- A 3-hop MuSiQue question can be placed AAA (0 crossings), AAB (1) or ABA (2).
- A 4-hop question can go up to ABAB (3).

The question, N and the distractors stay fixed, so crossings is the only thing that changes. Distractors pad both workers to the same paragraph and token count.

The synthetic toy gives exact control over 0/1/2 crossings, with unlimited n. Shard A holds people facts, shard B holds city facts, and both include same-relation distractors. The chain is A→B or A→B→A.

## Hypotheses

**Part 0: the data has to need composition.**
- Per question, run three probes: closed-book, each gold paragraph alone, and all gold paragraphs.
- A question requires composition if closed-book is wrong and every single-paragraph probe is wrong.
- Report every result on the full set and on that subset.
- Don't filter on B1 being right; that inflates every gap.

**H1 Encoding.** Encoding paragraphs separately costs accuracy only when the chain crosses workers, and the cost grows with crossings, even though the orchestrator holds every paragraph's KV.
- Test: B1 vs B1-mask vs K0, by crossings.
  - B1 − B1-mask = missing cross-paragraph attention.
  - B1-mask − K0 = encoding each block at local positions and shifting the keys afterward.
- Prediction: B1 − K0 ≈ 0 at 0 crossings, then grows with crossings.
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

- **Toy dial**: synthetic, 0/1/2 crossings.
- **MuSiQue-Ans dev**: 2-, 3- and 4-hop questions.
- **HotpotQA distractor**: only the subset that fails the single-paragraph probe. Comparison questions are scored against the majority-class rate.

## Rules for every run

- Qwen3-4B, bf16, greedy decoding.
- One scorer for every arm: normalized EM/F1, with number words normalized ("five" = "5").
- The answer format and thinking mode are forced by prefill, the same way in every arm.
- B1's paragraph order matches K0's stitched order.
- Every arm runs on the same questions, with 3 placement seeds.
- Statistics:
  - McNemar per pair, paired bootstrap CIs, Holm correction across the family.
  - "Grows with crossings" is tested as an interaction.
  - "No effect" claims use TOST at ±2 EM.
- The prediction and decision rule go in a run's config before the run.

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
src/latent_dispatch/   cache ops, channels, scorer
configs/               one config per run, including its prediction
experiments/           one entry point per hypothesis
tests/                 unit tests against real model output
results/               summaries in git; raw generations stay out (see .gitignore)
docs/                  AI-assistance log, notes
```

## Setup

TBD. Known constraints:
- Qwen3 needs transformers ≥ 4.51.
- On 5.14 the cache-expand op copies instead of returning a view, and the old fork's test fails.
- So pin a version once the ported tests pass.

## Borrowed code

Code ported from [JoshBong/LatentMAS](https://github.com/JoshBong/LatentMAS) (a fork of [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS), Apache-2.0) keeps a link and license note at the top of each file, as the course requires. Code taken from any other source, such as CacheBack or RecursiveMAS, gets the same.

## AI assistance

The course requires exact acknowledgment of coding assistance (how and what) in the report appendix, and GenAI is not allowed for writing the report itself. Log every use in [docs/AI_ASSISTANCE.md](docs/AI_ASSISTANCE.md) as it happens.

## References

- LatentMAS, arXiv 2511.20639
- RecursiveMAS / RecursiveLink, arXiv 2604.25917
- CacheBack, arXiv 2609.32046 (code: maxr0ssi/cacheback)
- Min et al. 2019, "Compositional Questions Do Not Necessitate Multi-hop Reasoning"

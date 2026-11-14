# Research journal

For the people and agents working on this repo. `README.md` holds the design: the question, the arms, the hypotheses and the rules for every run. This file holds what has happened since: what is running, what was decided, which ideas are dead and why, and what is still open. Read the newest entry before proposing or reviewing an experiment.

Newest entry first. Add a dated entry when something is decided or ruled out. Don't rewrite old entries; correct them in a new one. Say how each claim was checked: a run, a test, a paper read, or only an argument.

## 2026-11-13: masking follow-up prepared for Colab

`colab_masking.ipynb` is a preset for the existing runner: preload the 11-13 archive, then run Qwen3-4B/bf16 on the full benchmark with seeds `[0, 1, 2]`, completing Part 0/H1/H2 and `h1_masking`. The smaller original seed-0/100-per-cell scope is available in the first cell. No new experiment was launched during this setup.

The preset stores its checkout, results, and executed notebooks on Google Drive for resuming across runtimes. Before appending results, the writer recovers a missing final newline or removes only an interrupted final fragment so that question can rerun.

The five A-side visibility conditions are blocked, gold-only, one distractor, all except gold, and full. Tokens and true positions stay fixed and the answer tail sees both blocks. Full runs filter by Part 0 composition labels, save EM and gold-answer likelihood, record selections and coverage, and compare endpoints against saved B1/B1-mask outputs. The full masking scope has 21,024 candidate question/layout rows before the composition filter; the final count depends on the completed probes. Smoke uses the runner's fixed unfiltered questions and now writes its Part 0 labels separately from full labels. See `MASKING_RUN.md` for the launch link and outputs.

The core mask endpoints and likelihood code passed tiny-model and Qwen3-0.6B checks during initial implementation. This packaging step performs static validation only; the Colab preset retains the normal model tests and smoke gates for Josh's run. Bridge-span interventions in B remain a later stage after observing the A-side results.

## 2026-11-13, later: the runner loads earlier runs and takes its code from GitHub

- `colab.ipynb` no longer carries the code inside itself. It clones this repo at `COMMIT` and prints the commit it runs. New code runs as soon as it is on GitHub, and nothing has to be repacked.
- `LOAD = ["Qwen3-4B-bfloat16-2026-11-13"]` starts a run from that tarball in `archive/`, so the questions it answered aren't run again. The tarball is checked first: the row format, and that its rows were made with the same benchmark files and prompts as the code about to run.
- The runner issues listed in the 11-12 entry are fixed. The gate judges only the smoke questions, so its verdict is the same on a first run, a rerun, and a run that starts from loaded results. A left-out arm's smoke answers are moved out of the tables' way. A notebook that fails in its smoke no longer stops the others. The commit is printed, and a failed notebook shows its real error. `GATE_CUT = 101` is no longer needed.
- Checked: fast tests for the loader, the same-inputs check and the gate. On the 11-13 rows the smoke questions are identified exactly (the same count per arm as that run's smoke audit) and no arm is judged broken. The runner's cells were dry-run on a laptop with the notebooks stubbed out. Not checked: a full run on Colab with this runner, so watch the first one through its smoke.

## 2026-11-13: first results (Qwen3-4B, preliminary)

The 11-12 run finished: 14.5 hours on an L4, 13,700 rows, every row on commit `621c3fa`, no arm left out, no notebook failed. The rows and the executed notebooks are in `archive/Qwen3-4B-bfloat16-2026-11-13.tar.gz`. Run `tar xzf archive/Qwen3-4B-bfloat16-2026-11-13.tar.gz` at the repo root: it fills `results/Qwen3-4B-bfloat16/` and `notebooks/runs/` (both ignored by git), where the notebooks' load cells expect them.

Preliminary means: one seed, 100 source questions per cell, one model, thinking off. Numbers are EM. One cell's 95% interval is about ±7 points, and nothing is corrected across the many cells below. Checked by rebuilding the tables from the saved rows with the functions in `analysis`; they match what the notebooks printed. The numbers on "questions that require composition" keep only the source questions that `part0_requires_composition.json` marks true and then use the same `analysis.paired`. No notebook does that yet (known issue 1), so those were computed by hand.

### Part 0: what the benchmark measures for this model

| | closed book | B1 | best single gold paragraph | questions that require composition |
|---|---|---|---|---|
| MuSiQue 2-hop | 0 | 29 | 9 | 85% |
| MuSiQue 3-hop | 1 | 41 | 32 | 64% |
| MuSiQue 4-hop | 2 | 18 | 15 | 81% |
| HotpotQA bridge | 6 | 55 | 34 | 50% |
| HotpotQA comparison | 48 | 57 | 57 | 32% |
| toy, 2 and 3 chained facts | 0 | 100, 98 | 0 | 100% |

A question requires composition if closed book and every single-gold-paragraph probe got it wrong.

On the MuSiQue questions that require composition, B1 scores 14–27 at 2-hop, 14–20 at 3-hop and 7–10 at 4-hop. On the rest it scores 80–89 (58–79 at 4-hop). Most of B1's MuSiQue score comes from questions that one paragraph already answers. At 3 and 4 hops this model can barely chain facts even with everything in one context. (The label selects on the model's own failures, so part of that low score is selection.)

### H1: what separate encoding costs

Toy with 2 distractor sets, B1 / B1-mask / K0:

| | AA | AB | BA | BB |
|---|---|---|---|---|
| one fact | 100 / 100 / 97 | 100 / 88 / 89 | 100 / 91 / 88 | 100 / 100 / 100 |
| two chained facts | 100 / 93 / 93 | 100 / 37 / 27 | 100 / 29 / 18 | 100 / 93 / 92 |
| three chained facts | 99 / 97 / 97 | 98 / 22 / 19 | 100 / 24 / 18 | 100 / 97 / 97 |

MuSiQue:

| | B1 / B1-mask / K0 | B1 − B1-mask (95% interval) |
|---|---|---|
| 2-hop AA | 35 / 35 / 33 | 0 (−6, 6) |
| 2-hop AB | 29 / 22 / 18 | +7 (−1, 15) |
| 2-hop BA | 35 / 20 / 17 | +15 (8, 23) |
| 2-hop BB | 24 / 29 / 32 | −5 (−12, 1) |
| 3-hop, four placements | B1 41–44 | +1, +1, −1, +5 |
| 4-hop, five placements | B1 18–22 | +2, −7, −3, +6, −1 |

- **The loss is the missing attention between blocks, not the local positions.** B1-mask − K0 is within ±7 points everywhere except the toy's two-fact questions (+10, +11).
- **MuSiQue shows a cost only at 2-hop, with the two facts on different workers.** AB and BA pooled: +11 (5, 17.5). AA and BB pooled: −2.5 (−7, 2). 3-hop and 4-hop are flat because B1 has almost nothing to lose there (Part 0).
- **The crossing trend doesn't appear on MuSiQue.** B1 − K0 over the BB baseline is 7.5, 3.0 and −7.0 at 1, 2 and 3 crossings. Crossings and hop count are confounded: 3 crossings exists only at 4-hop.
- **Chain direction matters at 2-hop.** On the 85 questions that require composition, B1 − B1-mask is +16.5 (9, 25) at BA and +3.5 (−5, 12) at AB. The difference is +12.9 (3.5, 22.4). In BA the second hop's paragraph comes first in the sequence.
- **B1 is not a clean ceiling.** On the same 100 questions, B1 scores 11 points lower when the gold paragraphs sit in the second block than in the first (4, 19). At BB, K0 beats B1 by 8 (2, 15; not significant after correction). Encoding the gold block without the distractor block in view seems to help.
- One-fact toy questions lose about 10 points at AB and BA. Asked "Who is the mayor of Tarnova?", B1-mask and K0 answer the person whose favorite city is Tarnova (worker A's block) instead of the mayor (worker B's block). So about 10 of the 63–82 points lost on chained questions aren't about chaining.

How the failures look on the toy at AB and BA, from a keyword count over the saved answers (not hand-labelled): under B1-mask and K0, 93–100% of wrong answers name the bridge entity in their text, and 50–70% of all answers then say the next fact isn't in the material. It is, in the other worker's block. Typical: "The mayor of Dev Anand's favorite city (Kastrel) is not mentioned in the provided material."

Reading (an argument, not a test): the model does the first lookup at the answer position and fails the second, even with the bridge written out. Under B1 it never needs the second lookup, because the fact that comes later in the sequence was encoded with the earlier fact and the question in view. That is "chaining at the evidence" from the 11-12 entry, and the answer-position path looks weak in this model.

### H2: K1, each worker adds m latent steps

| | B1 | m = 0 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| toy AB | 100 | 15 | 44 | 41 | 37 |
| toy BB | 100 | 90 | 91 | 88 | 90 |
| MuSiQue 2-hop AB | 29 | 17 | 15 | 14 | 15 |
| MuSiQue 2-hop BB | 24 | 28 | 30 | 31 | 30 |

Latent steps help on the toy (+29 at m = 8; 18, 40), with no gain from more steps and still far below B1. Nothing on MuSiQue. The T1 text-report control wasn't run, so there is no text baseline yet.

### What this changes

- The gate for the mask experiment (11-12 entry, step 1) passes on the toy and on 2-hop MuSiQue. The reading above predicts that letting block B see only A's gold paragraph recovers most of the toy loss, and that a distractor paragraph doesn't.
- The 4-hop cells the proposal used have nothing to decompose for this model with thinking off.
- 3-hop and 4-hop MuSiQue can't test H1 until B1 can answer them: a bigger model, thinking on (known issue 6 first), or a claim limited to 2-hop and the toy. Not decided.
- MuSiQue tables should be reported on the questions that require composition (known issue 1).
- "B1 − K0 at BB" isn't a zero-cost control, because B1 depends on where the gold paragraphs sit. The design needs a line on this.

Output quality: 97–98% of answers are boxed in B1, B1-mask and K0. Answers cut at the 256-token cap: B1 1.6%, B1-mask 1.8%, K0 3.2% (almost all scored wrong), and 7–9% in every arm at 4-hop.

## 2026-11-12: first Qwen3-4B run, and a direction for the H1 mechanism

### What is running

- A preliminary run on Colab (L4): Qwen3-4B, bfloat16, seed 0, 100 questions per cell, Part 0 + H1 + H2. H2 runs K1 only; its T1 control is skipped.
- Code: commit `621c3fa`. Every results row records its commit. The tests passed on this code on the Colab machine earlier in the day.
- The run is driven by a newer `colab.ipynb` than the one in `621c3fa`. It carries the code inside itself, and its smoke gate only leaves out arms whose output is broken. Its review found the runner issues listed below.
- No results yet. The smoke pass (2 questions per cell; 62 questions for H1, all datasets pooled) gave B1 69, B1-mask 50, K0 47 EM. That checks that outputs are usable. It is not a result, so don't cite it.

### Known issues in `621c3fa`

Found in review, not fixed yet. None changes the answers the run is generating. 1–4 are analysis only and get rerun on the saved rows.

1. Part 0 writes the list of questions that require composition (`part0_requires_composition.json`), but no later notebook reads it.
2. The toy's single-shard leak check uses a 5% threshold. It should be compared with the guess rate, about 1/3.
3. `analysis.crossing_trend` tests gap(c) − gap(0) at each crossing level, not a trend across levels.
4. `stats.tost` and `analysis.equivalent`: the bootstrap interval collapses when the sample is tiny or unanimous.
5. `results.run` appends after a line that a killed run cut off, so the next row merges with it. That loses one row, which reruns on resume.
6. `model.greedy` stops at end-of-sequence inside forced thinking, and the scorer can then read `\boxed` out of the thinking text. Only forced-thinking arms are affected (H4, not run yet).

Smaller: the audit's boxed % should count only text after `</think>`, and default options aren't stored in each row.

In the Colab runner as it ran (reviewed 11-12, not fixed yet; none of these fired in this run, which was a first pass on a fresh machine with no arm left out):

- The gate judges every saved row, not only the smoke's. On a second "Run all" on the same machine it can leave an arm out mid-run if more than a fifth of that arm's saved answers hit the token cap. Until this is fixed, set `GATE_CUT = 101` before rerunning.
- An arm the gate leaves out keeps its smoke rows, and the notebooks' tables still load them.
- The notebook doesn't show or check which commit it runs: Colab hides the output of its shell commands, and an existing `/content/latent-dispatch` is reused without comparing it to the packed code. The commit in each results row is the record.
- Smaller: a notebook that fails during its smoke stops the whole run; packing sits outside the error handling; results files left by other sessions are judged too; a failed notebook's message is cut off before the real error.

### Direction: which attention does separate encoding lose?

H1 splits the cost of separate encoding into B1 − B1-mask (attention between workers' blocks is missing) and B1-mask − K0 (blocks were encoded at local positions). This is about the first part.

Terms. Worker A's block always comes before worker B's in the orchestrator's sequence. A *bridge entity* is an intermediate answer in the chain: in "who is the mayor of Ines Farrow's favorite city?", it is the city. B1-mask ("blocked") is B1's tokens at B1's positions with attention between worker blocks cut during encoding. The answer prompt attends to everything in every condition.

Why this gap matters. A model can chain two in-context facts in two places:

- at the evidence: when it reads B's mention of the bridge entity, it links that mention to the one in A;
- at the answer position: look up hop 1, then hop 2, layer after layer.

The blocked mask removes only the first. A's tokens are the same in B1 and B1-mask, because a causal first block never sees later blocks. So the whole gap sits in B's token representations.

The proposal (from Ajit's design review, 11-12; not built). Recompute the full sequence from its tokens under four masks: blocked; bridge-related spans in B may also attend to A; an equal number of position-matched random spans in B may attend to A; full attention. Blocked must reproduce B1-mask and full must reproduce B1. If bridge access recovers more than random access, the loss runs through a few bridge mentions.

What we would change, in order:

1. **Gate on H1.** Read B1 − B1-mask by placement first, and go only if there is a gap at crossing placements (AB, BA). AA-type and BB-type show how much of a gap is not about the chain. AA-type is a placebo here: B holds no gold, so the mask only changes distractor tokens.
2. **Mask A's side first.** Vary what in A is visible to B: nothing, A's gold paragraph(s) only, one distractor paragraph only, everything except the gold paragraph(s), everything. Paragraph spans follow from the `gold` field, so there is no string matching and no token budget. This splits the gap into "the bridge fact wasn't visible" and everything else, which is the project's question: what has to cross between workers.
3. **Then B's side** (the proposal above), with A's side fixed to whatever step 2 found sufficient. One caveat: any B token that can see A can pass what it read to later B tokens, so random spans placed before the gold fact can act as a relay. Random beating blocked doesn't refute the bridge story; bridge ≈ random at small budgets would. Match random spans on which side of the gold fact they sit. Nested spans (mention, its sentence, its paragraph) are a more natural dose than token budgets.
4. **Start with 2-hop MuSiQue and the toy.** 2-hop has 1252 source questions, and AB/BA is already a mirrored pair. 4-hop has 246 and sits near the floor with thinking off.
5. **Score the log-probability of the gold answer** as well as EM. Small masks move EM by a point or two.
6. **Use only questions that require composition** (Part 0's list; known issue 1).

A masked recompute is an upper bound on cache repair. Tokens that weren't selected still read the updated states of the ones that were, which replacing selected KV entries in an existing cache (CacheBlend-style) can't give them. If the mask experiment fails, cheap repair is dead. If it works, repair is the follow-up.

### Ruled out

- **"Only forward crossings (hop i on A, hop i+1 on B) can be pre-composed, so mirrored placements aren't order controls."** Claimed in review and retracted the same day. A's tokens never see B, but B's tokens see both, so B's mention of the bridge entity can attend back to A's fact whichever hop each side holds. Crossings stays the dial, and chain direction is a factor to report. How large the direction effect is isn't settled (small in Huang et al., large in Yu et al., below). Our AB vs BA rows measure it.
- **"We already tried re-encoding the cross-worker attention."** We didn't. The September pilot probe (Qwen2.5-3B, in the LatentMAS fork) removed each worker's chat header from its cached block and rotated the cached keys to their stitched positions, with no model rerun, on one question that a single worker could answer. Every repair in the pilots was on the text side: sequential notes, extra passes, a stricter final prompt. Nothing has been tested at the attention level.

### Open

- With thinking off, the model still reasons in its answer text when it can't answer directly. In the smoke pass, answers averaged 40 tokens under B1 and 60 under B1-mask. EM under the mask therefore includes whatever the model recovers by writing the chain out. Scoring the log-probability of an answer forced right after the prompt would measure the latent path alone. Decide which of the two H1's claim is about.
- Whether to add a selector that needs no gold labels next to the oracle bridge spans: query attention (InfoFlow KV, ProphetKV) or KV deviation (CacheBlend).

### Prior work checked

Numbers come from the papers' text, fetched on 11-12. Recheck before citing.

- Parallel Context Windows (Ratner et al., ACL 2023, arXiv 2212.10947). Windows that can't attend to each other hurt HotpotQA bridge questions (27.1 → 24.0 EM on J1-Grande) and help comparison questions.
- Guo et al., "How Do LLMs Perform Two-Hop Reasoning in Context?" (2502.13913). With distractors, pretrained LLMs guess among the candidate answers (target probability 0.82 → 0.20 at five distractors). A trained model does both lookups at the query token.
- Huang et al., "Masking in Multi-hop QA" (ACL 2025, 2505.11754). On MuSiQue, gold documents in chain order vs reversed moves zero-shot Qwen2.5-7B by about 2.5 points. Fine-tuning with bidirectional attention among the documents adds about 5.
- Yu et al., "Repetition of Misordered Context" (NAACL Findings 2025, 2410.07103). The best and worst order of supporting documents differ by 10–26 F1.
- InfoFlow KV (2603.05353). MuSiQue from LongBench: full prefill .372, no recompute .10–.13, CacheBlend .26–.28, recomputing a query-selected 15% of tokens .338. Their no-recompute baseline caches chunks without the question and at local positions, so it isn't comparable to B1-mask. ProphetKV (2602.02579) reports 96–101% of full accuracy at 20%. Neither says which tokens get selected.
- Graph-KV (2506.07334). On a backbone fine-tuned for block attention, HotpotQA: sequential about 77, block-parallel about 71, chunks attending to the top-3 retrieved chunks about 75.
- Liang et al., "When KV Cache Reuse Fails in Multi-Agent Systems" (2601.08343). A judge reading candidates from reused KV caches loses cross-candidate attention, most for later blocks.
- Not found: a training-free study that restores specific cross-document attention edges under a dependency dial. That is from about a dozen searches, not a full sweep.

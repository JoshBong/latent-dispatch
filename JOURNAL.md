# Research journal

For the people and agents working on this repo. `README.md` holds the design: the question, the arms, the hypotheses and the rules for every run. This file holds what has happened since: what is running, what was decided, which ideas are dead and why, and what is still open. Read the newest entry before proposing or reviewing an experiment.

Newest entry first. Add a dated entry when something is decided or ruled out. Don't rewrite old entries; correct them in a new one. Say how each claim was checked: a run, a test, a paper read, or only an argument.

## 2026-11-16, evening: MuSiQue masking tables re-scored and re-screened (F1, stricter composition filter)

Two problems found by reading answers (entries below) were checked against the main masking result, using only saved rows: the 8B run in `archive/Qwen3-8B-bfloat16-masking-2026-11-15.tar.gz` and Elliott's 4B float16 run on branch `results/t4-masking-2026-11-15`. No new inference.

**Scoring.** Exact match marks "Kenton" wrong against "Kenton County". Token F1 is already saved in every row; report it next to exact match from now on. F1 is not a complete fix: it forgives a dropped suffix but does not judge whether an answer is right.

**Screening.** A question is labelled composition-required when the closed-book probe and both single-supporting-paragraph probes fail exact match. Since exact match under-scores, some kept questions are answerable without combining paragraphs. All probe answers of the 94 questions kept by either model were read once and labelled `same_entity`, `valid_alt` or `wrong`, with two per-question flags, `gold_issue` and `single_paragraph_sufficient`. The labels and a reason for each are in `archive/screening-2026-11-16/musique_2hop_probe_verdicts.json`. Counts: at least one non-wrong probe in 24 of 79 questions at 8B and 26 of 87 at 4B; `gold_issue` 28; `single_paragraph_sufficient` 15; 44 of 94 with nothing flagged. The screening is retrospective, done after the results were seen, by one reader, with about ten borderline calls and some judgments from memory. It needs a second reader before it is used as more than a sensitivity check.

MuSiQue 2-hop, AB+BA, paired differences in F1 points. Intervals: 4,000-resample bootstrap, 95%, resampling groups of source questions that share a supporting paragraph.

| Qwen3-8B | as labelled (79) | non-wrong probes dropped (55) | also no flag (37) |
|---|---|---|---|
| gold − blocked | +23 (16, 32) | +29 (19, 38) | +28 (17, 40) |
| gold − distractor | +23 (16, 32) | +29 (19, 39) | +30 (19, 42) |
| gold − full | +8 (3, 14) | +12 (5, 19) | +10 (2, 18) |
| full − blocked | +15 (8, 23) | +17 (8, 26) | +18 (6, 30) |

| Qwen3-4B | as labelled (87) | non-wrong probes dropped (61) | also no flag (43) |
|---|---|---|---|
| gold − blocked | +14 (7, 21) | +18 (8, 27) | +18 (7, 30) |
| gold − distractor | +16 (10, 23) | +22 (14, 31) | +23 (14, 34) |
| gold − full | 0 (−4, 5) | +2 (−5, 8) | +2 (−7, 11) |
| full − blocked | +13 (6, 21) | +16 (7, 26) | +17 (5, 29) |

Mean F1 by condition on the middle set (blocked, distractor, all-but-gold, gold, full): 8B 30, 30, 32, 59, 47; 4B 20, 16, 16, 38, 36. Exact match on the same set: 8B 20, 19, 22, 51, 37; 4B 16, 10, 7, 30, 29. Exact-match differences move the same way (8B gold − blocked +24 as labelled, +31 on both screened sets).

A cruder automatic filter (drop questions whose best probe has F1 of 0.5 or more; 61 questions at 8B, 65 at 4B) leaves the differences where they were: 8B gold − blocked +25 (16, 34), gold − full +10 (3, 18); 4B +15 (7, 23) and +1 (−5, 7).

**Reading.** The earlier tables are not wrong as computed; they are diluted by questions that do not need both paragraphs or have a bad gold answer. Removing those raises every difference by 4 to 7 points. Access to the supporting paragraph beats blocked and distractor-only access at both sizes on every version of the question set. Supporting-paragraph-only access beats full access at 8B on every version. At 4B that difference is unresolved, which is not the same as zero, and the two models were screened on different question sets, so no size trend is claimed.

**Thinking run under F1** (25 MuSiQue sources, AB+BA): blocked 22 off, 48 forced; gold 54, 57; full 43, 57. Gain from thinking: blocked +26 (10, 42), gold +3 (−10, 18), full +14 (−3, 31). Whether thinking helps once the supporting paragraph is visible is unresolved at this sample size.

**Toy layout finding, qualification.** In the toy layouts, the two "Cities" paragraphs are in different workers exactly when the two "People" paragraphs are, so the layout split reported below cannot be attributed to the Cities paragraphs specifically.

## 2026-11-16, later still: reading the answers of the Qwen3-4B thinking pilot

Same exercise on the 1,050 selected answers of the 4B thinking pilot (branch `codex/results-t4-thinking-25-2026-11-16`; joint reading B1, separate caches K0, text reports T1; thinking off or forced at 1,024). "Checked" and "read" as in the entry below.

**1. "Thinking off" is not the same condition for the text-report arm.** With thinking off, the answering model writes visible step-by-step reasoning under B1 and K0 but a bare boxed answer under T1 (checked):

| thinking off | B1 | K0 | T1 |
|---|---|---|---|
| toy: bare boxed answers of 100, median generated tokens | 1, 71 | 5, 81 | 97, 7 |
| MuSiQue: bare boxed answers of 75, median generated tokens | 42, 13 | 37, 25 | 71, 7 |

So the toy gain from thinking is "no reasoning to reasoning" for T1 and "untagged reasoning to tagged reasoning" for K0. The two gains are not comparable as they stand.

**2. Toy text reports fail in the answering model, not in the reports, on two-fact chains.** Both facts are in the reports in 50 of 50 two-fact layouts (read, by a name-proximity check plus four report pairs in full). The 30 failures are abstentions ("Unknown", "Cannot be determined"; checked). Each worker sees half the chain and, despite being told not to answer, writes a verdict that the question cannot be answered; the answering model repeats it (read).

**3. Three-fact text reports stay at 70 with thinking because a worker drops a fact.** In 14 of 50 layouts the report omits the mayor's birthplace, since the worker holding it does not know who the mayor is: 0 of 14 correct there, 35 of 36 when all three facts are reported (read; 2 of the 14 reports read in full).

**4. MuSiQue text reports, 18 to 12 with thinking, is noise plus scoring.** Counting same-entity answers it is 38 to 34 (read, all 50 split layouts). The gold answer string is absent from the reports in 19 of 50 layouts. Some reports state things the paragraphs do not say.

**5. Exact match under-scores MuSiQue here too** (read; no answer was wrongly marked correct in 118 EM-true rows):

| split placements, as scored → same-entity | off | forced |
|---|---|---|
| B1 | 30 → 38 | 38 → 58 |
| K0 | 12 → 14 | 32 → 48 |
| T1 | 18 → 38 | 12 → 34 |

**6. Toy K0 at the cap looks like the 8B blocked failures.** Every K0 failure names the right bridge city (read, all 73 thinking-off failures). At the cap there are no literal loops: in 23 of 28 failures the model re-lists the distractor "Cities" paragraph and insists the fact is absent, in 5 it states the right fact and then retracts it (read, all 35 capped traces). Statements that mix the two workers' same-template paragraphs appear only under K0: 0 of 200 B1 rows, 0 of 200 T1 rows, 5 of 100 K0 off, 15 of 100 K0 forced (read, by pattern match). None of the capped failures looks like steady progress cut short.

**7. The toy "three-fact" task is A, B, A:** the third fact sits in the same paragraph as the first, so it adds a lookup in an already-read paragraph and not a second crossing between workers.

## 2026-11-16, later: reading the answers of the 8B masked-thinking run (corrections to the entry below)

The entry below reported exact match, intervals and cap rates without anyone reading the generated answers. All 600 answers of the thinking run have now been read (most in full, capped traces by head and tail). "Checked" means recomputed from the rows by script; "read" means from reading the texts and not yet confirmed by a second reader.

**1. Exact match under-scores MuSiQue, most with thinking on.** The model writes "Kenton County" in its reasoning and boxes "Kenton", which EM marks wrong (checked on the Kenton and Crockett questions). Recount treating same-entity answers as correct (read, all 300 MuSiQue answers, EM %):

| | blocked | gold | full |
|---|---|---|---|
| thinking off, as scored | 6 | 44 | 30 |
| thinking off, same-entity | 12 | 48 | 36 |
| forced, as scored | 36 | 42 | 42 |
| forced, same-entity | 46 | 60 | 58 |

Gold above full with thinking off survives (12 points). **The claim below that no improvement from reasoning is observed once the supporting paragraph is visible is withdrawn:** gold goes 48 to 60 under the recount. The 4B and 8B masking tables in earlier entries use the same scorer and have not been re-scored; quote them with that caveat until they are.

**2. The thinking cap is hit because of the question, not the condition.** Five MuSiQue questions hit the 1,024-token cap in all six forced runs (checked). In the 59 capped MuSiQue traces there are no repetition loops (read): in 29 the model finds the evidence early and will not accept it, in 22 it has the right paragraph and wavers over the form of the answer, in 8 it never finds the second paragraph. Three of the 25 questions ask where James Watt was educated and share one supporting paragraph that says only that he worked at the University of Glasgow; with thinking off the model asserts it and scores, with thinking on it notices the gap. A 4,096-token budget is therefore not expected to change much, and the "next run" named below is dropped.

**3. The 25-question MuSiQue sample.** One question cannot be answered from its context: "Assomada" appears in no paragraph (checked). Three questions share one hop-2 paragraph and two share another, so there are fewer independent questions than 25. Several gold answers are debatable (read). A fresh sample should screen for both.

**4. Toy: with thinking on, the blocked failures are one layout.** Each toy layout holds two "Cities" paragraphs, one from the question's own facts and one distractor. EM for the blocked condition (checked):

| blocked, 8B | both Cities paragraphs in one worker | Cities paragraphs in different workers |
|---|---|---|
| forced thinking | 100 (n=25) | 44 (n=25) |
| thinking off, full masking run | 64 (n=100) | 65 (n=100) |

Gold and full access remove the effect (96 to 100). In 12 of the 17 capped traces the model finds the bridge city, then lists only the distractor Cities paragraph, refers to "the Cities section" as one section, and re-scans until the cap; a few traces quote a paragraph spliced from both (read). In BA layouts the fact that is not found sits in worker A, whose encoding is the same in every condition. Reading, not a test: two similar paragraphs encoded without seeing each other are treated as one. A positional cause has not been ruled out; the position ids of worker B under blocked against gold and full have not been compared. Thinking's +8 on toy blocked is churn: it fixed 13 failures and broke 9 answers that were right without it (read). Toy scoring is clean: no mis-scored answers in 300 rows (read, with a script check that the prediction equals the last boxed answer).

**5. Full access against gold-only access.** In the 7 layouts where gold-only is right and full is wrong with thinking off, none of full's answers comes from one of worker A's distractor paragraphs, and in 6 of 7 full gives the same wrong answer as blocked (read). So full access fails to help there; the texts do not show A's distractors leaking into answers.

Not yet explained: in the full masking run the distractor condition scores 81 when both Cities paragraphs are in one worker and 61 when they are split (checked, n=100 each).

## 2026-11-16: three paper ablations on Qwen3-8B (masked thinking, likelihood, endpoint validation)

The three runners from PR #2 (`codex/paper-completion-ablations`, every row stamped `116151f`), run on a Colab A100-40GB, bfloat16, sdpa, seed 0. All three finished: validation 96/96, likelihood 255/255, thinking 600/600, no duplicate ids. Archives: `archive/Qwen3-8B-bfloat16-paper_v1-{thinking,likelihood,validation}-2026-11-16.tar.gz`. The validation archive leaves out the 112 saved first-token logit files (32 MB); the rows keep the top-5 trace per token and the summary keeps the per-pair logit comparisons. Numbers below were recomputed from the rows with the repo's `score`, paired by source question, 4,000-resample bootstrap, 95% intervals, unadjusted.

**Masked thinking.** 25 toy two-fact and 25 MuSiQue 2-hop composition-required sources, AB and BA, so 50 layouts per cell. Thinking off or forced with a 1,024-token thinking budget and a 1,024-token answer cap. EM %:

| | blocked | gold | full |
|---|---|---|---|
| toy, thinking off | 64 | 100 | 100 |
| toy, forced | 72 | 98 | 100 |
| MuSiQue, thinking off | 6 | 44 | 30 |
| MuSiQue, forced | 36 | 42 | 42 |

| paired difference | toy | MuSiQue |
|---|---|---|
| full − blocked, off | +36 (24, 48) | +24 (8, 42) |
| full − blocked, forced | +28 (18, 38) | +6 (−6, 18) |
| gold − blocked, off | +36 (24, 48) | +38 (18, 58) |
| gold − blocked, forced | +26 (16, 36) | +6 (−4, 18) |
| gold − full, off | 0 (0, 0) | +14 (4, 26) |
| gold − full, forced | −2 (−6, 0) | 0 (−10, 12) |
| blocked, forced − off | +8 (−10, 24) | +30 (12, 48) |
| gold, forced − off | −2 (−6, 0) | −2 (−16, 12) |
| thinking gain under blocked − gain under gold | +10 (−6, 26) | +32 (14, 50) |
| gold/off − blocked/forced | +28 (18, 38) | +8 (−12, 26) |

Mean generated tokens and seconds per answer on MuSiQue: gold/off 38 tokens, 2.4 s; blocked/forced 744 tokens, 40.8 s. Mean thinking tokens: MuSiQue blocked 681, gold 618, full 607; toy 609, 313, 292.

**The thinking cap confounds every forced number.** Share of forced answers that used the whole 1,024-token thinking budget: toy blocked 34%, gold 4%, full 0%; MuSiQue blocked 44%, gold 36%, full 38%. For blocked + forced, EM is 24 (n=17) when the cap was hit and 97 (n=33) when not on the toy, and 18 (n=22) against 50 (n=28) on MuSiQue. This split is not causal: the capped answers may be the hard questions. It does mean no claim about a gap that survives thinking can be made at this budget. Next run: blocked and full, forced, 4,096-token budget, same sources.

**Likelihood** (toy, 85 questions, three candidates each, thinking off). At AB, distractor − blocked: gold boxed log-probability −5.69 nats (−7.77, −3.68), gold minus best wrong candidate +1.78 (−0.12, 3.63), while archived EM on those 42 questions is 50 against 86. At BA the boxed change is −0.31 (−2.16, 1.69). Gold − blocked on the gold-minus-best-wrong margin: +27.6 (24.9, 30.2) at AB, +23.8 (21.7, 26.1) at BA. So the distractor's accuracy bump at toy AB is not matched by a clear gain in preference for the right answer; the boxed-probability drop may include a format effect and has not been decomposed further.

**Endpoint validation** (24 layouts picked for diagnosis, not a random sample; `masked_full` against `b1` and `masked_blocked` against `b1_mask` in one session). First-token argmax agrees in 46 of 48 pairs; the largest first-token logit difference is 1.59. Generated tokens are identical in 17 of 24 full pairs and 10 of 24 blocked pairs, and at every first divergence the top two logits are within 0.75 of each other. Consistent with numerical sensitivity under bfloat16; not resolved until the divergent cases are repeated at higher precision.

**Reading.** On MuSiQue, answer-time reasoning recovers most of what blocked encoding loses, and no improvement from reasoning is observed once the supporting paragraph is visible. On the toy it recovers little at this budget. Gold-only access again scores above full access on MuSiQue with thinking off, on a 25-source subset of the earlier 79, so this is not an independent replication. Elliott's Qwen3-4B thinking pilot (branch `codex/results-t4-thinking-25-2026-11-16`) shows the same cap pattern: toy K0 with forced thinking scores 100 when the cap is not hit (n=36 and n=29) and 7 and 29 when it is; on that branch text reports go 40 to 94 on the two-fact toy with thinking and 18 to 12 on split MuSiQue with 29 of 50 at the cap, so "text passing fails" does not stand as a general claim.

**Correction to the 2026-11-15 night entry.** It called access to the gold paragraph "both sufficient and necessary" on MuSiQue. That is too strong. What the rows show is that access to the supporting paragraph recovers the loss and that the distractor and all-but-gold conditions do not; the paragraph was chosen with gold labels, the intervention is at paragraph level, and full access scoring below gold-only means other cross-block access also matters.

## 2026-11-15, night: attention masking on Qwen3-8B

The masking design from the 4B T4 pilot, run on Qwen3-8B (launcher `colab_mask_8b.ipynb`, code at `f4914e4`, bfloat16, seed 0, thinking off, EM). Worker B is re-encoded from scratch in every condition; the conditions differ only in which of worker A's paragraphs B may attend to: none (`blocked`, the same as B1-mask), one distractor paragraph, everything except A's gold paragraph, only A's gold paragraph, or all of A (`full`, the same as B1). 100 toy_d2 two-fact sources and 79 MuSiQue 2-hop sources, all of them labelled composition-required by Part 0 on this model. The rows are in `archive/Qwen3-8B-bfloat16-masking-2026-11-15.tar.gz` (sha256 `99f93d7d…206bdd35`): 716 rows per mask arm (616 for `distractor`, which has no toy AA rows), one commit, no duplicate ids, 0.3–1.9% of answers at the 256-token cap. The tarball also carries the B1, B1-mask and K0 rows it was started from.

| | blocked | distractor | except gold | gold | full |
|---|---|---|---|---|---|
| toy_d2 hop1 AA | 99 | – | 99 | 100 | 100 |
| toy_d2 hop1 AB | 60 | 82 | 82 | 100 | 100 |
| toy_d2 hop1 BA | 69 | 60 | 60 | 100 | 100 |
| toy_d2 hop1 BB | 94 | 100 | 100 | 94 | 100 |
| MuSiQue 2-hop AA | 33 | 30 | 29 | 33 | 34 |
| MuSiQue 2-hop AB | 10 | 11 | 15 | 38 | 29 |
| MuSiQue 2-hop BA | 19 | 16 | 18 | 39 | 29 |
| MuSiQue 2-hop BB | 33 | 30 | 23 | 33 | 23 |

Paired differences at AB+BA, 95% bootstrap interval over sources:

| | toy_d2 hop1 | MuSiQue 2-hop |
|---|---|---|
| gold − blocked | +36 (30, 42) | +24 (15, 34) |
| gold − distractor | +29 (22, 36) | +25 (16, 34) |
| gold − full | 0 (0, 0) | +9 (3, 16) |
| full − blocked | +36 (30, 42) | +15 (7, 23) |
| distractor − blocked | +6 (−2, 14) | −1 (−5, 4) |
| except gold − blocked | +6 (−2, 14) | +2 (−4, 8) |

- **Access to A's gold paragraph alone recovers the whole loss on both datasets.** The 4B pilot found the same (gold − blocked: toy +68 (60, 74), MuSiQue +11 (5, 19)).
- **On MuSiQue, access to everything except the gold paragraph recovers nothing** (+2), and a distractor paragraph recovers nothing (−1). So the gold paragraph is both sufficient and necessary there.
- **On MuSiQue, gold-only beats full access by 9 points** (38 and 39 against 29 and 29). At 4B this difference was +1 (−5, 6). Not explained. One reading is that A's distractor paragraphs hurt B when it can see them; nothing here tests that. The BB cells point the same way (blocked 33, full 23).
- **The toy distractor effect is at AB only**: +22 (10, 34) at AB, −9 (−21, 3) at BA. The 4B pilot showed the same split (AB 66 against 34, BA 24 against 29). Not explained.

Endpoints against the 11-14 run, on the 364 questions both hold: `blocked` gives the same EM as B1-mask on 94.5% of them (59.6 against 61.3), `full` the same as B1 on 96.7% (72.3 against 73.4). They should be the same computation. The cause of the 3–5% that differ has not been looked into.

What it changes: the result that one paragraph's access accounts for the loss now holds at two model sizes. The masked conditions re-encode all of B, so this shows which access is sufficient; it is not a repair method and saves no compute as run. The paragraph is chosen with the gold labels.

Checked: rows read with `results.load_all` from the unpacked tarball; intervals from 2,000 bootstrap resamples over source questions. Not checked: the endpoint mismatch, the gold-answer log-probabilities saved in the rows, F1, other seeds. Nothing is corrected for the number of comparisons; gold − full on MuSiQue is the one most exposed to that.

---

## 2026-11-15: sequential repair on Qwen3-8B

One and two rounds of sequential reading (`scan_1`, `scan_2`, notebook `h5_rounds`) on Qwen3-8B, 50 source questions per cell, seed 0, thinking off, EM. B1 and K0 were loaded from the 11-14 archive, not rerun. The rows are in `archive/Qwen3-8B-bfloat16-2026-11-15.tar.gz`, which holds every 8B row so far: 6,150 rows, 750 for each scan arm, no duplicates. 3.6% of one-round and 3.9% of two-round answers hit the 256-token cap. Real-model tests were skipped again (`RUN_TESTS=False`, same reason as the 11-14 night entry).

| | B1 | K0 | 1 round | 2 rounds |
|---|---|---|---|---|
| MuSiQue 2-hop AB | 46 | 18 | 32 | 40 |
| MuSiQue 2-hop BA | 38 | 20 | 14 | 20 |
| toy_d2 hop1 AB | 100 | 82 | 94 | 98 |
| toy_d2 hop1 BA | 100 | 58 | 22 | 92 |
| toy_d2 hop2 AB | 100 | 72 | 4 | 66 |
| toy_d2 hop2 BA | 100 | 48 | 24 | 44 |
| MuSiQue 3-hop AAB, ABA | 32, 40 | 34, 24 | 42, 42 | 42, 44 |
| MuSiQue 4-hop AABB, ABAB, ABBA | 18, 24, 22 | 22, 16, 16 | 20, 32, 24 | 20, 30, 22 |

Paired differences at AB+BA, 95% bootstrap interval over the 50 sources:

| | 2 rounds − K0 | B1 − 2 rounds | 1 round − K0 | 2 rounds − 1 round |
|---|---|---|---|---|
| MuSiQue 2-hop | +11 (0, 22) | +12 (2, 23) | +4 (−5, 14) | +7 (−2, 15) |
| toy_d2 hop1 | +25 (14, 36) | +5 (1, 9) | −12 (−22, −1) | +37 (30, 44) |
| toy_d2 hop2 | −5 (−20, 10) | +45 (35, 55) | −46 (−58, −34) | +41 (32, 50) |

- **On MuSiQue 2-hop, two rounds recovers about half the gap at 8B** (K0 19, two rounds 30, B1 42). At 4B it reached 31 against B1's 35, a gap of +4 (−6, 14). At 8B the gap to B1 is +12 (2, 23).
- **All of the MuSiQue gain is at AB.** At BA two rounds equals K0 (20 and 20).
- **On two toy facts, two rounds nearly matches B1** (95 against 100). **On three it is no better than K0** (55 against 60).
- **One round is worse than K0 on the toy at 8B** (three facts AB: 4 against 72). The 8B's K0 is already fair there, and a single pass in an order that doesn't match the chain loses it.
- 3-hop and 4-hop MuSiQue differences are small and mixed. The 4-hop cells also carry the cap problem noted in the 11-14 night entry.

What it changes: "sequential rereading recovers most of the loss" held at 4B and does not hold at 8B. What the data supports is that it recovers part of the loss, depends on reading order, and two rounds is not enough for a three-fact chain. The caveats from the 11-14 entry still apply: note budgets differ between one and two rounds, and one seed.

Checked: archive restored with `results.load_archive`; rows read with `results.load_all`; intervals from 2,000 bootstrap resamples over source questions. Not checked: note-cap rates, F1, other seeds.

## 2026-11-14, night: H1 on Qwen3-8B (scale trend, third point)

H1 (B1, B1-mask, K0) on Qwen3-8B, 50 source questions per cell, seed 0, thinking off, EM. It took 6.7 hours on an A100. The rows are in `archive/Qwen3-8B-bfloat16-2026-11-14.tar.gz`: 4,650 rows, all on `18892fb`, no duplicates.

The real-model tests were skipped for this run (`RUN_TESTS=False`). On the 8B, 17 of 18 passed. The one that failed requires K0 with all paragraphs on one worker to give exactly the same text as B1. The two answers were both correct but split at the first word ("The mayor of…" vs "To determine…"), which is a bf16 near-tie, not a bug. The test that checks the RoPE shift against native encoding passed. The results confirm it: with all gold on one worker, K0 matches B1 (below). That test should compare first-token probabilities instead of exact text.

The 8B writes longer answers (89–98 generated tokens on average, thinking off) and 5–7% hit the 256-token cap (B1 4.7%, B1-mask 7.2%, K0 7.1%), against 2–3% at 4B. The cap costs the split arms slightly more than B1.

*Corrected 11-15. The first version of these tables gave the 4B column from the 11-13 run's 100 questions, not the matched 50, and its gap table pooled single-fact questions (hop0) with the chained ones. GPT-6.1-Sol's re-score caught both; the numbers below were recomputed and agree with it.*

B1 / K0 on the same questions for all three sizes (the 8B's 50 per cell):

| | 1.7B | 4B | 8B |
|---|---|---|---|
| toy_d2 hop1 AB | 84 / 8 | 100 / 32 | 100 / 82 |
| toy_d2 hop1 BA | 46 / 6 | 100 / 22 | 100 / 58 |
| toy_d2 hop2 AB | 24 / 20 | 100 / 22 | 100 / 72 |
| toy_d2 hop2 BA | 32 / 18 | 100 / 16 | 100 / 48 |
| MuSiQue 2-hop AB | 18 / 14 | 32 / 18 | 46 / 18 |
| MuSiQue 2-hop BA | 16 / 12 | 38 / 14 | 38 / 20 |
| MuSiQue 2-hop AA | 16 / 8 | 40 / 36 | 52 / 40 |
| MuSiQue 2-hop BB | 6 / 14 | 22 / 34 | 40 / 54 |

B1 − K0 at AB+BA, 95% bootstrap interval over sources:

| | 1.7B | 4B | 8B |
|---|---|---|---|
| toy_d2 hop1 (two facts) | +58 (49, 67) | +73 (63, 83) | +30 (20, 40) |
| toy_d2 hop2 (three facts) | +9 (−2, 20) | +81 (73, 88) | +40 (29, 51) |
| MuSiQue 2-hop | +4 (−1, 10) | +19 (10, 29) | +23 (13, 34) |
| MuSiQue 2-hop, AA+BB | 0 (−6, 6) | −4 (−11, 2) | −1 (−9, 8) |

- With all gold on one worker, K0 ≈ B1 on MuSiQue at every size.
- **On the toy the 8B recovers much of the split loss.** The gap falls by 43 points at two facts (−55, −31) and 41 at three (−53, −29) from 4B to 8B. The 1.7B's B1 chains two facts (65) and K0 drops it to 7; its B1 can't do three (28).
- **On MuSiQue 2-hop split the gap does not shrink with size.** It is +4, +19, +23. The step from 4B to 8B is +4 (−8, 16), so "grows with size" holds only from 1.7B to 4B.
- **Missing attention still accounts for most of the 8B's MuSiQue loss:** B1 − B1-mask is +17 (7, 27) and B1-mask − K0 is +6 (2, 12).
- **K0 has a direction effect on the toy that B1-mask does not.** K0 is 82 at AB and 58 at BA (two facts), 72 and 48 (three facts). B1-mask is 72 and 76, then 72 and 62. B1-mask keeps true positions, so part of K0's BA loss comes from the position shift and stitch, not from missing attention. At 4B the two arms were within about 7 points everywhere.
- **4-hop MuSiQue at 8B is not usable at this answer cap:** 24% of B1, 35% of B1-mask and 32% of K0 answers hit 256 tokens.

Reading (an argument, not tested): the 8B writes the chain out in its answer ("Dev Anand's favorite city is Kastrel. The mayor of Kastrel is…"), so its own text carries the bridge into the second lookup. That works on short, clean toy facts and not on MuSiQue paragraphs. The test is to force a short answer, or score the gold answer's log-probability right after the prompt, and see whether the 8B's toy K0 falls back toward 4B's. The 2-round repair run on the 8B is in the 11-15 entry.

Checked: rows read with `results.load_all`, intervals from 2,000 bootstrap resamples over source questions. Not checked: EM among capped answers, F1, other seeds.

## 2026-11-14, later: H1 on Qwen3-1.7B (scale trend, first point)

H1 (B1, B1-mask, K0) on Qwen3-1.7B, 100 source questions per cell, seed 0, thinking off, EM. It took 7.2 hours on a T4. The rows are in `archive/Qwen3-1.7B-bfloat16-2026-11-14.tar.gz`: 9,300 rows, all on `04e09e1`, no duplicates, no answers cut off, and almost all boxed (0–0.2% unboxed).

B1 / B1-mask / K0, next to the 4B numbers from the 11-13 entry:

| | 1.7B | 4B |
|---|---|---|
| toy_d2 hop1 AB | 90 / 6 / 7 | 100 / 37 / 27 |
| toy_d2 hop1 BA | 52 / 4 / 5 | 100 / 29 / 18 |
| toy_d2 hop1 AA | 74 / 50 / 40 | 100 / 93 / 93 |
| toy_d2 hop2, AB and BA | 23–33 / 13–18 / 19–20 | 98–100 / 22–24 / 18–19 |
| MuSiQue 2-hop, 4 placements (B1) | 5–14 | 24–35 |
| MuSiQue 3-hop (B1) | 23–25 | 41–44 |
| MuSiQue 4-hop (B1) | 2–7 | 18–22 |

Paired differences, 95% bootstrap interval over sources:

| | 1.7B | 4B |
|---|---|---|
| B1 − K0, toy_d2 hop0–2 at AB+BA | +25 (21, 30) | +57 (52, 61) |
| B1 − K0, toy_d2 hop0–2 at AA+BB | +8.7 (5.5, 12) | +3.8 (2, 6) |
| B1 − B1-mask, MuSiQue 2-hop AB+BA | +3.5 (0, 7.5) | +11 (5, 17.5) |

- **1.7B can barely chain even with everything in one context:** 23–37 on three-fact toy chains and 5–14 on MuSiQue 2-hop. So the gap is smaller mostly because B1 has less to lose.
- **Where 1.7B can chain, separate encoding removes it:** toy hop1 AB goes from 90 to 7. It is the same failure as at 4B.
- **B1 is not a clean ceiling at this size.** On toy hop1 it scores 90 at AB, 52 at BA, 74 at AA and 63 at BB. K0 also loses about 9 points with all gold on one worker. A trend across sizes should be reported as the gap relative to what B1 can do, not as raw points.

Checked: rows read with `results.load_all`, intervals from 2,000 bootstrap resamples over source questions.

## 2026-11-14: text reports and sequential repair (Qwen3-4B, preliminary)

The 11-13 overnight run used the new runner, starting from the 11-13 tarball. It ran two arms that had not been run before, at 50 source questions per cell, seed 0, thinking off, EM:

- **T1** (`t1_1024`): each worker writes a text report of what its paragraphs say about the question, and the orchestrator answers from the reports.
- **Repair** (`scan_1`, `scan_2`, notebook `h5_rounds`): the orchestrator reads one worker's cache at a time and keeps a one-line note between reads, for 1 or 2 rounds over all workers, then answers.

The rows are in `archive/Qwen3-4B-bfloat16-2026-11-14.tar.gz`. It includes everything in the 11-13 tarball, so it replaces it for `LOAD`. The 11-13 rows in it are byte-identical to the 11-13 tarball's, so nothing was rerun. The 1,700 new rows are all on `04e09e1`, with no duplicates, and 9 answers hit the length cap. That makes this the first full Colab run with the new runner: the load, the gate and the resume all worked.

Repair (B1 / K0 / 1 round / 2 rounds):

| | B1 | K0 | 1 round | 2 rounds |
|---|---|---|---|---|
| toy_d2 hop1 AB | 100 | 32 | 100 | 100 |
| toy_d2 hop1 BA | 100 | 22 | 14 | 86 |
| toy_d2 hop2 AB | 100 | 22 | 2 | 70 |
| toy_d2 hop2 BA | 100 | 16 | 8 | 56 |
| toy_d0, same four cells | 100 | 22–50 | 6–100 | 76–100 |
| MuSiQue 2-hop AB | 32 | 18 | 32 | 32 |
| MuSiQue 2-hop BA | 38 | 14 | 20 | 30 |
| MuSiQue 3-hop, 4-hop (5 cells) | 18–48 | 24–44 | 16–40 | 18–42 |

Paired differences, mean over source questions, 95% bootstrap interval over sources:

- 2 rounds − K0: toy_d2 +55 (47.5, 63), toy_d0 +56 (48, 64.5), MuSiQue 2-hop AB+BA +15 (6, 26), all MuSiQue +1.6 (−3.6, 7).
- B1 − 2 rounds, MuSiQue 2-hop AB+BA: +4 (−6, 14). No gap to B1 can be resolved there at n = 50.

- **Two sequential rounds recover most of the H1 loss on the toy and on 2-hop MuSiQue,** which is where H1 found a cost. 3-hop and 4-hop don't move, as expected, since B1 can't chain there at this size either (11-13 entry).
- **One round helps only when the chain is read in order** (hop1 AB, one crossing). Beyond that it does worse than K0 (toy_d2 hop2 AB: 2 vs 22). In the notes it writes "not mentioned" early, and the orchestrator then trusts that note. Rounds have to keep up with crossings.
- At 3 facts (hop2) two rounds still fall short of B1 (56–86). At hop2 BA the answer is in the final note only 8% of the time but EM is 56, so the final read of the caches does part of the work.

T1 (only four cells ran):

| | B1 | B1-mask | K0 | T1 |
|---|---|---|---|---|
| toy_d2 hop1 AB | 100 | 36 | 32 | 38 |
| toy_d2 hop1 BB | 100 | 96 | 96 | 100 |
| MuSiQue 2-hop AB | 32 | 18 | 18 | 20 |
| MuSiQue 2-hop BB | 22 | 30 | 34 | 46 |

T1 − K0: toy +5 (−5, 15), MuSiQue +7 (−4, 19).

- **T1 fails at the orchestrator, not in the reports.** On toy_d2 hop1 AB the gold answer is in the workers' reports for 50 of 50 questions, yet T1 gets 19 right. The orchestrator has both facts as text and still doesn't chain them. That is the same failure as K0, a missed second lookup at answer time. The reports also state what they can't answer ("The document does not provide information about the mayor of Kastrel"), which may push the orchestrator to answer "not stated". That is a guess and would need a controlled test (reports with those sentences removed).
- MuSiQue BB, where T1 beats B1 by 24, has not been looked into. Possibly the worker filters the distractors when it writes its report.

Caveats on the repair result:

- **Note budget.** One round writes one note and two rounds write three, so the two-round gain mixes more reads with more space for intermediate facts. Some longer-hop notes also hit the 48-token note cap. Equal total note budgets across 1 and 2 rounds would separate the two.
- **One vs two rounds on MuSiQue 2-hop is not resolved:** 26 vs 31 pooled over AB+BA. "Rounds have to keep up with crossings" rests on the toy only.

What it changes: the comparison that matters is independent vs sequential, not latent vs text. Whether the handoff is caches or text, the orchestrator fails to make the second lookup when the facts arrive separately. A second sequential pass with a few tokens of notes recovers most of the loss at 2-hop. That is the case for routed re-query.

Checked: rows read with `results.load_all`, scored by the repo's `score`, intervals from 2,000 bootstrap resamples over source questions. Not checked: other seeds, other models, F1, hand labels of the notes.

Still running on 11-14: Qwen3-1.7B H1 at 100 per cell. Qwen3-8B H1 is queued and needs a bigger GPU than a T4.

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

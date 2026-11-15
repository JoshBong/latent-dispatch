# Run the archived experiment with attention masking

Open [the masking launcher in Colab](https://colab.research.google.com/github/ajit-mallav/latent-dispatch/blob/masking-experiment/colab_masking.ipynb), select an **L4 or A100 GPU**, and choose **Run all**. The launcher calls the existing `colab.ipynb` runner. It does not contain a second implementation of the experiment or runner.

The source is the published fork branch so the notebook can run before the pull request is merged. After merging, Josh can use `REPO = "https://github.com/JoshBong/latent-dispatch.git"` and `COMMIT = "main"`. Pin `COMMIT` to a commit hash for a fixed code version. The launcher mounts Google Drive and uses `/content/drive/MyDrive/latent-dispatch-masking-2026-11-14` for the checkout, per-question results, and executed notebooks. Authorize Drive when Colab prompts. Run all again with the same `ROOT` to resume, including on a new runtime. The dated default avoids silently reusing the older masking checkout. The saved checkout keeps its existing commit; choose a new `ROOT` when changing source or code version.

The preset is:

```python
MODEL = "Qwen/Qwen3-4B"
DTYPE = "bfloat16"
ROOT = "/content/drive/MyDrive/latent-dispatch-masking-2026-11-14"
SEEDS = [0, 1, 2]
N_PER_CELL = None
LOAD = None  # pick the latest matching archive in the checkout
NOTEBOOKS = ["part0_ruler", "h1_encoding", "h2_thinking", "h1_masking"]
SKIP = {"h2_thinking": ["t1_1024"]}
RUN_TESTS = True
```

This is a complete smoke-then-full execution of the full benchmark across all three data seeds, with masking added. The archive covers part of seed 0; the runner completes the remaining Part 0/H1/H2 rows before the full masking stage. To run only Josh's original sample plus masking, use `SEEDS = [0]` and `N_PER_CELL = 100`. No model experiments were run as part of preparing this setup.

Set `MODEL` in the launcher to select the same design at another size. `LOAD = None` selects only the newest dated archive for that model and dtype in `ROOT/archive/`; it never loads another model's archive. Set an explicit list to pin an archive, or `LOAD = []` to start without one. The common runner still validates the archive contents and benchmark/prompt provenance before using it.

| Model setting | Archive currently selected | What must still be completed |
|---|---|---|
| `Qwen/Qwen3-1.7B` | `Qwen3-1.7B-bfloat16-2026-11-14` | Part 0 labels, missing three-seed H1/H2 rows, and all masking rows |
| `Qwen/Qwen3-4B` (default) | `Qwen3-4B-bfloat16-2026-11-14` | Missing three-seed Part 0/H1/H2 rows and all masking rows |
| `Qwen/Qwen3-8B` | None until its archive is in the checkout | Missing Part 0/H1/H2 and masking rows; existing rows in `ROOT/results/` are still reused |

Use an A100 for 8B. When Josh's 8B run finishes, copy its cumulative tarball into `ROOT/archive/`, or select a new checkout containing the published archive. The launcher will then select it on the next run. An explicit missing archive fails before starting the common runner. Results and executed runner logs are stored separately for each model/dtype, so changing `MODEL` in the same source checkout does not mix model outputs.

The common runner checks the archived files, builds and checksum-checks the benchmarks, runs unit and real-model checks, executes smoke passes, then runs the selected notebooks in order. Each answer is saved immediately. An interrupted final JSONL fragment is removed before appending so that question can rerun; a complete final row missing its newline is preserved. Part 0's smoke labels have their own filename so they cannot replace the full labels loaded from the archive. The masking smoke uses a fixed, unfiltered sample to match the runner's smoke audit; the full run requires complete, non-smoke composition labels for its requested seeds.

The masking stage recomputes the unchanged B1 tokens at their true positions and changes only which A tokens B can attend to. The final answer can still attend to both blocks.

| Condition | What B may see in A |
|---|---|
| `blocked` | Nothing; reproduces B1-mask |
| `gold` | Gold paragraphs only |
| `distractor` | One reproducibly sampled distractor paragraph from those before A's first gold paragraph |
| `before_gold` | Everything in A before its first gold paragraph |

The two controls use only A tokens that come before A's gold. A attends causally within itself, so an A token after a gold paragraph has already read it and would relay the gold fact to B. When A's gold is its first paragraph the controls are undefined and those rows are skipped for those two arms. On the seed-0, 100-per-cell sample, before the Part 0 composition filter, that leaves 42–43 of 100 toy AB/BA rows (toy workers hold two paragraphs), no toy AA rows, and 83–100 of 100 MuSiQue rows per placement.
| `full` | All tokens; reproduces B1 |

The masking datasets are the two-fact toy (`toy_d2`, `hop1`) and two-hop MuSiQue, with AA/AB/BA/BB placements. Full-run analysis uses questions labeled as requiring composition by Part 0. The default scope has 21,024 candidate question/layout rows before that filter: 6,000 toy rows and 15,024 MuSiQue rows. The number that passes the filter is determined by the completed three-seed Part 0 probes; it is not known in advance. As a smaller reference, the archived seed-0/100-per-cell sample has 740 eligible question rows before the controls' before-gold filter; the notebook prints the exact per-arm counts. The notebook prints eligible, saved, and remaining counts before loading a model.

The new rows record exact-match output, selected A paragraph indices, visible token counts, and the teacher-forced log-probability of the first gold answer in boxed format. Both summed and per-token mean log-probabilities are saved. Existing B1 and B1-mask outputs are read for paired endpoint checks; they are not regenerated by those checks. Masked endpoints are newly computed because their likelihoods and direct mask outputs are part of this experiment. Names beginning `mask_a_` keep these results separate from the previous arms.

Read the smoke's raw answers and cut rates. The existing runner stops on broken B1 and skips broken arms; the masking notebook requires both blocked and full endpoints and fails explicitly if either is skipped. A skipped non-endpoint arm is excluded from its tables. Missing full-run labels or missing expected result rows also fail explicitly. If an answer cap needs changing, the results loader rejects mixing different options under the same arm filename; use a fresh results folder or a new arm name for that rerun.

The raw results and executed notebooks stay in the Drive checkout across runtimes. The runner also creates `/content/results-Qwen3-4B-bfloat16.tar.gz` after each full notebook; download that runtime-local bundle before deleting the runtime if you want a single file to share. It includes:

- Previous and new JSONL rows under `results/Qwen3-4B-bfloat16/`.
- `h1_masking-plan.json`, recording the exact commit, scope, conditions, and resume counts.
- `notebooks/runs/Qwen3-4B-h1_masking.ipynb`, with output audits, coverage, accuracy, endpoint checks, and paired confidence intervals for EM and gold likelihood.
- The other executed notebooks from the original experiment.

This implements the journal's first masking stage, varying A's visibility. Selecting bridge-specific versus matched random query spans inside B remains the subsequent experiment, guided by these results. Dense attention masks require memory quadratic in context length.

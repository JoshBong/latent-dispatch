# Completed T4 masking pilot — November 15, 2026

This branch preserves the completed masking run without changing `main` or PR #1.
Its parent `f4914e4` contains PR #1 at `e7fbc32` plus the test-only float16 RoPE
rounding tolerance adjustment used for the run. Model and masking code are unchanged.

- Model: Qwen/Qwen3-4B, float16, NVIDIA T4.
- Scope: seed 0, first 100 source questions each from the two-fact toy and two-hop MuSiQue.
- Composition checks: closed-book and both single-gold-paragraph probes for all 200 sources.
- Eligible: 187 sources, expanded to 748 AA/AB/BA/BB layouts.
- Mask rows: blocked 748; gold 748; distractor 648; except-gold 748; full 748. Total: **3,640**.
- The 100 excluded distractor controls are toy AA layouts with no A distractor.
- Finished successfully at 2026-11-15 06:07 UTC (01:07 EST), exit code 0.
- Masking runtime: approximately 4 hours 12 minutes; job including missing probes: 4 hours 55 minutes.

## Saved artifacts

- [Executed notebook](Qwen3-4B-float16-h1_masking-only.ipynb): saved tables, audits, and paired comparisons.
- [Restorable results archive](../../archive/Qwen3-4B-float16-2026-11-15.tar.gz): raw JSONL results, composition labels,
  manifests, and the executed notebook under the repository's standard archive layout.
- [Masking plan](h1_masking-plan.json): conditions, eligible counts, options, and original resume counts.
- [Launch manifest](masking-only-launch.json): source commit, prerequisite coverage at launch, and runner SHA256.
- [Composition labels](part0_requires_composition.json).
- [Exact Python runner](run-masking-only.py) and [VM wrapper](run-masking-only-vm.sh).
- [SHA256 checksums](SHA256SUMS), with paths relative to the repository root.

The archive preserves all saved float16 rows, including smoke and prerequisite rows
from the earlier over-scoped launch. The completed masking notebook filters to the
documented pilot scope; those extra rows do not enter its masking tables.
The runner files preserve the original VM paths and are not general-purpose installers.

## Restore and inspect

From a checkout of this branch with the project dependencies installed:

```bash
sha256sum -c reports/2026-11-15-t4-masking/SHA256SUMS
python -c 'from latent_dispatch.results import load_archive; load_archive("archive/Qwen3-4B-float16-2026-11-15.tar.gz")'
```

The existing loader keeps files already present, so restoring does not replace
local results. The directly linked notebook can be read without extracting anything.
No additional GPU run is needed to inspect it.

Validation before publishing: 206 fast tests and all 21 Qwen3-4B/float16 GPU checks
passed before the run; the completed notebook has no error cells; all expected mask
IDs are present without duplicates; the recorded runner hash matches the saved script;
the packaged archive was restored and validated with the repository's loader.
Scientific interpretations of the results have not yet been reviewed.

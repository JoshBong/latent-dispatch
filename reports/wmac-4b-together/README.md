# Qwen3-4B FP16 same-worker control

Requested follow-up: the control layout, not the positional follow-up. Uses the
exact Python experiment script extracted from colab_wmac_together.ipynb at
9747d6f, SHA256 e1f9b72fc3b3b5ead0b54e42373e0057ab487f5a9d5009f0c1cb51a1dbe08de2.
Core experiment files match 64044d8 byte-for-byte; no model, masking, cache or
decoding implementation is changed. The notebook's outer setup now accepts
MODELS=["4b"] with an L4 and selects --dtype float16 for that key; 8B defaults
remain BF16/A100.

Frozen input SHA256:
8abf29425611b1f3b65f5016fd40651ef0281bdb6de9852e1df2940f33953058.
Same 200 source questions and aliases as the completed Qwen3-4B main run.
Each source has AA and BB layouts, both supporting paragraphs together in the
same worker, preserving the full paragraph multiset and worker paragraph counts.
Conditions are blocked and full, thinking off: 400 layouts × 2 = 800 answers.
No optional B1 or positional experiment is run. Main answer cap remains 1,024.

The unchanged script selftest passed on CPU. Qwen3-4B's pinned tokenizer matches;
all 400 layouts fit, zero are dropped, with 1,165–4,646 context tokens (median
2,406). Before the main control, a separate smoke evaluates the longest AA and BB
layouts under both conditions. All four must finish with nonempty boxed answers
and no answer cap. All four longest-layout GPU smoke checks passed. The full control started
and saved 39/800 answers at the latest observed checkpoint; its reported pace
was 2.94 seconds/answer. Full completion remains pending: see FINAL.md after
publication. Runtime confirms Qwen3-4B, revision 1cfa9a7208912126459214e8b04321603b3df60c,
float16, NVIDIA L4, torch 2.6.0+cu124, transformers 4.57.6, SDPA.

The requested GPU is one L4, FP16, matching the main run's hardware/precision.
VM: latent-wmac-l4-v3, us-central1-b. A restart initially hit capacity; a fresh
hold was then obtained and consumed. The temporary reservation and unused
snapshot were deleted. The VM has a four-hour hard stop. The wrapper emits the
configured completion/failure email marker and schedules shutdown ten minutes
after exit. Original main-run results and checkout are preserved.

Remote checkout: /home/elkalt/latent-dispatch-wmac-together
Unit: latent-wmac-4b-together
Status/log: /home/elkalt/wmac-4b-together.status and wmac-4b-together.log
Output: /home/elkalt/wmac-4b-together/4b/together/{rows.jsonl,plan.json,status.json}
Result archive: /home/elkalt/wmac-4b-together-results.tar.gz

The small complete/partial archive is also written as checksum-framed chunks to
the already enabled serial log, allowing recovery after shutdown if local SSH
collection misses the ten-minute window. Missing chunks are rejected; source
and integrity selftests are in artifact.py. This is transport, not model output.
The input bundle is referenced by its immutable hash rather than copied into
every log. Full framing/recovery must still be verified on actual cloud output.

Analysis should compare (full−blocked) in AA/BB against the same sources' AB/BA
main-run gap, paired by source. A near-zero same-worker gap would support the
claim that cross-worker dependence matters, rather than a generic masking cost.
It would not establish equivalence merely because a confidence interval contains
zero. Benchmark annotation ambiguity and the main run's other limitations remain.

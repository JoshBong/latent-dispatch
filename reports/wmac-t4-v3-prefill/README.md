# Memory-saving prefill retry

The original PR #3 smoke attempt is preserved in ../wmac-t4-v3, including its
failure log and raw rows. It passed 74 tests and all 16 validation smoke answers,
then saved five MuSiQue smoke answers before B1's longest joint prefill ran out
of GPU memory. No main responses were generated.

Model.prefill now calls the same decoder backbone directly with identical
input IDs, cache/position IDs, attention mask and precision. It avoids the
unused vocabulary projection and retained intermediate layer states, and clones
the final-token hidden state so its view cannot retain a full-sequence tensor.
The forward method used for scoring/generation is unchanged.

102 CPU tests passed, including cache/final-hidden comparisons for whole,
cached, chunked and shifted masked prefill on Qwen3, Llama, Mistral and Phi3. The retry
also requires a real Qwen3-4B FP16 short-context bitwise comparison and a
longest-frozen-MuSiQue prefill memory check before the usual smoke/full run.
Actual GPU results will be recorded after capacity is obtained.

The exact frozen questions, accepted answers, model revisions, FP16 precision,
context and generation limits are unchanged. The shared prepared.json SHA256
remains 8abf29425611b1f3b65f5016fd40651ef0281bdb6de9852e1df2940f33953058.
The original shared bundle is in ../wmac-t4-v3/shared-prepared.tar.gz.

Because code hashes are part of the row fingerprint, this code revision gets
fresh output manifests and smoke rows. The earlier 21 smoke rows remain in the
original attempt's archive; they are not silently relabeled as new-code rows.
No main responses existed to reuse. Model weights/tokenizer caches are reused.

The user explicitly approved the memory change and further GPU testing after
the failure. The first restart was rejected for zone capacity; a broad T4
search tried all 51 advertised zones without finding capacity. All experiment
VMs remain stopped and no reservation remains. GPU parity/memory checks and
the retry smoke/full run have NOT executed; only the 102 CPU tests passed.

The first real-model retry passed the short-context bitwise comparison but
still exhausted T4 memory on the longest prefill. Its log is preserved in
first-memory-check-failure.log.gz. Prefill now processes at most 1,024 query
tokens at a time, retaining the complete preceding cache and explicit positions.
No input is shortened. FP16 operation partitioning can change rounding, so the
GPU gate additionally compares a 1,152-token chunked run with unchunked output
at relative hidden/KV and next-logit error <=1e-3, with the same next token.
All 102 CPU checks passed, including cross-boundary cache, masks, positions and
next-logit comparisons. Real-model GPU results are pending the second check.

The automatic-backend chunking trial was rejected: real-model chunked
hidden/KV relative error was 0.05969, above the 1e-3 gate. Its log is preserved.
Tiny-tensor profiling on the real T4 confirmed native GQA selects the math
kernel while expanded KV selects efficient attention. The next revision keeps
math attention throughout fresh unmasked T4 prefill, avoiding an implicit
backend switch between chunks; already-cached/masked inputs keep their original
automatic backend. Newer CUDA GPUs retain unchunked prefill. All 102 CPU tests
passed again. The real-model gate now checks cached as well as fresh chunking.

## Final diagnostic outcome

The backend-controlled real-model trial also failed the 1e-3 gate: maximum
relative hidden/KV error was 0.0592878. No smoke or main run was launched under
chunking. Chunked prefill was removed from the active code; both rejected
revisions remain in Git history with their logs. The remaining decoder-only
prefill optimization preserves the full-context execution shape, but the T4
longest-context OOM is unresolved. A larger-memory GPU is the proposed next
step, subject to quota, capacity, a fresh longest-context smoke, and validation.

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

94 CPU tests passed, including cache/final-hidden comparisons for whole,
cached and shifted masked prefill on Qwen3, Llama, Mistral and Phi3. The retry
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
search was started. This record alone is not evidence that GPU tests passed
or that the full run started.

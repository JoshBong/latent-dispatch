# WMAC v3 Qwen3-4B FP16 on one L4

The user approved switching the 4B job to one L4 after T4 longest-context OOM
and rejected chunking parity checks. This run keeps full-context prefill with
the decoder-only optimization, not chunking or quantization. Questions, aliases,
model revision, FP16 weights and generation/context budgets remain unchanged.

VM latent-wmac-l4-v3 is RUNNING in us-central1-b, g2-standard-4, with a
verified NVIDIA L4 (23034 MiB), driver 580.178.04, torch 2.6.0+cu124 and
transformers 4.57.6. Runtime source commit is a432bfd. All 102 prescribed
checks passed on this VM before real-model checks started. Real-model/smoke
results and full-run completion are still pending. The consumed reservation
was deleted; the original source VMs remain stopped and unchanged.

The exact shared prepared.json SHA256 is
8abf29425611b1f3b65f5016fd40651ef0281bdb6de9852e1df2940f33953058.
Its bundle/review packet remains in ../wmac-t4-v3/shared-prepared.tar.gz.

The existing launcher's `t4` argument selects the 4B FP16 component group,
not a physical-device assertion. We run that group with the separate run name
wmac_gpu_v3_l4. Frozen config/manifest fields named t4 and gpu-t4 retain their
logical-profile meaning; actual runtime hardware is L4 and is recorded in the
runtime metadata, GPU diagnostic, wrapper, instance record and this report.
The later A100 run must reuse the same prepared.json.

The wrapper gates full execution on 102 CPU checks, real-model bitwise cache/
hidden-state parity versus the original causal LM at short/medium/cached input,
longest frozen MuSiQue joint-prefill fit, then all validation/MuSiQue smoke rows.
Main target: 96 validation answers and 2,400 MuSiQue answers, plus probes/smoke.

Expected remote checkout: /home/elkalt/latent-dispatch-wmac-l4
Unit: latent-wmac-l4; status/log: /home/elkalt/wmac-l4.status and wmac-l4.log.
The result archive retains its logical-profile suffix:
results/wmac_gpu_v3_l4-t4.tar.gz.

Partial/final results will be published to codex/wmac-l4-v3. The email alert
reports exit outcome independently of the local publisher. VM shutdown is
scheduled ten minutes after exit; GCP enforces a 24-hour STOP limit. A local
publisher retrieves and pushes artifacts, then stops the VM. If the local host
is unavailable, results remain on the stopped VM disk for recovery.

# WMAC v3 T4 run

New VM `latent-wmac-t4-v3`, europe-central2-c, n1-standard-4 + Tesla T4.
Pinned PR #3 runtime commit: `64044d8c8db029af3cbede5984f02bf9ff43740e`.
CUDA is verified; torch 2.6.0+cu124 and transformers 4.57.6. All 74 prescribed
checks passed on this VM before the smoke run started.

The frozen shared bundle was prepared once, including all three tokenizers,
and the copied prepared.json matches SHA256
`8abf29425611b1f3b65f5016fd40651ef0281bdb6de9852e1df2940f33953058`.
shared-prepared.tar.gz preserves that exact bundle plus its review packet for
the later A100 run. No model credentials are included.

Only Qwen3-4B FP16 runs on this T4. The wrapper runs smoke-only first, followed
by `python scripts/run_wmac_gpu.py t4` only after successful smoke completion.
The target is 96 validation answers and 2,400 main MuSiQue answers, plus probes
and smoke outputs (3,124 generated answers overall). The six MuSiQue conditions
are blocked, gold, full, matched irrelevant access, B1, and original K0.
Old thinking-pilot responses are preserved separately and are not imported into
this new protocol. The PR runner resumes this protocol's own saved rows.

Remote paths:
- Checkout: /home/elkalt/latent-dispatch-wmac-v3
- Unit: latent-wmac-t4
- Log/status: /home/elkalt/wmac-t4.log and /home/elkalt/wmac-t4.status
- Archive: results/wmac_gpu_v3-t4.tar.gz

Completion/failure emits the configured serial-log email alert. The wrapper
schedules VM shutdown ten minutes after exit, and GCP enforces a 24-hour STOP
limit. The source VMs remain stopped and unchanged; the consumed reservation
and temporary source snapshot were deleted.

A local publisher watches for exit, retrieves the complete/partial artifacts,
and pushes them to this branch before stopping the VM. This publisher requires
the local computer to remain available. If it is interrupted, saved results
remain on the stopped VM disk for recovery. The independent email/shutdown
paths remain on the cloud VM. A launch record is not evidence of completion;
read the eventual final status and component summaries.

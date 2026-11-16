# Prepared 50-per-cell continuation — not started

**Launch is blocked by T4 capacity as of November 16, 2026.** The original
`latent-dispatch-gpu` VM could not restart in me-west1-c. Additional VM requests
and a broad pass over advertised T4 zones using short-lived capacity holds also
returned ZONE_RESOURCE_POOL_EXHAUSTED. No new model evaluations were performed.
The original stopped VM and its saved cache are intact. Failed disposable
allocations were removed, and the temporary source snapshot was removed after
the unsuccessful search. No reservation is intentionally retained.

## Ready-to-resume scope

Continues the completed 25-per-cell Qwen3-4B FP16 thinking pilot. Target:
350 layouts × six conditions = 2100 responses. **1150 main responses already
exist**, leaving **950 new responses**: 75 B1-forced and 175 in each other
condition. All 12 smoke responses are reusable. Result names, model, dtype,
thinking budget (1024), answer cap (1024), and report budget (1024) are unchanged.
The completed 25-per-cell artifacts remain in reports/2026-11-16-t4-thinking-25.

resume-baseline.json records the saved byte hashes, counts and unchanged options.
verify_thinking_cache.py gates inference on those original byte prefixes and
unique IDs. Local simulation using the actual results.run functions verified
that only missing IDs invoke generation, saved bytes are preserved, and a second
resume invokes no additional generation. The runtime wrapper requires a working
CUDA T4 and refuses CPU fallback.

## Launch when capacity becomes available

Use the original VM's existing environment and cached rows, or snapshot its
stopped disk to provision an equivalent environment elsewhere. The temporary
snapshot used for this search was deleted; create a fresh one before using the
provisioner's --source-snapshot option. Apply this branch to the VM's existing
checkout, preserve the prior launch manifest/log/status, and validate the cache
before starting scripts/run_thinking_vm.sh as the elkalt user in a detached unit.
Do not rename result files or change generation options.

The original VM now has a 24-hour GCP STOP limit and automatic restart disabled.
New VMs should use the same limit. The guest wrapper schedules shutdown ten
minutes after success/failure, allowing result retrieval. Update the existing
email alert for any new instance ID and restart the publisher only after the new
job's status is running. No continuation publisher has been started while the
job is blocked, and the existing email policy has not been redirected to a
nonexistent instance.

Provisioning supports saved-environment snapshots. Subnet discovery reads the
VPC's authoritative links and each required subnet explicitly; unreadable
ranges stop allocation. All 13 provisioning tests pass.

# Running 50-per-cell Qwen3-4B thinking continuation

Obtained one T4 in **us-east1-c** on November 16, 2026 after a broad capacity
search. Active VM: `latent-thinking-t4-50`, instance ID 7396938339717671203.
Machine: n1-standard-4 + one Tesla T4 (15360 MiB reported VRAM), Ubuntu 22.04,
with the original Python environment, model/tokenizer files, benchmark and
response cache restored from a snapshot. The original me-west1-c VM remains
stopped and unchanged. The temporary reservation was consumed and deleted;
the source snapshot is removed after cache verification.

Launch preflight verified every recorded cache byte hash and unique ID. CUDA
must be available on a T4; CPU fallback is refused. Model/tokenizer loading is
offline, using the restored files. Source runtime commit: `133b213`.
All model code and generation settings match the completed 25-per-cell pilot.
The detached unit is `latent-thinking-50`. The runtime has a GCP-enforced 24-hour
STOP limit, a guest shutdown fallback, and shutdown ten minutes after job exit.

Existing email alert 8621860823762638783 now monitors this instance and remains
enabled for elliott.kalt@gmail.com. Local service latent-thinking-publish-50
collects and pushes results/logs on completion or failure, then stops this VM.
The publisher runs on the local computer; the VM's email and shutdown paths do
not require the local publisher to stay connected. Saved responses remain on
the boot disk even if retrieval is interrupted.

## Continuation scope

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


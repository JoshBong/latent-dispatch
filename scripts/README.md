# T4 thinking pilot

Run `scripts/run_thinking_vm.sh` on the existing T4. It runs fast tests,
Qwen3-4B/float16 real-model tests, and then `run_thinking_pilot.py`.
The branch includes fix/think-eos so EOS during reasoning closes thinking and
continues to a final answer.

Scope: seed 0, first 50 rows per cell: toy hop1/hop2 AB/BA and MuSiQue 2hop
AB/BA/BB (350 question layouts; source questions repeat across placements).
All six arms run on those same layouts: B1, K0, T1 with thinking forced/off.
Thinking budget 1024, answer cap 256, T1 report budget 1024 per worker.
Forced arms run first. T1 enables thinking only in the orchestrator; worker
reports use the existing report prompt. No masking or composition-filter run.

The smoke pass checks one toy and one MuSiQue question in every arm, prints raw
outputs, and requires an untruncated boxed answer after the closing think tag.
Results resume under unique thinking_pilot names. The runner prints thinking
length and cap counts, writes a summary CSV, and checkpoints an archive after
each arm and on Python exit. Raw rows flush after every question.

The VM wrapper records success/failure, emits the existing Monitoring serial
marker, and shuts down on exit. A separate guest shutdown is scheduled at 24h.
No new email policy is created. Artifacts remain on the boot disk on shutdown.

Paths on the VM:
- checkout: /home/elkalt/latent-dispatch-thinking
- log/status: /home/elkalt/thinking-experiment.log and .status
- bundle in checkout: results-Qwen3-4B-float16-thinking.tar.gz

The bundle contains scripts as well as results and is not directly accepted by
results.load_archive; inspect/extract its named files explicitly.

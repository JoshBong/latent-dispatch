# Qwen3-4B T4 thinking pilot

Final VM status:
```
exit_code=1
finished_at=2026-11-15T22:51:47+00:00
```

Seed 0, 50 layouts per cell, seven cells, 350 question layouts. B1/K0/T1 with thinking off and forced up to 1024 tokens. FP16, 256-token answer cap, 1024-token worker reports. See scripts/README.md for protocol.

The archive contains raw JSONL rows, manifests, scripts, and logs. It uses a custom layout and is not a results.load_archive archive. Partial rows are preserved on failure. See row-counts.json and the log before interpreting results.

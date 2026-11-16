# 50-per-cell continuation

Continues the completed 25-per-cell Qwen3-4B FP16 thinking pilot, preserving
all result names and generation settings. Target: 350 layouts × six conditions,
2100 responses. Existing main cache: 1150 responses; new work: 950 responses.
All 12 smoke responses are also reused. The completed 25-per-cell report is
preserved in reports/2026-11-16-t4-thinking-25.

resume-baseline.json records the cache hashes, counts, and unchanged options.
verify_thinking_cache.py gates launch on those exact saved bytes and unique IDs.
Local simulation with actual results.run functions verified that only missing IDs
invoke generation and that repeat resume invokes none.

The original me-west1-c T4 could not restart due to capacity. Provisioning from a
snapshot preserves its Python environment, downloaded model, benchmark and cache
on a separate VM; the original instance and disk remain stopped and unchanged.
The new VM has a 24-hour runtime limit and guest shutdown ten minutes after job
exit. Email monitoring is updated for the new instance. Final artifacts are to
be published here on completion or failure.

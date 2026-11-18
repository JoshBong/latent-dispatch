#!/bin/bash
set -Eeuo pipefail
exec >> /home/elkalt/wmac-l4.log 2>&1
phase=checks
finish() {
  result=$?
  trap - EXIT
  outcome=FAILED
  if [ "$result" -eq 0 ] && [ "$phase" = full ]; then outcome=COMPLETED; fi
  if [ "$result" -eq 2 ] || [ "$result" -eq 143 ]; then outcome=STOPPED; fi
  printf 'exit_code=%s\nfinished_at=%s\nphase=%s\n' "$result" "$(date -u +%FT%TZ)" "$phase" > /home/elkalt/wmac-l4.status
  printf 'LATENT_MASKING_JOB_RESULT outcome=%s experiment=wmac-v3 gpu=l4 exit_code=%s\n' "$outcome" "$result" | sudo tee /dev/ttyS0
  sudo shutdown -h +10 'WMAC L4 finished; saved results remain on disk.' || true
  exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM INT
cd /home/elkalt/latent-dispatch-wmac-l4
export PATH=/home/elkalt/latent-dispatch-masking-pr1/.venv/bin:$PATH
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
printf 'started_at=%s\nphase=checks\n' "$(date -u +%FT%TZ)" > /home/elkalt/wmac-l4.status
python -c 'import torch,transformers; assert torch.cuda.is_available(); assert "L4" in torch.cuda.get_device_name(0); print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))'
python -m pip freeze > /home/elkalt/wmac-l4-requirements.txt
python -m pytest tests/test_prefill_memory.py tests/test_kv.py tests/test_wmac_gpu.py tests/test_wmac_a100.py tests/test_masking.py tests/test_paper_ablations.py -q
phase=memory_check
python reports/wmac-l4-v3/gpu_prefill_check.py | tee /home/elkalt/wmac-l4-check.json
phase=smoke
printf 'phase=smoke\n' >> /home/elkalt/wmac-l4.status
python scripts/run_wmac_gpu.py t4 --run-name wmac_gpu_v3_l4 --smoke-only
phase=full
printf 'phase=full\n' >> /home/elkalt/wmac-l4.status
python scripts/run_wmac_gpu.py t4 --run-name wmac_gpu_v3_l4

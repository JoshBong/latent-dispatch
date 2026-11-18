#!/bin/bash
set -Eeuo pipefail
exec >> /home/elkalt/wmac-4b-together.log 2>&1
cd /home/elkalt/latent-dispatch-wmac-together
export PATH=/home/elkalt/latent-dispatch-masking-pr1/.venv/bin:$PATH
export PYTHONPATH=/home/elkalt/latent-dispatch-wmac-together
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
phase=checks
finish() {
 result=$?
 trap - EXIT
 outcome=FAILED
 if [ "$result" -eq 0 ] && [ "$phase" = full ]; then outcome=COMPLETED; fi
 if [ "$result" -eq 2 ] || [ "$result" -eq 143 ]; then outcome=STOPPED; fi
 printf 'exit_code=%s\nfinished_at=%s\nphase=%s\n' "$result" "$(date -u +%FT%TZ)" "$phase" > /home/elkalt/wmac-4b-together.status
 python reports/wmac-4b-together/artifact.py pack --home /home/elkalt --checkout "$PWD" --serial || true
 printf 'LATENT_MASKING_JOB_RESULT outcome=%s experiment=wmac-4b-together gpu=l4 exit_code=%s\n' "$outcome" "$result" | sudo tee /dev/ttyS0
 sudo shutdown -h +10 'WMAC 4B together control finished; results saved.' || true
 exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM INT
printf 'started_at=%s\nphase=checks\n' "$(date -u +%FT%TZ)" > /home/elkalt/wmac-4b-together.status
python -c 'import torch; assert torch.cuda.is_available() and "L4" in torch.cuda.get_device_name(0); print(torch.__version__,torch.cuda.get_device_name(0))'
python -m pip freeze > /home/elkalt/wmac-4b-together-requirements.txt
python reports/wmac-4b-together/wmac_together.py selftest
python reports/wmac-4b-together/artifact.py selftest
phase=smoke
printf 'phase=smoke\n' >> /home/elkalt/wmac-4b-together.status
python reports/wmac-4b-together/smoke.py
phase=full
printf 'phase=full\n' >> /home/elkalt/wmac-4b-together.status
python reports/wmac-4b-together/wmac_together.py run --prepared /home/elkalt/prepared.json --model 4b --dtype float16 --out /home/elkalt/wmac-4b-together --conditions blocked,full

#!/usr/bin/env bash
set -euo pipefail
cd /home/elkalt/latent-dispatch-thinking
exec >> /home/elkalt/thinking-experiment.log 2>&1
finish() {
  code=$?
  trap - EXIT
  printf 'exit_code=%s\nfinished_at=%s\n' "$code" "$(date -Is)" > /home/elkalt/thinking-experiment.status.tmp
  mv /home/elkalt/thinking-experiment.status.tmp /home/elkalt/thinking-experiment.status
  if [ "$code" = 0 ]; then outcome=COMPLETED; else outcome=FAILED; fi
  printf 'LATENT_MASKING_JOB_RESULT outcome=%s exit_code=%s experiment=thinking\n' "$outcome" "$code" | sudo tee /dev/ttyS0 || true
  # Leave ten minutes for the supervising chat to download and push artifacts.
  sudo shutdown -h +10
  exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'state=running\nstarted_at=%s\n' "$(date -Is)" > /home/elkalt/thinking-experiment.status
export PYTHONUNBUFFERED=1
PYTHON=/home/elkalt/latent-dispatch-masking-pr1/.venv/bin/python
# Model code is unchanged: fast tests and all 18 FP16 real-model checks passed
# on this VM before the initial smoke. The revised generation caps are smoke-tested below.
"$PYTHON" scripts/run_thinking_pilot.py

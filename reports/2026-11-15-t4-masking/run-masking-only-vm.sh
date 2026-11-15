#!/usr/bin/env bash
set -euo pipefail
cd /home/elkalt/latent-dispatch-masking-pr1
source .venv/bin/activate
export PYTHONUNBUFFERED=1
exec >> /home/elkalt/masking-experiment.log 2>&1
finish() {
  code=$?
  printf 'exit_code=%s\nfinished_at=%s\n' "$code" "$(date -Is)" > /home/elkalt/masking-experiment.status.tmp
  mv /home/elkalt/masking-experiment.status.tmp /home/elkalt/masking-experiment.status
  echo "=== Masking-only experiment exited with $code at $(date -Is) ==="
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'state=running\nscope=masking_only_pilot\nstarted_at=%s\n' "$(date -Is)" > /home/elkalt/masking-experiment.status.tmp
mv /home/elkalt/masking-experiment.status.tmp /home/elkalt/masking-experiment.status
echo "=== Masking-only pilot starting $(date -Is) ==="
python /home/elkalt/run-masking-only.py

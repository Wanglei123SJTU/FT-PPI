#!/bin/bash
set -euo pipefail

# Print sbatch GPU arguments for the best currently idle Hyak GPU type.
# Preference is intentionally based on throughput, not on queue fairness.
# For Slurm arrays, HYAK_GPU_MIN_IDLE can be set so one selected GPU type can
# fill the desired task parallelism without falling back to low-memory cards.

MIN_IDLE="${HYAK_GPU_MIN_IDLE:-1}"
ALLOW_LOW_MEMORY="${HYAK_ALLOW_LOW_MEMORY_GPU:-0}"

if [ -n "${HYAK_GPU_ARGS:-}" ]; then
  printf '%s\n' "$HYAK_GPU_ARGS"
  echo "selected_gpu_args=$HYAK_GPU_ARGS reason=HYAK_GPU_ARGS" >&2
  exit 0
fi

idle_gpu_count() {
  local partition="$1"
  local gpu_type="$2"
  sinfo -h -p "$partition" -o "%G|%D|%t" 2>/dev/null | awk -F'|' -v gpu="gpu:${gpu_type}:" '
    index($1, gpu) > 0 && $3 ~ /^idle/ {
      per_node = 0
      n = split($1, parts, ":")
      if (n >= 3) {
        per_node = parts[3] + 0
      }
      total += ($2 + 0) * per_node
    }
    END { print total + 0 }
  '
}

choose_from_idle() {
  local partition="$1"
  local gres="$2"
  local gpu_type="$3"
  local count
  count="$(idle_gpu_count "$partition" "$gpu_type")"
  if [ "$count" -ge "$MIN_IDLE" ]; then
    printf -- '--partition=%s --gres=%s\n' "$partition" "$gres"
    echo "selected_gpu_partition=$partition selected_gres=$gres idle_gpu_count=$count min_idle=$MIN_IDLE reason=idle_$gpu_type" >&2
    exit 0
  fi
}

choose_from_idle "ckpt-g2" "gpu:h200:1" "h200"
choose_from_idle "ckpt-g2" "gpu:l40s:1" "l40s"
choose_from_idle "ckpt-g2" "gpu:l40:1" "l40"
choose_from_idle "ckpt" "gpu:h200:1" "h200"
choose_from_idle "ckpt" "gpu:a100:1" "a100"
choose_from_idle "ckpt" "gpu:l40s:1" "l40s"
choose_from_idle "ckpt" "gpu:l40:1" "l40"
choose_from_idle "ckpt" "gpu:a40:1" "a40"

if [ "$ALLOW_LOW_MEMORY" = "1" ]; then
  choose_from_idle "gpu-rtx6k" "gpu:rtx6k:1" "rtx6k"
  choose_from_idle "ckpt" "gpu:rtx6k:1" "rtx6k"
  choose_from_idle "gpu-2080ti" "gpu:2080ti:1" "2080ti"
  choose_from_idle "ckpt" "gpu:2080ti:1" "2080ti"
fi

printf '%s\n' '--partition=ckpt --gres=gpu:a40:1'
echo "selected_gpu_args=--partition=ckpt --gres=gpu:a40:1 min_idle=$MIN_IDLE reason=no_high_memory_idle_gpu" >&2

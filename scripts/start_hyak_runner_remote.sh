#!/bin/bash
set -euo pipefail

BRANCH="${1:-main}"
POLL_SECONDS="${2:-60}"
ONCE="${3:-0}"

mkdir -p .hyak_runner
RUNNER_LOG=".hyak_runner/runner.out"
RUNNER_PID=".hyak_runner/runner.pid"
RUNNER_REPO_DIR="${HYAK_RUNNER_REPO_DIR:-$(pwd)}"

runner_is_alive() {
  if [ ! -s "$RUNNER_PID" ]; then
    return 1
  fi
  pid="$(cat "$RUNNER_PID")"
  if ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  ps -p "$pid" -o args= 2>/dev/null | grep -q "scripts/hyak_runner.sh"
}

if runner_is_alive; then
  echo "detached runner already running pid=$(cat "$RUNNER_PID")"
else
  echo "starting detached runner"
  nohup env \
    HYAK_RUNNER_REPO_DIR="$RUNNER_REPO_DIR" \
    HYAK_RUNNER_BRANCH="$BRANCH" \
    HYAK_RUNNER_POLL_SECONDS="$POLL_SECONDS" \
    HYAK_RUNNER_ONCE="$ONCE" \
    bash scripts/hyak_runner.sh >> "$RUNNER_LOG" 2>&1 < /dev/null &
  echo "$!" > .hyak_runner/launcher.pid
fi

echo "remote runner log: $RUNNER_LOG"
sleep 2
tail -n 120 -f "$RUNNER_LOG"

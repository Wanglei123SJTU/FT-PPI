#!/bin/bash
set -uo pipefail

REPO_DIR="${HYAK_RUNNER_REPO_DIR:-$HOME/FT-PPI}"
BRANCH="${HYAK_RUNNER_BRANCH:-main}"
TASK_DIR="${HYAK_RUNNER_TASK_DIR:-hyak_tasks}"
STATE_DIR="${HYAK_RUNNER_STATE_DIR:-.hyak_runner}"
POLL_SECONDS="${HYAK_RUNNER_POLL_SECONDS:-60}"
ONCE="${HYAK_RUNNER_ONCE:-0}"
STOP_ON_FAILURE="${HYAK_RUNNER_STOP_ON_FAILURE:-0}"

ts() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

log() {
  printf '[%s] %s\n' "$(ts)" "$*"
}

finish() {
  status=$?
  log "runner exiting with status $status"
}

trap finish EXIT

cd "$REPO_DIR" || {
  echo "Could not cd to REPO_DIR=$REPO_DIR" >&2
  exit 1
}

mkdir -p "$STATE_DIR/done" "$STATE_DIR/failed" "$STATE_DIR/running" "$STATE_DIR/logs"
printf '%s\n' "$$" > "$STATE_DIR/runner.pid"

log "runner started"
log "repo=$REPO_DIR branch=$BRANCH task_dir=$TASK_DIR poll_seconds=$POLL_SECONDS"

pull_repo() {
  log "pulling origin/$BRANCH"
  git fetch origin "$BRANCH" --prune
  git pull --ff-only origin "$BRANCH"
  log "at commit $(git rev-parse --short HEAD)"
}

ensure_env() {
  if [ -x ".venv-hyak/bin/python" ]; then
    log "environment present at .venv-hyak"
    return 0
  fi

  log "environment missing; running scripts/setup_hyak_env.sh"
  bash scripts/setup_hyak_env.sh
}

mark_done() {
  task_id="$1"
  status="$2"
  {
    echo "task_id=$task_id"
    echo "status=$status"
    echo "finished_at=$(ts)"
    echo "commit=$(git rev-parse --short HEAD)"
  } > "$STATE_DIR/done/$task_id"
}

mark_failed() {
  task_id="$1"
  status="$2"
  {
    echo "task_id=$task_id"
    echo "status=$status"
    echo "failed_at=$(ts)"
    echo "commit=$(git rev-parse --short HEAD)"
  } > "$STATE_DIR/failed/$task_id"
}

run_task() {
  task_path="$1"
  task_id="$(basename "$task_path" .sh)"
  started="$(date +%Y%m%d_%H%M%S)"
  task_log="$STATE_DIR/logs/${task_id}_${started}.log"

  log "starting task $task_id from $task_path"
  log "task log: $task_log"
  printf '%s\n' "$(ts)" > "$STATE_DIR/running/$task_id"

  set +e
  (
    set -uo pipefail
    export HYAK_RUNNER_REPO_DIR="$REPO_DIR"
    export HYAK_RUNNER_STATE_DIR="$STATE_DIR"
    export HYAK_RUNNER_TASK_ID="$task_id"
    export HYAK_RUNNER_TASK_PATH="$task_path"
    export HYAK_RUNNER_TASK_LOG="$task_log"
    bash "$task_path"
  ) 2>&1 | tee "$task_log"
  status=${PIPESTATUS[0]}
  set +e

  rm -f "$STATE_DIR/running/$task_id"

  if [ "$status" -eq 0 ]; then
    mark_done "$task_id" "$status"
    log "task $task_id completed"
    return 0
  fi

  if [ "$status" -eq 99 ]; then
    mark_done "$task_id" "$status"
    log "task $task_id requested runner stop"
    exit 0
  fi

  mark_failed "$task_id" "$status"
  log "task $task_id failed with status $status"
  if [ "$STOP_ON_FAILURE" = "1" ]; then
    exit "$status"
  fi
  return 0
}

run_pending_tasks() {
  if [ ! -d "$TASK_DIR" ]; then
    log "task directory $TASK_DIR does not exist"
    return 0
  fi

  found=0
  while IFS= read -r task_path; do
    [ -n "$task_path" ] || continue
    task_id="$(basename "$task_path" .sh)"
    if [ -e "$STATE_DIR/done/$task_id" ] || [ -e "$STATE_DIR/failed/$task_id" ]; then
      continue
    fi
    if [ -e "$STATE_DIR/running/$task_id" ]; then
      log "skipping task $task_id because running marker exists"
      continue
    fi

    found=1
    run_task "$task_path"
  done < <(find "$TASK_DIR" -maxdepth 1 -type f -name '*.sh' | sort)

  if [ "$found" -eq 0 ]; then
    log "no pending tasks"
  fi
}

while true; do
  pull_repo || log "git pull failed; will retry"
  ensure_env || log "environment check/setup failed; will retry"
  run_pending_tasks

  if [ "$ONCE" = "1" ]; then
    log "HYAK_RUNNER_ONCE=1, stopping after one loop"
    exit 0
  fi

  log "sleeping ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done

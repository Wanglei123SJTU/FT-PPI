#!/bin/bash
set -euo pipefail

echo "export_full_grid_allocation_artifacts_task_start"

ROOT="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/wine_full_grid_allocation"
if [ ! -d "$ROOT" ]; then
  echo "missing output root: $ROOT" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$TMP_DIR/wine_full_grid_allocation/figures"

for file in \
  full_grid_cell_metrics.csv \
  full_grid_by_rho_summary.csv \
  full_grid_allocation_summary.csv
do
  if [ ! -s "$ROOT/$file" ]; then
    echo "missing or empty required file: $ROOT/$file" >&2
    exit 1
  fi
  cp "$ROOT/$file" "$TMP_DIR/wine_full_grid_allocation/$file"
done

for budget in 300 500 700 1000; do
  for stem in estimated_mean variance_objective; do
    for ext in pdf png; do
      src="$ROOT/figures/full_grid_allocation_B${budget}_${stem}.${ext}"
      if [ ! -s "$src" ]; then
        echo "missing or empty required figure: $src" >&2
        exit 1
      fi
      cp "$src" "$TMP_DIR/wine_full_grid_allocation/figures/"
    done
  done
done

tar_path="$TMP_DIR/wine_full_grid_allocation_artifacts.tar.gz"
tar -C "$TMP_DIR" -czf "$tar_path" wine_full_grid_allocation

echo "exported_files"
find "$TMP_DIR/wine_full_grid_allocation" -type f -printf "%s %P\n" | sort

echo "BEGIN_FULL_GRID_ARTIFACT_TARBALL_BASE64"
base64 -w 0 "$tar_path"
echo
echo "END_FULL_GRID_ARTIFACT_TARBALL_BASE64"

echo "export_full_grid_allocation_artifacts_task_done"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 OUTPUT_DIR [COMMIT_MESSAGE]" >&2
  exit 1
fi

output_dir="$1"
commit_message="${2:-Update replication progress}"

if [[ ! -d "$output_dir" ]]; then
  exit 0
fi

shopt -s nullglob
artifacts=(
  "$output_dir/replication_results.json"
  "$output_dir/replication_report.md"
  "$output_dir/claim_matrix.csv"
  "$output_dir"/*.png
)
shopt -u nullglob

existing=()
for path in "${artifacts[@]}"; do
  if [[ -f "$path" ]]; then
    existing+=("$path")
  fi
done

if [[ ${#existing[@]} -eq 0 ]]; then
  exit 0
fi

git add -f "${existing[@]}"

if git diff --cached --quiet -- "${existing[@]}"; then
  exit 0
fi

git commit -m "$commit_message"
git push origin main

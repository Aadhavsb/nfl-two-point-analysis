#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/bhara/repos/nfl-two-point-analysis"
log_dir="$project_dir/logs"
log_file="$log_dir/build.log"

mkdir -p "$log_dir"
cd "$project_dir"
printf '\n===== build started %s =====\n' "$(date --iso-8601=seconds)" >> "$log_file"
/home/bhara/.local/bin/uv run nfl-two-point build --config config/analysis.yaml >> "$log_file" 2>&1
printf '===== build succeeded %s =====\n' "$(date --iso-8601=seconds)" >> "$log_file"

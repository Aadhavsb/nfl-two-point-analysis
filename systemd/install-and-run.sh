#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/bhara/repos/nfl-two-point-analysis"
install -m 0644 "$project_dir/systemd/nfl-two-point-build.service" /etc/systemd/system/nfl-two-point-build.service
systemctl daemon-reload
systemctl start nfl-two-point-build.service
systemctl status nfl-two-point-build.service --no-pager

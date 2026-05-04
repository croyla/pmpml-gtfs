#!/usr/bin/env bash
set -euo pipefail
echo ""
echo "Starting generator..."
poetry run python gtfs_parallel.py > debug_0.log
poetry run python gtfs_compat.py > debug_1.log
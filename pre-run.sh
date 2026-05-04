#!/usr/bin/env bash
set -euo pipefail
poetry install
rm -rf tmp/*
rm -rf gtfs_pmpml/*
rm -f ./*.log
#!/usr/bin/env bash
set -euo pipefail
# Run features.py with project's virtualenv Python so imports resolve correctly
./myvenv/bin/python -m src.models.features

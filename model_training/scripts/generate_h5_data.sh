#!/usr/bin/env bash
set -euo pipefail

python -m model_training.generate_h5_data "$@"

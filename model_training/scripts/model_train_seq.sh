#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if (( NPROC_PER_NODE > 1 )); then
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m model_training.model_train_seq "$@"
else
    python -m model_training.model_train_seq "$@"
fi

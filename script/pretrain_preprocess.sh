#!/usr/bin/env bash
# Preprocess one pre-training configuration without selecting a CUDA device.
#
# Usage:
#   bash script/pretrain_preprocess.sh --config CONFIG [CONFIG ...] [--set KEY=VALUE ...]
#
# Inputs:
#   --config       One or more tracked configuration layers.
#                 Include ignored local overlays after tracked configurations.
#   --set          Repeatable configuration override.
#
# Output:
#   Captures terminal output under the repository logs/preprocess directory.
#   Preprocesses included_datasets and held_out_evaluation_datasets.
#   Held-out metadata is separate from train/validation/test partitions.
#   Dataset statistics are saved beside terminal.log in dataset_summary.json.
#   Processed data and metadata locations come from the resolved invocation.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/launch_common.sh"

require_config_argument "$@"
create_terminal_log "preprocess"
cd "${PROJECT_ROOT}"

log_config_paths "$@"
unset CUDA_VISIBLE_DEVICES
run_with_terminal_log "${PYTHON_BIN}" -m factory.process "$@"
move_terminal_log "complete"
if [[ -f "${TERMINAL_LOG_DIRECTORY}/dataset_summary.json" ]]; then
    write_terminal_log_message \
        "Dataset summary: ${TERMINAL_LOG_DIRECTORY}/dataset_summary.json"
fi

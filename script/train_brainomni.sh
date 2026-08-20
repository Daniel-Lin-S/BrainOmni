#!/usr/bin/env bash
# Launch Stage-2 BrainOmni pre-training on the least-busy GPUs.
#
# Usage:
#   bash script/train_brainomni.sh --num-gpus N --config CONFIG [CONFIG ...] [--set KEY=VALUE ...]
#
# Inputs:
#   --num-gpus     Required DeepSpeed world size and selected device count.
#   --config       One or more tracked configuration layers.
#                 Include ignored local overlays after tracked configurations.
#   --set          Repeatable configuration override, including the required
#                  invocation.tokenizer_path Stage-1 handoff when needed.
#
# Output:
#   Selects GPUs by free VRAM, utilization, then device index; writes terminal
#   and trainer text logs below the repository logs directory. Checkpoints,
#   TensorBoard events, metrics, and sidecars use invocation.output_root.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/launch_common.sh"

num_gpus=""
arguments=()
while (( $# )); do
    case "$1" in
        --num-gpus)
            (( $# >= 2 )) || fail "--num-gpus requires a value."
            num_gpus="$2"
            shift 2
            ;;
        *)
            arguments+=("$1")
            shift
            ;;
    esac
done

[[ -n "${num_gpus}" ]] || fail "--num-gpus is required."
require_config_argument "${arguments[@]}"
select_vacant_gpus "${num_gpus}"
create_terminal_log "brainomni"
cd "${PROJECT_ROOT}"

log_config_paths "${arguments[@]}"
run_with_terminal_log \
    printf 'Selected CUDA devices: %s\n' "${CUDA_VISIBLE_DEVICES}"
run_with_terminal_log deepspeed --num_gpus="${num_gpus}" \
    --module brainomni.launcher "${arguments[@]}"

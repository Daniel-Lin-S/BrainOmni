#!/usr/bin/env bash
# Validate or repair completed BrainTokenizer or BrainOmni portable weights.
#
# Usage:
#   bash script/repair_pretrain_campaign.sh --campaign-root CAMPAIGN [--check-only]
#
# Inputs:
#   --campaign-root  Semantic campaign directory containing campaign metadata.
#   --check-only     Validate without attempting reconstruction from checkpoint.
#
# Output:
#   Prints the absolute verified campaign and portable-weight identities.
#   Repair never launches training or selects a CUDA device.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/launch_common.sh"

campaign_root=""
arguments=()
while (( $# )); do
    case "$1" in
        --campaign-root)
            (( $# >= 2 )) || fail \
                "--campaign-root requires an absolute or relative directory."
            campaign_root="$2"
            shift 2
            ;;
        *)
            arguments+=("$1")
            shift
            ;;
    esac
done

[[ -n "${campaign_root}" ]] || fail \
    "--campaign-root is required. Pass the semantic campaign directory."
cd "${PROJECT_ROOT}"
unset CUDA_VISIBLE_DEVICES
"${PYTHON_BIN}" -m factory.campaign_health \
    --campaign-root "${campaign_root}" "${arguments[@]}"

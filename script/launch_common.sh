#!/usr/bin/env bash

# Shared validation, logging, and GPU selection for pre-training launchers.
set -o pipefail

readonly SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIRECTORY}/.." && pwd)"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

require_config_argument() {
    local argument
    local found_config=false
    local next_is_config=false

    for argument in "$@"; do
        if [[ "${next_is_config}" == true ]]; then
            [[ "${argument}" != --* ]] || fail \
                "--config must be followed by at least one configuration file."
            found_config=true
            next_is_config=false
        elif [[ "${argument}" == "--config" ]]; then
            next_is_config=true
        fi
    done
    [[ "${next_is_config}" == false ]] || fail \
        "--config must be followed by at least one configuration file."
    [[ "${found_config}" == true ]] || fail \
        "A --config argument is required."
}

require_positive_integer() {
    local value="$1"
    local name="$2"

    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail \
        "${name} must be a positive integer; got '${value}'."
}

create_terminal_log() {
    local stage="$1"
    local timestamp

    timestamp="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
    BRAINOMNI_ATTEMPT_ID="${timestamp}-$$"
    TERMINAL_LOG_DIRECTORY="${PROJECT_ROOT}/logs/${stage}/pending"
    TERMINAL_LOG_DIRECTORY+="/${BRAINOMNI_ATTEMPT_ID}"
    TERMINAL_LOG_PATH="${TERMINAL_LOG_DIRECTORY}/terminal.log"
    BRAINOMNI_TERMINAL_LOG_PATH="${TERMINAL_LOG_PATH}"
    export BRAINOMNI_ATTEMPT_ID
    export BRAINOMNI_TERMINAL_LOG_PATH
    mkdir -p "${TERMINAL_LOG_DIRECTORY}"
}

run_with_terminal_log() {
    printf 'Terminal log: %s\n' "${TERMINAL_LOG_PATH}" \
        | tee -a "${TERMINAL_LOG_PATH}"
    "$@" 2>&1 | tee -a "${TERMINAL_LOG_PATH}"
}

select_vacant_gpus() {
    local requested="$1"
    local inventory
    local index
    local free_memory
    local total_memory
    local utilization
    local records=""
    local -a device_ids=()
    local -a ordered_records=()

    require_positive_integer "${requested}" "--num-gpus"
    inventory="$("${NVIDIA_SMI_BIN}" \
        --query-gpu=index,memory.free,memory.total,utilization.gpu \
        --format=csv,noheader,nounits)" || fail \
        "Unable to query GPU availability with ${NVIDIA_SMI_BIN}."
    [[ -n "${inventory}" ]] || fail "GPU availability query returned no GPUs."

    while IFS=',' read -r index free_memory total_memory utilization; do
        index="${index//[[:space:]]/}"
        free_memory="${free_memory//[[:space:]]/}"
        total_memory="${total_memory//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "${index}" =~ ^[0-9]+$ ]] || fail \
            "GPU availability returned invalid index '${index}'."
        [[ "${free_memory}" =~ ^[0-9]+$ ]] || fail \
            "GPU ${index} returned invalid free memory '${free_memory}'."
        [[ "${total_memory}" =~ ^[0-9]+$ ]] || fail \
            "GPU ${index} returned invalid total memory '${total_memory}'."
        [[ "${utilization}" =~ ^[0-9]+$ ]] || fail \
            "GPU ${index} returned invalid utilization '${utilization}'."
        (( free_memory <= total_memory )) || fail \
            "GPU ${index} free memory exceeds total memory."
        (( utilization <= 100 )) || fail \
            "GPU ${index} utilization exceeds 100 percent."
        records+="${free_memory} ${utilization} ${index}"$'\n'
    done <<< "${inventory}"

    mapfile -t ordered_records < <(
        printf '%s' "${records}" | sort -k1,1nr -k2,2n -k3,3n
    )
    for index in "${ordered_records[@]}"; do
        device_ids+=("${index##* }")
        (( ${#device_ids[@]} == requested )) && break
    done
    (( ${#device_ids[@]} == requested )) || fail \
        "Requested ${requested} GPUs, but only ${#device_ids[@]} are available."

    CUDA_VISIBLE_DEVICES="$(IFS=,; printf '%s' "${device_ids[*]}")"
    export CUDA_VISIBLE_DEVICES
}

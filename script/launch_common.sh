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
    TERMINAL_LOG_STAGE="${stage}"
    TERMINAL_LOG_PATH="${TERMINAL_LOG_DIRECTORY}/terminal.log"
    BRAINOMNI_TERMINAL_LOG_PATH="${TERMINAL_LOG_PATH}"
    export BRAINOMNI_ATTEMPT_ID
    export BRAINOMNI_TERMINAL_LOG_PATH
    mkdir -p "${TERMINAL_LOG_DIRECTORY}"
}

write_terminal_log_message() {
    local message="$1"

    printf '%s\n' "${message}" >&2
    printf '%s\n' "${message}" >> "${TERMINAL_LOG_PATH}"
}

move_terminal_log() {
    local state="$1"
    local destination_directory
    local destination_path
    local source_directory
    local error_message

    error_message="Could not remove empty terminal-log staging directory:"
    source_directory="${TERMINAL_LOG_DIRECTORY}"
    if [[ -z "${TERMINAL_LOG_STAGE:-}" ]]; then
        fail "Cannot move a terminal log before creating it."
    fi
    [[ -f "${TERMINAL_LOG_PATH}" ]] || return 0
    destination_directory="${PROJECT_ROOT}/logs/${TERMINAL_LOG_STAGE}/${state}"
    destination_directory+="/${BRAINOMNI_ATTEMPT_ID}"
    destination_path="${destination_directory}/terminal.log"
    mkdir -p "${destination_directory}"
    mv "${TERMINAL_LOG_PATH}" "${destination_path}"
    if ! rmdir "${source_directory}"; then
        fail "${error_message} ${source_directory}"
    fi
    TERMINAL_LOG_DIRECTORY="${destination_directory}"
    TERMINAL_LOG_PATH="${destination_path}"
    BRAINOMNI_TERMINAL_LOG_PATH="${destination_path}"
    export BRAINOMNI_TERMINAL_LOG_PATH
    write_terminal_log_message "Terminal log: ${destination_path}"
}

log_config_paths() {
    local argument
    local reading_configs=false
    local resolved_path

    write_terminal_log_message "Configuration files, in precedence order:"
    for argument in "$@"; do
        if [[ "${argument}" == "--config" ]]; then
            reading_configs=true
            continue
        fi
        if [[ "${reading_configs}" == true ]]; then
            if [[ "${argument}" == --* ]]; then
                reading_configs=false
            else
                resolved_path="$(realpath -e "${argument}")" || fail \
                    "Could not resolve configuration file: ${argument}"
                write_terminal_log_message "  ${resolved_path}"
                continue
            fi
        fi
    done
}

filter_progress_bars() {
    local line

    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ "${line}" == *$'\r'* ]] || printf '%s\n' "${line}"
    done
}

run_with_terminal_log() {
    local log_path="${TERMINAL_LOG_PATH}"
    local status

    write_terminal_log_message "Terminal log: ${log_path}"
    if "$@" 2>&1 | tee /dev/stderr | filter_progress_bars >> "${log_path}"; then
        return 0
    else
        status=$?
    fi

    move_terminal_log "failed"
    return "${status}"
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

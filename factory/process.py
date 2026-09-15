from math import isfinite
from statistics import median
from collections import Counter
from datetime import datetime, timezone
from time import monotonic
import os
import json
import logging
import argparse
from typing import Any
import torch
from pathlib import Path
import random
import numpy as np
from factory.utils import (
    process,
    infer_signal_type,
    filter_channel,
    rename_channel,
    set_montage,
    extract_pos_sensor_type,
    get_sensor_type_mask,
    split_pretrain_metadata,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from accessor import DataAccessor
from pretrain_config import (
    ConfigError,
    load_pretrain_launch_config,
    metadata_directory,
    preprocessing_directory,
    resolve_dataset_identities,
    selected_data_catalog,
    sha256_file,
)

from factory.channel_selection import channel_selection_provenance

FINISH_SCHEMA_VERSION = 2
SNAPSHOT_SCHEMA_VERSION = 1
WINDOW_MODALITIES = ("eeg", "meg", "emeg")
CHANNEL_TYPES = ("eeg", "meg", "grad")
PROGRESS_INTERVAL_SECONDS = 30
SUMMARY_FILENAME = "dataset_summary.json"


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def describe_recording_channels(
    raw: Any,
    dataset: str,
    selection: dict[str, list[str]] | None = None,
) -> tuple[int, int, int]:
    """Return the retained EEG, magnetometer, and GRAD channel counts."""
    raw = rename_channel(raw, dataset)
    raw = filter_channel(raw, dataset, selection)
    raw = set_montage(raw, dataset)
    _, sensor_type = extract_pos_sensor_type(raw.info)
    eeg_mask, mag_mask, grad_mask, _ = get_sensor_type_mask(sensor_type)
    return (
        int(eeg_mask.sum()),
        int(mag_mask.sum()),
        int(grad_mask.sum()),
    )


def discover_catalog_recordings(
    accessor: DataAccessor,
    config: dict,
    completion_records: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Discover selected recordings, validating only uncached headers.

    Parameters
    ----------
    accessor : DataAccessor
        Recording discovery and raw-header reader.
    config : dict
        Resolved launch settings selecting catalog roots and exclusions.
    completion_records : list[dict[str, object]] or None, optional
        Versioned records from this preprocessing settings cache, default
        None. Channel selection must match before any cached file is reused.
        Legacy path-only markers require header validation for migration.

    Returns
    -------
    list[dict[str, object]]
        Discovered paths and dataset IDs; only uncached entries have freshly
        validated header fields. Completed entries are filtered before work
        submission. File discovery still detects newly added recordings.
    """
    cached_records = (
        completion_records if completion_records is not None else []
    )
    validate_cached_channel_selection(cached_records, config)
    cached = {
        (record["dataset"], str(Path(record["recording_path"]).resolve()))
        for record in cached_records
    }
    catalog = selected_data_catalog(config, include_held_out=True)
    recordings: list[dict[str, object]] = []
    logger = logging.getLogger("processor")
    for dataset, definition in catalog.items():
        root = Path(definition["path"]).resolve()
        if not root.is_dir():
            raise ConfigError(f"Data catalog root is not a directory: {root}")
        selection = channel_selection_provenance(dataset, definition)
        logger.info("dataset=%s: discovering recordings in %s", dataset, root)
        dataset_files = accessor.search_brain_files(str(root), dataset)
        if not dataset_files:
            raise ConfigError(
                f"No supported recordings exist below data catalog root: {root}"
            )
        cached_files = [
            row for row in dataset_files
            if (dataset, str(Path(row["path"]).resolve())) in cached
        ]
        recordings.extend(cached_files)
        dataset_files = [
            row for row in dataset_files
            if (dataset, str(Path(row["path"]).resolve())) not in cached
        ]
        if cached_files:
            logger.info(
                "dataset=%s: skipped header validation for %d already "
                "complete recordings; %d new recordings to validate",
                dataset, len(cached_files), len(dataset_files),
            )
        if not dataset_files:
            continue
        logger.info(
            "dataset=%s: validating %d recording headers",
            dataset, len(dataset_files),
        )
        removed_compensation = 0
        last_progress = monotonic()
        for index, recording in enumerate(dataset_files, start=1):
            raw = None
            try:
                raw = accessor.read_brain_file(
                    recording["path"],
                    preload=False,
                )
                raw_sample_rate_hz = float(raw.info["sfreq"])
                raw_samples = int(raw.n_times)
                prior_comps = len(raw.info.get("comps", []))
                channel_counts = describe_recording_channels(
                    raw, dataset, selection,
                )
                removed_compensation += bool(
                    prior_comps and not raw.info.get("comps", [])
                )
                observed = infer_signal_type(raw, dataset, selection)
                if (
                    not isfinite(raw_sample_rate_hz)
                    or raw_sample_rate_hz <= 0
                    or raw_samples <= 0
                ):
                    raise ConfigError(
                        "Recording has an invalid sample count or rate: "
                        f"{Path(recording['path']).resolve()}."
                    )
            except Exception as error:
                raise ConfigError(
                    f"Could not validate {recording['path']} from {dataset}: "
                    f"{error}"
                ) from error
            finally:
                close = getattr(raw, "close", None)
                if callable(close):
                    close()
            if observed != definition["signal_type"]:
                raise ConfigError(
                    f"Dataset {dataset} declares {definition['signal_type']}, "
                    f"but {recording['path']} retains {observed} channels."
                )
            recording["channel_selection"] = selection
            recording["dataset_root"] = str(root)
            recording["signal_type"] = observed
            recording["raw_sample_rate_hz"] = raw_sample_rate_hz
            recording["raw_samples"] = raw_samples
            (
                recording["eeg_channels"],
                recording["meg_channels"],
                recording["grad_channels"],
            ) = channel_counts
            recordings.append(recording)
            if monotonic() - last_progress >= PROGRESS_INTERVAL_SECONDS:
                logger.info(
                    "dataset=%s: validated headers %d/%d",
                    dataset, index, len(dataset_files),
                )
                last_progress = monotonic()
        logger.info(
            "dataset=%s: header validation complete (%d recordings)",
            dataset, len(dataset_files),
        )
        if removed_compensation:
            logger.info(
                "dataset=%s: channel selection removed unusable compensation "
                "metadata from %d recording headers after excluding required "
                "reference channels",
                dataset, removed_compensation,
            )
    return recordings



def validate_cached_channel_selection(
    records: list[dict[str, object]], config: dict,
) -> None:
    """Reject cached recordings with different or unrecorded exclusions.

    Parameters
    ----------
    records : list[dict[str, object]]
        Completion records with dataset IDs and channel_selection mappings.
    config : dict
        Resolved launch configuration containing the selected catalog entries.
    """
    catalog = selected_data_catalog(config, include_held_out=True)
    for record in records:
        dataset = record["dataset"]
        if dataset not in catalog:
            continue
        expected = channel_selection_provenance(dataset, catalog[dataset])
        observed = record.get("channel_selection")
        if observed != expected:
            raise ConfigError(
                f"Cached channel selection for {dataset} is {observed!r}, "
                f"but the catalog requests {expected!r}. Use fresh "
                "processed_root and metadata_root directories, or migrate "
                "completion provenance only after verifying the old settings."
            )


def read_finish_records(
    finish_data: object,
) -> tuple[list[dict[str, object]], list[str]]:
    """Read versioned completion records or legacy raw-path markers."""
    if isinstance(finish_data, list):
        if not all(isinstance(path, str) and path for path in finish_data):
            raise ConfigError(
                "Legacy finish.json must contain non-empty paths."
            )
        return [], sorted(set(finish_data))
    if not isinstance(finish_data, dict):
        raise ConfigError("finish.json must be a list or a mapping.")
    if finish_data.get("schema_version") != FINISH_SCHEMA_VERSION:
        raise ConfigError(
            "finish.json has an unsupported schema_version. Rerun "
            "preprocessing after preserving the existing metadata."
        )
    records = finish_data.get("recordings")
    if not isinstance(records, list):
        raise ConfigError("finish.json recordings must be a list.")
    normalized: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ConfigError("finish.json recording entries must be mappings.")
        recording_path = record.get("recording_path")
        if not isinstance(recording_path, str) or not recording_path:
            raise ConfigError(
                "finish.json recording entries need a recording_path."
            )
        normalized.append(record)
    return normalized, []


def finish_payload(records: list[dict[str, object]]) -> dict[str, object]:
    """Return the versioned completion metadata persisted after processing."""
    return {
        "schema_version": FINISH_SCHEMA_VERSION,
        "recordings": sorted(
            records,
            key=lambda record: str(record["recording_path"]),
        ),
    }


def build_dataset_snapshots(
    completion_records: list[dict[str, object]],
    metadata_list: list[dict[str, object]],
) -> dict[str, object]:
    """Build aggregate and per-dataset preprocessing snapshots."""
    records_by_dataset: dict[str, dict[str, dict[str, object]]] = {}
    for record in completion_records:
        dataset = record.get("dataset")
        source = record.get("source_recording")
        raw_duration = record.get("raw_duration_seconds")
        preprocessed_duration = record.get("preprocessed_duration_seconds")
        windows = record.get("generated_windows")
        if not isinstance(dataset, str) or not dataset:
            raise ConfigError("Completion metadata has no dataset identity.")
        if not isinstance(source, str) or not source:
            raise ConfigError("Completion metadata has no source recording ID.")
        if (
            not isinstance(raw_duration, (float, int))
            or not isfinite(raw_duration)
            or raw_duration <= 0
        ):
            raise ConfigError(
                "Completion metadata has an invalid raw duration."
            )
        if (
            not isinstance(preprocessed_duration, (float, int))
            or not isfinite(preprocessed_duration)
            or preprocessed_duration <= 0
        ):
            raise ConfigError(
                "Completion metadata has an invalid preprocessed duration."
            )
        if not isinstance(windows, int) or windows < 0:
            raise ConfigError(
                "Completion metadata has an invalid window count."
            )
        dataset_records = records_by_dataset.setdefault(dataset, {})
        if source in dataset_records:
            raise ConfigError(
                f"Duplicate completion metadata for {dataset}/{source}."
            )
        dataset_records[source] = record
    if not records_by_dataset:
        raise ConfigError(
            "No completed recordings are available for snapshots."
        )
    windows_by_dataset: dict[str, list[dict[str, object]]] = {
        dataset: [] for dataset in records_by_dataset
    }
    for metadata in metadata_list:
        dataset = metadata.get("dataset")
        source = metadata.get("source_recording")
        if (
            not isinstance(dataset, str)
            or dataset not in records_by_dataset
        ):
            raise ConfigError(
                "Window metadata references a dataset without a completion "
                "record."
            )
        if (
            not isinstance(source, str)
            or source not in records_by_dataset[dataset]
        ):
            raise ConfigError(
                "Window metadata references an unknown source recording."
            )
        windows_by_dataset[dataset].append(metadata)
    datasets = {
        dataset: summarize_dataset_snapshot(
            records,
            windows_by_dataset[dataset],
        )
        for dataset, records in sorted(records_by_dataset.items())
    }
    aggregate_records = {
        f"{dataset}/{source}": record
        for dataset, records in records_by_dataset.items()
        for source, record in records.items()
    }
    aggregate_windows = [
        metadata
        for dataset in sorted(windows_by_dataset)
        for metadata in windows_by_dataset[dataset]
    ]
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "aggregate": summarize_dataset_snapshot(
            aggregate_records,
            aggregate_windows,
        ),
        "datasets": datasets,
    }


def summarize_dataset_snapshot(
    records: dict[str, dict[str, object]],
    windows: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize one dataset or the aggregate completed recordings."""
    source_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    channel_type_counts: Counter[str] = Counter()
    channel_counts: list[int] = []
    for window in windows:
        source = window.get("source_recording")
        if isinstance(source, str) and source not in records:
            dataset = window.get("dataset")
            if isinstance(dataset, str):
                source = f"{dataset}/{source}"
        if not isinstance(source, str) or source not in records:
            raise ConfigError(
                "Window metadata has an unknown source recording."
            )
        modality = window.get("window_modality")
        total_channels = window.get("channels")
        counts = {
            channel_type: window.get(f"{channel_type}_channels")
            for channel_type in CHANNEL_TYPES
        }
        if modality not in WINDOW_MODALITIES:
            raise ConfigError("Window metadata has an invalid window modality.")
        if not isinstance(total_channels, int) or total_channels <= 0:
            raise ConfigError("Window metadata has an invalid channel count.")
        if any(
            not isinstance(count, int) or count < 0
            for count in counts.values()
        ):
            raise ConfigError(
                "Window metadata has invalid channel-type counts."
            )
        if sum(counts.values()) != total_channels:
            raise ConfigError(
                "Window metadata channel-type counts do not match channels."
            )
        source_counts[source] += 1
        modality_counts[modality] += 1
        channel_type_counts.update(counts)
        channel_counts.append(total_channels)
    for source, record in records.items():
        expected_windows = record["generated_windows"]
        if source_counts[source] != expected_windows:
            raise ConfigError(
                f"Completion metadata window count disagrees for {source}."
            )
    total_windows = len(windows)
    total_channel_occurrences = sum(channel_counts)
    channel_distribution: dict[str, int | float | None]
    if channel_counts:
        channel_distribution = {
            "min": min(channel_counts),
            "max": max(channel_counts),
            "median": median(channel_counts),
        }
    else:
        channel_distribution = {"min": None, "max": None, "median": None}
    return {
        "completed_recordings": len(records),
        "raw_duration_seconds": sum(
            float(record["raw_duration_seconds"])
            for record in records.values()
        ),
        "preprocessed_duration_seconds": sum(
            float(record["preprocessed_duration_seconds"])
            for record in records.values()
        ),
        "generated_windows": total_windows,
        "window_channel_count": channel_distribution,
        "window_modality_proportions": {
            modality: modality_counts[modality] / total_windows
            if total_windows
            else None
            for modality in WINDOW_MODALITIES
        },
        "channel_proportions": {
            channel_type: channel_type_counts[channel_type]
            / total_channel_occurrences
            if total_channel_occurrences
            else None
            for channel_type in CHANNEL_TYPES
        },
    }


def migrate_legacy_completion_records(
    legacy_paths: list[str],
    discovered_recordings: list[dict[str, object]],
    metadata_list: list[dict[str, object]],
    processed_root: str,
    sample_rate_hz: float,
    logger: logging.Logger,
) -> list[dict[str, object]]:
    """Upgrade legacy path markers and their window metadata in place."""
    recordings_by_path = {
        str(Path(recording["path"]).resolve()): recording
        for recording in discovered_recordings
    }
    windows_by_directory: dict[Path, list[dict[str, object]]] = {}
    for metadata in metadata_list:
        metadata_path = metadata.get("path")
        if not isinstance(metadata_path, str) or not metadata_path:
            raise ConfigError("Legacy window metadata has no artifact path.")
        directory = Path(metadata_path).resolve().parent
        windows_by_directory.setdefault(directory, []).append(metadata)
    completions: list[dict[str, object]] = []
    for legacy_path in legacy_paths:
        recording = recordings_by_path.get(str(Path(legacy_path).resolve()))
        if recording is None:
            raise ConfigError(
                "Cannot migrate a legacy completion marker without its "
                f"configured recording: {Path(legacy_path).resolve()}."
            )
        dataset = recording.get("dataset")
        raw_sample_rate_hz = recording.get("raw_sample_rate_hz")
        raw_samples = recording.get("raw_samples")
        channel_counts = {
            channel_type: recording.get(f"{channel_type}_channels")
            for channel_type in CHANNEL_TYPES
        }
        if not isinstance(dataset, str) or not dataset:
            raise ConfigError("Discovered recording has no dataset identity.")
        if (
            not isinstance(raw_sample_rate_hz, float)
            or not isfinite(raw_sample_rate_hz)
            or raw_sample_rate_hz <= 0
            or not isinstance(raw_samples, int)
            or raw_samples <= 0
        ):
            raise ConfigError(
                "Discovered recording has invalid header timing metadata: "
                f"{Path(str(recording['path'])).resolve()}."
            )
        if any(
            not isinstance(count, int) or count < 0
            for count in channel_counts.values()
        ):
            raise ConfigError(
                "Discovered recording has invalid channel-count metadata: "
                f"{Path(str(recording['path'])).resolve()}."
            )
        channels = sum(channel_counts.values())
        if channels <= 0:
            raise ConfigError(
                "Discovered recording has no retained EEG or MEG channels: "
                f"{Path(str(recording['path'])).resolve()}."
            )
        raw_path = Path(str(recording["path"])).resolve()
        dataset_root = Path(str(recording["dataset_root"])).resolve()
        try:
            source_recording = raw_path.relative_to(dataset_root).as_posix()
        except ValueError as error:
            raise ConfigError(
                f"Recording {raw_path} is outside configured root "
                f"{dataset_root}."
            ) from error
        preprocessed_samples = max(
            int(round(sample_rate_hz / raw_sample_rate_hz * raw_samples)),
            1,
        )
        completion = {
            "recording_path": str(raw_path),
            "dataset": dataset,
            "source_recording": source_recording,
            "raw_duration_seconds": raw_samples / raw_sample_rate_hz,
            "preprocessed_duration_seconds": (
                preprocessed_samples / sample_rate_hz
            ),
        }
        output_directory = (
            Path(processed_root)
            / dataset
            / Path(source_recording).with_suffix("")
        ).resolve()
        windows = windows_by_directory.get(output_directory, [])
        for metadata in windows:
            if channels != int(metadata.get("channels", -1)):
                raise ConfigError(
                    "Legacy window channel metadata disagrees with its "
                    f"recording: {Path(str(metadata['path'])).resolve()}."
                )
            eeg_channels = channel_counts["eeg"]
            meg_channels = channel_counts["meg"]
            grad_channels = channel_counts["grad"]
            metadata.update(
                {
                    "source_recording": source_recording,
                    "raw_duration_seconds": completion[
                        "raw_duration_seconds"
                    ],
                    "preprocessed_duration_seconds": completion[
                        "preprocessed_duration_seconds"
                    ],
                    "window_modality": (
                        "eeg"
                        if eeg_channels == channels
                        else "meg"
                        if eeg_channels == 0
                        else "emeg"
                    ),
                    "eeg_channels": eeg_channels,
                    "meg_channels": meg_channels,
                    "grad_channels": grad_channels,
                }
            )
        completion["generated_windows"] = len(windows)
        completions.append(completion)
    logger.info(
        "migrated legacy completion metadata for %d recording(s).",
        len(completions),
    )
    return completions


def build_grouped_dataset_summary(
    records: list[dict[str, object]],
    windows: list[dict[str, object]],
    config: dict,
) -> dict[str, object]:
    """Summarize selected pretraining and evaluation datasets independently.

    Parameters
    ----------
    records : list[dict[str, object]]
        Completion records with source identities, durations and window counts.
    windows : list[dict[str, object]]
        Window metadata containing dataset, source and retained channel counts.
    config : dict
        Resolved configuration selecting training and held-out dataset IDs.

    Returns
    -------
    dict[str, object]
        Preprocessing settings and two sections, included_datasets and
        evaluation_datasets, each with aggregate and per-dataset statistics.
        An unrequested evaluation section has state=not_requested and no
        numerical aggregate. Unselected cached datasets are excluded.
    """
    groups = {
        "included_datasets": sorted(selected_data_catalog(config)),
        "evaluation_datasets": sorted(
            config["invocation"]["held_out_evaluation_datasets"]
        ),
    }
    if set(groups["included_datasets"]) & set(groups["evaluation_datasets"]):
        raise ConfigError("Pretraining and evaluation summary groups overlap.")
    result = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "preprocessing": config["campaign"]["data"]["preprocessing"],
    }
    for group, datasets in groups.items():
        if not datasets:
            result[group] = {
                "state": "not_requested", "aggregate": None, "datasets": {},
            }
            continue
        selected_records = [r for r in records if r["dataset"] in datasets]
        missing = set(datasets) - {r["dataset"] for r in selected_records}
        if missing:
            raise ConfigError(
                f"No completion records for {group}: {sorted(missing)}."
            )
        snapshots = build_dataset_snapshots(
            selected_records, [w for w in windows if w["dataset"] in datasets],
        )
        result[group] = {
            "state": "complete",
            "aggregate": snapshots["aggregate"],
            "datasets": snapshots["datasets"],
            "channel_selection": {
                name: channel_selection_provenance(
                    name, config["invocation"]["data_catalog"][name],
                ) for name in datasets
            },
        }
    return result


def write_dataset_summary(
    summary: dict[str, object], directory: Path,
) -> Path:
    """Write a finite JSON summary beside the run's terminal log.

    Parameters
    ----------
    summary : dict[str, object]
        Two-group dataset summary including actual preprocessing settings.
    directory : Path
        Run log directory. The launcher moves its entire contents on completion.

    Returns
    -------
    Path
        Absolute dataset_summary.json path.
    """
    text = json.dumps(summary, indent=4, allow_nan=False) + "\n"
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SUMMARY_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def select_pending_recordings(
    recordings: list[dict[str, object]], completed_paths: set[str],
    logger: logging.Logger,
) -> list[dict[str, object]]:
    """Select pending recordings and report cache reuse for each dataset.

    Parameters
    ----------
    recordings : list[dict[str, object]]
        Discovered recordings with dataset IDs and absolute paths; pending
        entries also contain validated header fields.
    completed_paths : set[str]
        Canonical paths already processed under verified settings.
    logger : logging.Logger
        Parent process logger receiving dataset-scoped status messages.

    Returns
    -------
    list[dict[str, object]]
        Only recordings needing signal preprocessing.
    """
    totals = Counter(row["dataset"] for row in recordings)
    pending = [row for row in recordings
               if str(Path(row["path"]).resolve()) not in completed_paths]
    pending_counts = Counter(row["dataset"] for row in pending)
    for dataset, total in sorted(totals.items()):
        count = pending_counts[dataset]
        if not count:
            logger.info(
                "dataset=%s: skipped signal preprocessing; all %d "
                "recordings already complete", dataset, total,
            )
        else:
            logger.info(
                "dataset=%s: %d recordings to preprocess; %d/%d already "
                "complete and skipped", dataset, count, total - count, total,
            )
    return pending


def get_logger():
    logger = logging.getLogger(name="processor")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
        "%H:%M:%S",
    )

    screenHandler = logging.StreamHandler()
    screenHandler.setLevel(logging.INFO)
    screenHandler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(screenHandler)

    return logger


def parse_arg():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    terminal = os.environ.get("BRAINOMNI_TERMINAL_LOG_PATH")
    log_directory = (
        Path(terminal).resolve().parent if terminal
        else Path(__file__).resolve().parents[1] / "logs" / "preprocess"
        / "manual" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"-{os.getpid()}"
        )
    )
    parser.add_argument("--summary-dir", type=Path, default=log_directory)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    config = load_pretrain_launch_config(args.config, args.overrides)
    preprocessing = config["campaign"]["data"]["preprocessing"]
    invocation = config["invocation"]
    TIME = preprocessing["segment_seconds"]
    STRIDE = preprocessing["stride_seconds"]
    max_workers = invocation["preprocess_workers"]
    logger = get_logger()
    logger.info("initializing accessor...")
    accessor = DataAccessor(read_only=False)

    # pretrain data part
    processed_pretrain_path = os.path.join(
        invocation["processed_root"],
        preprocessing_directory(config).name,
    )

    pretrain_metadata_path = str(metadata_directory(config))
    preprocessing_metadata_path = str(preprocessing_directory(config))

    os.makedirs(pretrain_metadata_path, exist_ok=True)
    os.makedirs(preprocessing_metadata_path, exist_ok=True)

    finish_path = os.path.join(
        preprocessing_metadata_path,
        "finish.json",
    )
    info_path = os.path.join(
        preprocessing_metadata_path,
        "info.json",
    )
    snapshot_path = os.path.join(
        preprocessing_metadata_path,
        "dataset_snapshots.json",
    )

    logger.info("loading archives...")
    finish_data: object = []
    if os.path.exists(finish_path):
        with open(finish_path, "r") as f:
            finish_data = json.load(f)
    completion_records, legacy_finish_paths = read_finish_records(
        finish_data
    )
    completed_paths = {
        str(Path(str(record["recording_path"])).resolve())
        for record in completion_records
    } | {str(Path(path).resolve()) for path in legacy_finish_paths}

    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            metadata_list = json.load(f)
    else:
        metadata_list = []
    logger.info("searching_brain_files...")
    brain_files = discover_catalog_recordings(
        accessor, config, completion_records,
    )
    if legacy_finish_paths:
        logger.info(
            "migrating %d legacy completed recording(s) from metadata...",
            len(legacy_finish_paths),
        )
        completion_records.extend(
            migrate_legacy_completion_records(
                legacy_finish_paths,
                brain_files,
                metadata_list,
                processed_pretrain_path,
                float(preprocessing["sample_rate_hz"]),
                logger,
            )
        )
        logger.info("migrated legacy completion metadata.")

    validate_cached_channel_selection(completion_records, config)
    logger.info("filtering brain files...")
    brain_files = select_pending_recordings(
        brain_files, completed_paths, logger,
    )

    if brain_files:
        logger.info("start processing...")
    else:
        logger.info("All selected recordings are cached; no signal processing.")
    counter = 0
    failures = []
    pending_counts = Counter(row["dataset"] for row in brain_files)
    finished_counts: Counter[str] = Counter()
    last_progress = {dataset: monotonic() for dataset in pending_counts}
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=get_logger,
    ) as executor:
        futures = {}
        for recording in brain_files:
            future = executor.submit(
                process,
                accessor,
                recording["path"],
                preprocessing["sample_rate_hz"],
                preprocessing["low_frequency_hz"],
                preprocessing["high_frequency_hz"],
                recording["dataset"],
                recording["dataset_root"],
                recording["signal_type"],
                processed_pretrain_path,
                TIME,
                STRIDE,
                recording["channel_selection"],
            )
            futures[future] = recording
        for future in as_completed(futures):
            try:
                segments_metadata, completion = future.result()
                metadata_list += segments_metadata
                completion_records.append(completion)
                counter += 1
                dataset = completion["dataset"]
                finished_counts[dataset] += 1
                done, total = finished_counts[dataset], pending_counts[dataset]
                if (
                    done == total
                    or monotonic() - last_progress[dataset]
                    >= PROGRESS_INTERVAL_SECONDS
                ):
                    logger.info(
                        "dataset=%s: signal preprocessing completed %d/%d "
                        "recordings", dataset, done, total,
                    )
                    last_progress[dataset] = monotonic()
                if counter % 1000 == 0:
                    with open(finish_path, "w") as f:
                        json.dump(finish_payload(completion_records), f)
                    with open(info_path, "w") as f:
                        json.dump(metadata_list, f)

            except Exception as error:
                recording_path = futures[future]["path"]
                failures.append((recording_path, str(error)))
                logger.error(
                    "dataset=%s: failed to preprocess %s: %s",
                    futures[future]["dataset"], Path(recording_path).resolve(),
                    error,
                )

    logger.info("finish processing ...")
    metadata_list = sorted(metadata_list, key=lambda x: x["path"])
    with open(finish_path, "w") as f:
        json.dump(finish_payload(completion_records), f)
    with open(info_path, "w") as f:
        json.dump(metadata_list, f)
    if failures:
        failed_paths = [str(Path(path).resolve()) for path, _ in failures]
        raise RuntimeError(
            f"Preprocessing failed for {len(failures)} recording(s): "
            f"{failed_paths}. Successful recording metadata was preserved at "
            f"{Path(info_path).resolve()}. Correct the reported loader or "
            "recording errors, then rerun the same preprocessing command."
        )
    snapshots = build_dataset_snapshots(completion_records, metadata_list)
    with open(snapshot_path, "w") as f:
        json.dump(snapshots, f, indent=4)

    config = resolve_dataset_identities(config)
    logger.info(
        "resolved preprocessing datasets: %s",
        config["campaign"]["data"]["included_datasets"],
    )

    seed_everything(seed=config["campaign"]["seed"])
    train, val, test, held_out = split_pretrain_metadata(
        metadata_list,
        config["campaign"]["data"]["split_ratios"],
        config["campaign"]["data"]["included_datasets"],
        invocation["held_out_evaluation_datasets"],
    )
    with open(os.path.join(pretrain_metadata_path, "train.json"), "w") as f:
        json.dump(train, f, indent=4)
    with open(os.path.join(pretrain_metadata_path, "val.json"), "w") as f:
        json.dump(val, f, indent=4)
    with open(os.path.join(pretrain_metadata_path, "test.json"), "w") as f:
        json.dump(test, f, indent=4)
    for dataset, dataset_metadata in held_out.items():
        with open(
            os.path.join(pretrain_metadata_path, f"{dataset}.json"), "w"
        ) as f:
            json.dump(dataset_metadata, f, indent=4)
    summary = build_grouped_dataset_summary(
        completion_records, metadata_list, config,
    )
    summary["source_metadata"] = {
        "finish_sha256": sha256_file(Path(finish_path)),
        "info_sha256": sha256_file(Path(info_path)),
    }
    summary_path = write_dataset_summary(summary, args.summary_dir)
    logger.info("Dataset summary saved to %s", summary_path)

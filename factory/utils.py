import os
import mne
import time
from pathlib import Path
import random
from typing import Any
import numpy as np
import torch
from constant import SAMPLE_RATE, LOW, HIGH
from factory.brain_constant import (
    EXCLUDE_DICT,
    RENAME_DICT,
    HPI_LIST,
    MONTAGE_DICT,
    CUSTOM_MONTAGE_DICT,
    SENSOR_TYPE_DICT,
)
from accessor import DataAccessor, write_torch_warpper

MNE_PREPROCESS_JOBS = 1


def filter_channel(raw, dataset: str):
    exclude = []
    if dataset in EXCLUDE_DICT.keys():
        exclude = list(EXCLUDE_DICT[dataset])

    for i in ["HEO", "VEO", "EKG", "EMG"]:
        if i in raw.info.ch_names and i not in exclude:
            exclude.append(i)

    if dataset == "Omega":
        indices = mne.pick_types(
            raw.info, meg=True, eeg=False, ref_meg=False, exclude=exclude
        )
    else:
        indices = mne.pick_types(
            raw.info, meg=True, eeg=True, ref_meg=False, exclude=exclude
        )
    raw.pick(indices)
    return raw


def infer_signal_type(raw: Any, dataset: str) -> str:
    """Return the retained EEG/MEG modality for one raw recording.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording whose channels are selected in place.
    dataset : str
        Dataset ID used by the shared channel-selection policy.

    Returns
    -------
    str
        ``"eeg"``, ``"meg"``, or ``"both"``.

    Raises
    ------
    ValueError
        Raised when channel selection retains neither EEG nor MEG channels.
    """
    filtered = filter_channel(raw, dataset)
    eeg_indices = mne.pick_types(filtered.info, eeg=True)
    meg_indices = mne.pick_types(
        filtered.info,
        meg=True,
        ref_meg=False,
    )
    has_eeg = bool(eeg_indices.size)
    has_meg = bool(meg_indices.size)
    if has_eeg and has_meg:
        return "both"
    if has_eeg:
        return "eeg"
    if has_meg:
        return "meg"
    raise ValueError(
        f"No EEG or MEG channels remain after filtering dataset {dataset}."
    )


def rename_channel(raw, dataset: str):
    if dataset in RENAME_DICT.keys():
        raw.rename_channels(RENAME_DICT[dataset])
    return raw


def set_montage(raw, dataset: str):
    if (
        dataset not in MONTAGE_DICT.keys()
        and dataset not in CUSTOM_MONTAGE_DICT.keys()
    ):
        return raw
    if dataset in CUSTOM_MONTAGE_DICT.keys():
        montage = mne.channels.read_custom_montage(CUSTOM_MONTAGE_DICT[dataset])
        raw.set_montage(montage)
        return raw
    montage = mne.channels.make_standard_montage(MONTAGE_DICT[dataset])
    raw.set_montage(montage)
    return raw


def extract_pos_sensor_type(info):
    """
    kind = {1(FIFFV_MEG_CH), 2(FIFFV_EEG_CH)}
    coil_type = {
        1(FIFFV_COIL_EEG),
        4001(FIFFV_COIL_MAGNES_MAG),
        3012(FIFFV_COIL_VV_PLANAR_T1),
        201609,                          #(AXIAL_GRAD)
        5001,                            #(AXIAL_GRAD)
        3022(FIFFV_COIL_VV_MAG_T1),
        3024(FIFFV_COIL_VV_MAG_T3),
        6001(FIFFV_COIL_KIT_GRAD),
    }
    """
    pos = []
    sensor_type = []
    # kind_dict = {1: "meg", 2: "eeg"}
    for i in info["chs"]:
        kind = int(i["kind"])
        assert kind in [1, 2], f"Unknown sensor kind:{i['kind']}"
        coil_type = str(i["coil_type"])
        # eeg
        if kind == 2:
            pos.append(np.hstack([i["loc"][:3], np.array([0.0, 0.0, 0.0])]))
            sensor_type.append(SENSOR_TYPE_DICT["EEG"])
        # meg
        else:
            xyz = i["loc"][:3]
            dir_idx = 3
            if "PLANAR" in coil_type:
                dir_idx = 1
            dir = i["loc"][3 * dir_idx : 3 * (dir_idx + 1)]
            pos.append(np.hstack([xyz, dir]))

            if "MAG" in coil_type:
                sensor_type.append(SENSOR_TYPE_DICT["MAG"])
            else:
                sensor_type.append(SENSOR_TYPE_DICT["GRAD"])

    pos = np.stack(pos).astype(np.float32)
    sensor_type = np.array(sensor_type).astype(np.int32)

    return pos, sensor_type


def get_sensor_type_mask(sensor_type: np.ndarray):
    eeg_mask = sensor_type == SENSOR_TYPE_DICT["EEG"]
    mag_mask = sensor_type == SENSOR_TYPE_DICT["MAG"]
    grad_mask = sensor_type == SENSOR_TYPE_DICT["GRAD"]
    meg_mask = mag_mask | grad_mask
    return eeg_mask, mag_mask, grad_mask, meg_mask


def _auto_detect_bad_channels(raw_data: mne.io.Raw, threshold: int = 10):
    spectrum = raw_data.compute_psd(
        tmax=1000000, average="mean", verbose=False
    )  # fmax
    data = spectrum.data + 1e-16
    ch_names = np.array(spectrum.ch_names)
    log_data = np.log(data)
    # Euclidean distance between channel pairs
    distances = np.linalg.norm(
        log_data[:, None, :] - log_data[None, :, :], axis=2
    )
    mean_distances = np.mean(distances, axis=1)

    # Use IQR (interquartile range) to identify outliers
    Q1 = np.percentile(mean_distances, 25)
    Q3 = np.percentile(mean_distances, 75)
    IQR = Q3 - Q1
    threshold_upper = Q3 + threshold * IQR
    threshold_lower = Q1 - threshold * IQR

    outlier_indices = np.where(
        (mean_distances > threshold_upper) | (mean_distances < threshold_lower)
    )[0]
    bad_channels = ch_names[outlier_indices].tolist()

    return bad_channels


def auto_detect_bad_channels(raw: mne.io.Raw, eeg_mask, mag_mask, grad_mask):
    bad_channels = []
    if eeg_mask.any():
        bad_channels += _auto_detect_bad_channels(
            raw.copy().pick(picks=mne.pick_types(raw.info, eeg=True))
        )
    return bad_channels


def filter_resample_preprocess(
    raw,
    dataset: str,
    sample_rate_hz: float,
    low_frequency_hz: float,
    high_frequency_hz: float,
):
    """Filter and resample one recording without nested worker pools."""
    notch_freqs = [50, 60]
    if notch_freqs:
        raw = raw.notch_filter(
            freqs=notch_freqs,
            n_jobs=MNE_PREPROCESS_JOBS,
            verbose=False,
        )
    if dataset in HPI_LIST:
        raw = mne.chpi.filter_chpi(raw, include_line=False, verbose=False)
    raw = raw.resample(
        sample_rate_hz,
        n_jobs=MNE_PREPROCESS_JOBS,
        verbose=False,
    )
    raw = raw.filter(
        low_frequency_hz,
        high_frequency_hz,
        n_jobs=MNE_PREPROCESS_JOBS,
        verbose=False,
    )
    return raw


def normalize_pos(pos: np.ndarray, eeg_mask, meg_mask):
    if eeg_mask.any():
        eeg_mean = np.mean(pos[eeg_mask, :3], axis=0, keepdims=True)
        pos[eeg_mask, :3] -= eeg_mean
        eeg_scale = np.sqrt(3 * np.mean(np.sum(pos[eeg_mask, :3] ** 2, axis=1)))
        pos[eeg_mask, :3] /= eeg_scale
    if meg_mask.any():
        meg_mean = np.mean(pos[meg_mask, :3], axis=0, keepdims=True)
        pos[meg_mask, :3] -= meg_mean
        meg_scale = np.sqrt(3 * np.mean(np.sum(pos[meg_mask, :3] ** 2, axis=1)))
        pos[meg_mask, :3] /= meg_scale
    return pos


def sensortype_wise_normalize(_data: np.ndarray, eeg_mask, mag_mask, grad_mask):
    # didn't do per channel z-score
    data = _data.copy()
    if eeg_mask.any():
        eeg_data = data[eeg_mask, :]
        eeg_mean = np.mean(
            eeg_data, axis=0, keepdims=True
        )  # reset virtual reference
        eeg_data = eeg_data - eeg_mean
        eeg_std = (
            np.std(eeg_data) + 1.0e-5
        )  # Preserve group magnitude relationships.
        data[eeg_mask, :] = eeg_data / (eeg_std)

    if mag_mask.any():
        mag_data = data[mag_mask, :]
        mag_mean = np.mean(mag_data, axis=0, keepdims=True)
        mag_data = mag_data - mag_mean
        mag_std = np.std(mag_data) + 1.0e-13
        data[mag_mask, :] = mag_data / mag_std

    if grad_mask.any():
        grad_data = data[grad_mask, :]
        grad_mean = np.mean(grad_data, axis=0, keepdims=True)
        grad_data = grad_data - grad_mean
        grad_std = np.std(grad_data) + 1.0e-13
        data[grad_mask, :] = grad_data / grad_std

    return data.astype(np.float32)


def accept_segment(seg_data: np.ndarray, pos: np.ndarray):
    bad = (np.isnan(seg_data).any()) | (np.isnan(pos).any())
    return ~bad


def split_to_segments_save(
    accessor: DataAccessor,
    data: np.ndarray,
    pos: np.ndarray,
    sensor_type: np.ndarray,
    eeg_mask: np.ndarray,
    mag_mask: np.ndarray,
    grad_mask: np.ndarray,
    meg_mask: np.ndarray,
    path: str,
    dataset: str,
    dataset_root: str,
    ready_path: str,
    raw_duration_seconds: float,
    preprocessed_duration_seconds: float,
    signal_type: str,
    sample_rate_hz: float,
    TIME: int,
    STRIDE: int,
):
    segments_metadata = []
    start = 0
    end = int(start + TIME * sample_rate_hz)
    stride_length = int(STRIDE * sample_rate_hz)
    root_path = Path(dataset_root).resolve()
    raw_path = Path(path).resolve()
    try:
        relative_path = raw_path.relative_to(root_path)
    except ValueError as error:
        raise ValueError(
            f"Recording {raw_path} is outside configured root {root_path}."
        ) from error
    brain_file_folder_path = str(
        Path(ready_path) / dataset / relative_path.with_suffix("")
    )
    accessor.mkdir(brain_file_folder_path)

    while end < data.shape[1]:
        seg_data = sensortype_wise_normalize(
            data[:, start:end], eeg_mask, mag_mask, grad_mask
        )
        if accept_segment(seg_data, pos):
            seg_data_path = os.path.join(
                brain_file_folder_path, f"{len(segments_metadata)}_data.pt"
            )
            seg_data_dict = {
                "x": torch.from_numpy(seg_data),
                "pos": torch.from_numpy(pos),
                "sensor_type": torch.from_numpy(sensor_type),
            }
            accessor.write(seg_data_dict, seg_data_path, write_torch_warpper)
            time.sleep(0.01)
            metadata = {
                "dataset": dataset,
                "path": seg_data_path,
                "channels": seg_data.shape[0],
                "signal_type": signal_type,
                "is_eeg": bool((sensor_type == SENSOR_TYPE_DICT["EEG"]).all()),
                "source_recording": relative_path.as_posix(),
                "raw_duration_seconds": raw_duration_seconds,
                "preprocessed_duration_seconds": (
                    preprocessed_duration_seconds
                ),
                "window_modality": (
                    "eeg"
                    if eeg_mask.all()
                    else "meg"
                    if meg_mask.all()
                    else "emeg"
                ),
                "eeg_channels": int(eeg_mask.sum()),
                "meg_channels": int(mag_mask.sum()),
                "grad_channels": int(grad_mask.sum()),
                "is_meg": bool(
                    (
                        (sensor_type == SENSOR_TYPE_DICT["MAG"])
                        | (sensor_type == SENSOR_TYPE_DICT["GRAD"])
                    ).all()
                ),
            }
            segments_metadata.append(metadata)
        start += stride_length
        end += stride_length
    return segments_metadata


def split_pretrain_metadata(
    data: list[dict[str, Any]],
    split_ratios: dict[str, float],
    training_datasets: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    """Split training metadata while isolating requested held-out datasets."""
    training_names = set(training_datasets)
    observed_names = {item["dataset"] for item in data}
    held_out_names = observed_names - training_names
    held_out = {
        dataset: [item for item in data if item["dataset"] == dataset]
        for dataset in sorted(held_out_names)
    }
    data = [item for item in data if item["dataset"] in training_names]
    if not data:
        raise ValueError(
            "No metadata remains for train, validation, and test splits after "
            "held-out datasets were removed. Include at least one training "
            "dataset and rerun preprocessing."
        )
    random.shuffle(data)
    N = len(data)
    train_end = int(N * split_ratios["train"])
    validation_end = train_end + int(N * split_ratios["validation"])
    train = data[:train_end]
    val = data[train_end:validation_end]
    test = data[validation_end:]
    return train, val, test, held_out


def process(
    accessor: DataAccessor,
    path: str,
    sample_rate_hz: float,
    low_frequency_hz: float,
    high_frequency_hz: float,
    dataset: str,
    dataset_root: str,
    signal_type: str,
    ready_path: str,
    TIME: int,
    STRIDE: int,
):
    raw = accessor.read_brain_file(path)
    raw = rename_channel(raw, dataset)
    raw = filter_channel(raw, dataset)
    raw = set_montage(raw, dataset)
    raw_duration_seconds = raw.n_times / raw.info["sfreq"]

    pos, sensor_type = extract_pos_sensor_type(raw.info)
    eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(sensor_type)
    pos = normalize_pos(pos, eeg_mask, meg_mask)

    raw = filter_resample_preprocess(
        raw,
        dataset,
        sample_rate_hz,
        low_frequency_hz,
        high_frequency_hz,
    )
    preprocessed_duration_seconds = raw.n_times / raw.info["sfreq"]

    bad_channels = auto_detect_bad_channels(raw, eeg_mask, mag_mask, grad_mask)
    if len(bad_channels) > 0:
        raw.info["bads"] += bad_channels
    if raw.info["bads"]:
        raw.interpolate_bads(
            reset_bads=True,
            mode="accurate",
            origin=(0.0, 0.0, 0.04),
            verbose=False,
        )

    data = raw.get_data()
    del raw
    # data float64 everything not normalized
    segments_metadata = split_to_segments_save(
        accessor,
        data,
        pos,
        sensor_type,
        eeg_mask,
        mag_mask,
        grad_mask,
        meg_mask,
        path,
        dataset,
        dataset_root,
        ready_path,
        raw_duration_seconds,
        preprocessed_duration_seconds,
        signal_type,
        sample_rate_hz,
        TIME,
        STRIDE,
    )
    root_path = Path(dataset_root).resolve()
    raw_path = Path(path).resolve()
    try:
        source_recording = raw_path.relative_to(root_path).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Recording {raw_path} is outside configured root {root_path}."
        ) from error
    completion = {
        "recording_path": str(raw_path),
        "dataset": dataset,
        "source_recording": source_recording,
        "raw_duration_seconds": raw_duration_seconds,
        "preprocessed_duration_seconds": preprocessed_duration_seconds,
        "generated_windows": len(segments_metadata),
    }
    return segments_metadata, completion

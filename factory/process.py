import os
import json
import logging
import argparse
import torch
from pathlib import Path
import random
import numpy as np
from factory.utils import (
    process,
    infer_signal_type,
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
)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discover_catalog_recordings(
    accessor: DataAccessor,
    config: dict,
) -> list[dict[str, str]]:
    """Discover and validate recordings from every selected catalog root."""
    catalog = selected_data_catalog(config)
    recordings: list[dict[str, str]] = []
    for dataset, definition in catalog.items():
        root = Path(definition["path"]).resolve()
        if not root.is_dir():
            raise ConfigError(f"Data catalog root is not a directory: {root}")
        dataset_files = accessor.search_brain_files(str(root), dataset)
        if not dataset_files:
            raise ConfigError(
                f"No supported recordings exist below data catalog root: {root}"
            )
        for recording in dataset_files:
            raw = None
            try:
                raw = accessor.read_brain_file(
                    recording["path"],
                    preload=False,
                )
                observed = infer_signal_type(raw, dataset)
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
            recording["dataset_root"] = str(root)
            recording["signal_type"] = observed
            recordings.append(recording)
    return recordings


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
    logger.addHandler(screenHandler)

    return logger


def parse_arg():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
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

    logger.info("searching_brain_files...")
    brain_files = discover_catalog_recordings(accessor, config)
    logger.info("loading archives...")
    if os.path.exists(finish_path):
        with open(finish_path, "r") as f:
            finish = json.load(f)
    else:
        finish = []

    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            metadata_list = json.load(f)
    else:
        metadata_list = []

    logger.info("filtering brain files...")
    brain_files = [i for i in brain_files if i["path"] not in finish]

    logger.info("start processing...")
    counter = 0
    failures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
            )
            futures[future] = recording["path"]
        for future in as_completed(futures):
            try:
                segments_metadata, finished_path = future.result()
                metadata_list += segments_metadata
                finish.append(finished_path)
                counter += 1
                if counter % 1000 == 0:
                    with open(finish_path, "w") as f:
                        json.dump(finish, f)
                    with open(info_path, "w") as f:
                        json.dump(metadata_list, f)

            except Exception as error:
                recording_path = futures[future]
                failures.append((recording_path, str(error)))
                logger.error(
                    "Failed to preprocess %s: %s",
                    Path(recording_path).resolve(),
                    error,
                )

    logger.info("finish processing ...")
    metadata_list = sorted(metadata_list, key=lambda x: x["path"])
    with open(finish_path, "w") as f:
        json.dump(finish, f)
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

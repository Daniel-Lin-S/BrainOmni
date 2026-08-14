import os
import json
import logging
import argparse
import torch
import random
import numpy as np
from factory.utils import (
    process,
    split_pretrain_metadata,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from accessor import DataAccessor
from pretrain_config import load_pretrain_config, metadata_directory

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-config", type=str)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    config = load_pretrain_config(
        args.config, args.local_config, args.overrides
    )
    preprocessing = config["campaign"]["data"]["preprocessing"]
    invocation = config["invocation"]
    TIME = preprocessing["segment_seconds"]
    STRIDE = preprocessing["stride_seconds"]
    max_workers = invocation["preprocess_workers"]
    selected_datasets = config["campaign"]["data"]["included_datasets"]
    logger = get_logger()
    logger.info("initializing accessor...")
    accessor = DataAccessor(read_only=False)

    # pretrain data part
    processed_pretrain_path = os.path.join(
        invocation["processed_root"],
        metadata_directory(config).name,
    )

    pretrain_metadata_path = str(metadata_directory(config))

    os.makedirs(
        pretrain_metadata_path,
        exist_ok=True,
    )

    finish_path = os.path.join(pretrain_metadata_path, "finish.json")
    info_path = os.path.join(pretrain_metadata_path, "info.json")

    logger.info("searching_brain_files...")
    brain_files = accessor.search_brain_files(root_path=invocation["raw_root"])
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
    if selected_datasets != ["*"]:
        brain_files = [
            item for item in brain_files if item["dataset"] in selected_datasets
        ]

    logger.info("start processing...")
    counter = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in brain_files:
            futures.append(
                executor.submit(
                    process,
                    accessor,
                    i["path"],
                    preprocessing["sample_rate_hz"],
                    preprocessing["low_frequency_hz"],
                    preprocessing["high_frequency_hz"],
                    i["dataset"],
                    processed_pretrain_path,
                    TIME,
                    STRIDE,
                )
            )
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

            except Exception as e:
                logger.info(f"An error occurred: {e}")

    logger.info("finish processing ...")
    metadata_list = sorted(metadata_list, key=lambda x: x["path"])
    with open(finish_path, "w") as f:
        json.dump(finish, f)
    with open(info_path, "w") as f:
        json.dump(metadata_list, f)

    seed_everything(seed=config["campaign"]["seed"])
    train, val, test, new_device_dataset_dict = split_pretrain_metadata(
        metadata_list,
        config["campaign"]["data"]["split_ratios"],
    )
    with open(os.path.join(pretrain_metadata_path, "train.json"), "w") as f:
        json.dump(train, f, indent=4)
    with open(os.path.join(pretrain_metadata_path, "val.json"), "w") as f:
        json.dump(val, f, indent=4)
    with open(os.path.join(pretrain_metadata_path, "test.json"), "w") as f:
        json.dump(test, f, indent=4)
    for dataset in new_device_dataset_dict.keys():
        with open(os.path.join(pretrain_metadata_path, f"{dataset}.json"), "w") as f:
            json.dump(new_device_dataset_dict[dataset], f, indent=4)

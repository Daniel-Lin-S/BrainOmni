import os
import json
import torch
import random
from constant import SEED
from constant import PRETRAIN_DTYPE
from accessor import DataAccessor, load_torch_warpper
from torch.utils.data import DataLoader, BatchSampler

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_list, accessor: DataAccessor):
        super().__init__()
        self.metadata_list = metadata_list
        self.accessor = accessor

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        data = self.accessor.read(
            self.metadata_list[idx]["path"],
            load_torch_warpper,
        )
        data["x"] = data["x"].to(PRETRAIN_DTYPE)
        data["pos"] = data["pos"].to(PRETRAIN_DTYPE)
        data["path"] = self.metadata_list[idx]["path"]
        dataset = self.metadata_list[idx].get("dataset")
        if dataset is not None:
            if not isinstance(dataset, str) or not dataset:
                raise ValueError(
                    "Pre-training metadata has an invalid dataset identity: "
                    f"{dataset!r}."
                )
            data["dataset"] = dataset
        return data


def collate_fn(batch):
    data = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            data[key] = torch.stack([i[key] for i in batch])
    data["path"] = [i["path"] for i in batch]
    if "dataset" in batch[0]:
        datasets = [item.get("dataset") for item in batch]
        if any(
            not isinstance(dataset, str) or not dataset
            for dataset in datasets
        ):
            raise ValueError(
                "Each batch item must provide one non-empty dataset identity."
            )
        data["dataset"] = datasets
    return data


def training_dataset_ids(metadata_path: str) -> tuple[str, ...]:
    """Return sorted training dataset identifiers from resolved metadata.

    Parameters
    ----------
    metadata_path : str
        Directory containing the resolved ``train.json`` metadata.

    Returns
    -------
    tuple[str, ...]
        Non-empty, sorted dataset identifiers used by the training split.
    """
    train_path = os.path.join(metadata_path, "train.json")
    with open(train_path, "r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, list) or not metadata:
        raise ValueError(
            f"Training metadata is empty: {os.path.abspath(train_path)}."
        )
    datasets = []
    for item in metadata:
        if not isinstance(item, dict):
            raise ValueError(
                "Training metadata must contain mappings, got "
                f"{type(item).__name__}."
            )
        dataset = item.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(
                "Training metadata requires non-empty dataset identities, got "
                f"{dataset!r}."
            )
        datasets.append(dataset)
    return tuple(sorted(set(datasets)))


def build_fixed_monitor_batch(
    metadata_path: str,
    accessor: DataAccessor,
    rank: int,
    world_size: int,
    batch_size: int,
) -> dict[str, torch.Tensor | list[str]]:
    """Build one deterministic channel-compatible validation batch per rank.

    Parameters
    ----------
    metadata_path : str
        Directory containing ``val.json`` split metadata.
    accessor : DataAccessor
        Read-only processed-tensor accessor.
    rank : int
        Distributed rank selecting one deterministic shard.
    world_size : int
        Number of distributed ranks.
    batch_size : int
        Maximum samples in the fixed per-rank micro-batch.

    Returns
    -------
    dict[str, torch.Tensor or list[str]]
        Collated CPU batch with equal channel counts across samples.
    """
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"Expected rank in [0, {world_size}), got {rank}."
        )
    if batch_size <= 0:
        raise ValueError(
            f"Fixed monitor batch size must be positive, got {batch_size}."
        )
    validation_path = os.path.join(metadata_path, "val.json")
    with open(validation_path, "r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, list) or not metadata:
        raise ValueError(
            f"Validation metadata is empty: {os.path.abspath(validation_path)}."
        )
    channel_groups: dict[int, list[dict]] = {}
    for item in metadata:
        if not isinstance(item, dict):
            raise ValueError(
                "Validation metadata must contain mappings, got "
                f"{type(item).__name__}."
            )
        channels = item.get("channels")
        path = item.get("path")
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(
                f"Validation metadata has invalid channel count {channels!r}."
            )
        if not isinstance(path, str) or not os.path.isabs(path):
            raise ValueError(
                "Validation metadata requires absolute processed paths, got "
                f"{path!r}."
            )
        channel_groups.setdefault(channels, []).append(item)
    selected = None
    for channels in sorted(channel_groups):
        group = sorted(channel_groups[channels], key=lambda item: item["path"])
        shard = group[rank::world_size]
        if not shard:
            shard = group
        selected = shard[:batch_size]
        break
    if not selected:
        raise ValueError(
            "Cannot form one fixed validation monitor batch per rank from "
            f"{os.path.abspath(validation_path)}."
        )
    dataset = BrainDataset(selected, accessor)
    return collate_fn([dataset[index] for index in range(len(dataset))])


class Bucket:
    def __init__(self, batch_size):
        self.data = []
        self.batch_size = batch_size

    def append(self, x):
        self.data.append(x)

    def shuffle(self):
        random.shuffle(self.data)

    def __len__(self):
        if len(self.data) % self.batch_size == 0:
            return len(self.data) // self.batch_size
        return len(self.data) // self.batch_size + 1

    def __iter__(self):
        for i in range(len(self)):
            yield (
                self.data[i * self.batch_size : (i + 1) * self.batch_size]
                if (i + 1) * self.batch_size <= len(self.data)
                else self.data[i * self.batch_size :]
            )


class BucketBatchSampler(BatchSampler):
    def __init__(self, channel_list, batch_size, rank):
        self.rank = rank
        channel_set = sorted(list(set(channel_list)))
        self.num_buckets = len(channel_set)
        self.buckets = {i: Bucket(batch_size) for i in range(self.num_buckets)}

        for idx, channel in enumerate(channel_list):
            bucket_idx = channel_set.index(channel)
            self.buckets[bucket_idx].append(idx)

        self.bucket_sample_sequence = []
        for i in range(self.num_buckets):
            self.bucket_sample_sequence += [i] * (len(self.buckets[i]) - 1)
        self.last_sample_sequence = [i for i in range(self.num_buckets)]

    def __iter__(self):
        rand = random.Random(SEED + self.rank)
        rand.shuffle(self.bucket_sample_sequence)
        for bucket_idx in range(self.num_buckets):
            self.buckets[bucket_idx].shuffle()
        buckets = [iter(self.buckets[i]) for i in range(self.num_buckets)]
        for i in self.bucket_sample_sequence+self.last_sample_sequence:
            yield next(buckets[i])

    def __len__(self):
        return sum([len(self.buckets[i]) for i in range(self.num_buckets)])


def build_brain_bucket_dataloader(
    mode,
    ratio,
    metadata_path,
    accessor,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool = False,
):
    with open(os.path.join(metadata_path, f"{mode}.json"), "r") as f:
        metadata_list = json.load(f)
    # 多卡划分
    channels_set = sorted(set([i["channels"] for i in metadata_list]))
    replicated_metadata_list = []
    for channels in channels_set:
        channel_metadata_list = [
            item
            for item in metadata_list
            if item["channels"] == channels
        ]
        random.shuffle(channel_metadata_list)
        channel_metadata_list = channel_metadata_list[
            : int(len(channel_metadata_list) * ratio)
        ]
        len_replicas = len(channel_metadata_list) // world_size
        replicated_metadata_list += channel_metadata_list[
            rank * len_replicas : (rank + 1) * len_replicas
        ]
        replicated_metadata_list += channel_metadata_list[
            world_size * len_replicas :
        ]
    brain_dataset = BrainDataset(
        metadata_list=replicated_metadata_list, accessor=accessor
    )

    return DataLoader(
        dataset=brain_dataset,
        batch_sampler=BucketBatchSampler(
            [i["channels"] for i in replicated_metadata_list],
            batch_size=batch_size,
            rank=rank,
        ),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

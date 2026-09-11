"""Read raw neural recordings and prepared tensor or JSON artifacts.

Recording discovery walks a configured root, excluding hidden directories
and nested derivatives. BIDS roots select subject directories only. CTF .ds
directories are recording containers; FIF
split continuations are opened through the first split by MNE. An explicitly
configured derivative root can be used directly. Paths returned are absolute.
"""

import os
import re
import io
import mne
import json
import torch
from pathlib import Path
from typing import Any
from constant import DATA_ROOT_PATH
from tqdm import tqdm


BIDS_DESCRIPTION_NAME = "dataset_description.json"
BIDS_SUBJECT_PREFIX = "sub-"
EXCLUDED_RECORDING_DIRECTORIES = frozenset({"derivatives"})
FIF_SPLIT_PATTERN = re.compile(r"(?:^|_)split-(\d+)_")
FIRST_FIF_SPLIT = 1


def is_useful_path(name: str) -> bool:
    """Reject noise recordings and FIF parts MNE reads via the first split."""
    lower = name.lower()
    if any(marker in lower for marker in ("empty", "noise", "hz.ds")):
        return False
    split = FIF_SPLIT_PATTERN.search(lower)
    return not (
        lower.endswith(".fif") and split
        and int(split.group(1)) > FIRST_FIF_SPLIT
    )


def load_torch_warpper(path):
    return torch.load(path, weights_only=True)


def write_torch_warpper(data, path):
    torch.save(data, path)


def load_json_warpper(path):
    if isinstance(path, str):
        with open(path, "r") as f:
            return json.load(f)
    assert isinstance(path, io.BytesIO)
    return json.loads(path.getvalue().decode("utf-8"))


def write_json_warpper(data, path):
    if isinstance(path, str):
        with open(path, "w") as f:
            return json.dump(data, f, indent=4)
    assert isinstance(path, io.BytesIO)
    json_string = json.dumps(data, indent=4)
    return path.write(json_string.encode("utf-8"))


class DataAccessor:
    """
    Load processed tensor files.
    """

    def __init__(self, read_only: bool = True):
        self.read_only = read_only
        self.brain_read_func_dict = {
            "fif": mne.io.read_raw_fif,
            "con": mne.io.read_raw_kit,
            "bdf": mne.io.read_raw_bdf,
            "edf": mne.io.read_raw_edf,
            "vhdr": mne.io.read_raw_brainvision,
            "ds": mne.io.read_raw_ctf,
            "set": mne.io.read_raw_eeglab,
            "cnt": mne.io.read_raw_cnt,
            "gdf": mne.io.read_raw_gdf,
        }

    def search_brain_files(
        self,
        root_path: str,
        dataset: str,
    ) -> list[dict[str, str]]:
        """Find supported MNE recordings under one catalog dataset root.

        Parameters
        ----------
        root_path : str
            Absolute dataset root from the local data catalog.
        dataset : str
            Catalog dataset ID assigned to every discovered recording.

        Returns
        -------
        list of dict of str to str
            Recording mappings with ``path`` and catalog ``dataset`` fields.
        """
        brain_files: list[dict[str, str]] = []
        root_path = str(Path(root_path).resolve())
        bids_root = (Path(root_path) / BIDS_DESCRIPTION_NAME).is_file()
        for root, directories, names in tqdm(os.walk(root_path)):
            if root == root_path and bids_root:
                directories[:] = [
                    name for name in directories
                    if name.startswith(BIDS_SUBJECT_PREFIX)
                ]
                names = []
            directories[:] = sorted(
                name for name in directories
                if not name.startswith(".")
                and name not in EXCLUDED_RECORDING_DIRECTORIES
            )
            containers = [
                name for name in directories if name.lower().endswith(".ds")
            ]
            directories[:] = [
                name for name in directories if name not in containers
            ]
            for name in sorted([*names, *containers]):
                extension = name.rsplit(".", 1)[-1].lower()
                if (
                    not name.startswith(".")
                    and extension in self.brain_read_func_dict
                    and is_useful_path(name)
                ):
                    brain_files.append({
                        "path": os.path.join(root, name), "dataset": dataset,
                    })
        return brain_files

    def read_brain_file(self, path: str, preload: bool = True) -> Any:
        """Read one supported MNE recording using its filename extension."""
        extension = path.rsplit(".", 1)[-1].lower()
        return self.brain_read_func_dict[extension](
            path, verbose=False, preload=preload
        )

    def get_usage_folder_name(self, path: str):
        return path.replace(DATA_ROOT_PATH, "").split("/")[0]

    def get_dataset_folder_name(self, path: str):
        resolved_path = Path(path).resolve()
        for parent in resolved_path.parents:
            if (parent / "dataset_description.json").is_file():
                return parent.name
        return path.replace(DATA_ROOT_PATH, "").split("/")[1]

    def replace_usage_folder_name(self, path: str, new_usage: str):
        return path.replace(
            f"/{self.get_usage_folder_name(path)}/", f"/{new_usage}/"
        )

    def mkdir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def exist(self, path: str):
        return os.path.exists(path)

    def read(self, path: str, read_func):
        assert self.exist(path)
        return read_func(path)

    def write(self, data, path: str, write_func):
        if self.read_only or self.get_usage_folder_name(path) in [
            "raw",
            "evaluate",
        ]:
            return None
        write_func(data, path)

    def remove(self, path):
        if self.read_only:
            return None
        assert self.exist(path)
        os.remove(path)

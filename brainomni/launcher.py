"""Launch configurable BrainOmni pre-training."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch

from brainomni.config import BrainOmniTrainerConfig
from brainomni.trainer import Trainer
from pretrain_config import (
    load_pretrain_config,
    resolve_dataset_identities,
    write_run_artifacts,
)


def seed_everything(seed: int) -> None:
    """Seed all supported random number generators for a campaign."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse configuration and DeepSpeed-provided launch arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--launcher", type=str)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--local-config", type=str)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def create_run_directory(
    rank: int,
    config: dict[str, object],
    model_config: dict[str, object],
    tokenizer_identity: dict[str, str],
) -> str:
    """Create a run directory and write configuration artifacts on rank zero."""
    invocation = config["invocation"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_path = Path(invocation["output_root"]) / invocation["run_name"]
    run_path = run_path / f"exp_{timestamp}"
    if rank == 0:
        write_run_artifacts(
            run_path,
            config,
            model_config,
            tokenizer_identity,
        )
    return str(run_path)


def main() -> None:
    """Resolve settings and start distributed BrainOmni training."""
    args = parse_args()
    config = load_pretrain_config(
        args.config, args.local_config, args.overrides
    )
    config = resolve_dataset_identities(config)
    seed_everything(config["campaign"]["seed"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    trainer_config = BrainOmniTrainerConfig(config, world_size)
    run_path = create_run_directory(
        rank,
        config,
        trainer_config.get_model_cfg(),
        trainer_config.tokenizer_identity,
    )
    Trainer(
        trainer_config,
        local_rank,
        rank,
        world_size,
        exp_path=run_path,
    ).main()


if __name__ == "__main__":
    main()

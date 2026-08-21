"""Launch one exact-semantic BrainOmni pre-training campaign."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from brainomni.config import BrainOmniTrainerConfig
from brainomni.trainer import Trainer
from factory.campaign import (
    ensure_training_campaign,
    prepare_campaign,
    record_attempt_repair,
    update_attempt_status,
)
from factory.training_runtime import destroy_distributed_process_group
from pretrain_config import (
    load_pretrain_launch_config,
    resolve_dataset_identities,
)


def seed_everything(seed: int) -> None:
    """Seed all supported random-number generators for a campaign."""
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
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
    )
    return parser.parse_args()


def main() -> None:
    """Resolve settings and start or resume BrainOmni training."""
    args = parse_args()
    config = load_pretrain_launch_config(args.config, args.overrides)
    config = resolve_dataset_identities(config)
    seed_everything(config["campaign"]["seed"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    trainer_config = BrainOmniTrainerConfig(config, world_size)
    context = prepare_campaign(
        config,
        trainer_config.get_model_cfg(),
        tokenizer_identity=trainer_config.tokenizer_identity,
        config_paths=args.config,
        overrides=args.overrides,
        world_size=world_size,
        rank=rank,
    )
    if rank == 0 and trainer_config.tokenizer_health.repaired:
        record_attempt_repair(context, trainer_config.tokenizer_health)
    training_required = ensure_training_campaign(context, rank=rank)
    try:
        Trainer(
            trainer_config,
            local_rank,
            rank,
            world_size,
            campaign=context,
            training_required=training_required,
        ).main()
    except Exception as error:
        if rank == 0:
            update_attempt_status(context, "failed", error)
        raise
    else:
        if rank == 0:
            update_attempt_status(context, "complete")
    finally:
        destroy_distributed_process_group()


if __name__ == "__main__":
    main()

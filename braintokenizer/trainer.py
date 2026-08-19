from pathlib import Path
import os
import math
import torch
import logging
import deepspeed
import matplotlib.pyplot as plt
import deepspeed.comm as dist
from tqdm import tqdm
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from accessor import DataAccessor
from pretrain_dataset import build_brain_bucket_dataloader
from factory.campaign import (
    CampaignContext,
    export_completed_weights,
    repository_attempt_log_directory,
    validate_checkpoint,
)
from factory.training_runtime import (
    evaluation_metadata_available,
    evaluation_metadata_path,
    existing_evaluation_matches,
    load_completed_portable,
    resume_distributed_checkpoint,
    save_distributed_checkpoint,
    write_evaluation_metrics,
)
from braintokenizer.config import BrainTokenizerTrainerConfig
from braintokenizer.model import BrainTokenizer
from braintokenizer.metrics import MetricsComputer


def batched_bincount(x, num_classes, dim):
    target = torch.zeros(
        (list(x.shape[:-1]) + [num_classes]), dtype=x.dtype, device=x.device
    )
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


class EmptyLogger:
    def info(self, *awargs, **kwargs):
        return None


class EmptyWriter:
    def add_scalar(self, *awargs, **kwargs):
        return None

    def close(self):
        return None

    def add_figure(self, *awargs, **kwargs):
        return None


class Trainer:
    def __init__(
        self,
        cfg: BrainTokenizerTrainerConfig,
        local_rank: int,
        rank: int,
        world_size: int,
        campaign: CampaignContext,
        training_required: bool,
    ):
        # prepare basic environment
        self.cfg = cfg
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.campaign = campaign
        self.training_required = training_required
        self.exp_path = str(campaign.root)
        self.ckpt_path = str(campaign.checkpoint_root)
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.accessor = DataAccessor(read_only=True)

        # configuration
        self.epoch = 0
        self.total_epoch = cfg.epoch
        self.best_eval_loss = 1000000000000
        self.logger = self.build_logger()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()

        self.logger.info("=> Building train dataloader ...")
        self.train_loader = self.build_dataloader(
            mode="train",
            ratio=self.cfg.train_data_ratio,
            persistent_workers=True,
        )
        self.logger.info("=> Building val dataloader ...")
        self.val_loader = self.build_dataloader(
            mode="val", ratio=self.cfg.val_data_ratio
        )

        self.train_step_counter = 0

        train_total_steps = (
            len(self.train_loader)
            * self.total_epoch
            // self.cfg.gradient_accumulation_steps
        )

        self.logger.info(
            "=> Building model and initializing distributed environment..."
        )
        self.cfg.ds_config["scheduler"]["params"][
            "total_num_steps"
        ] = train_total_steps
        self.cfg.ds_config["scheduler"]["params"]["warmup_num_steps"] = int(
            train_total_steps * self.cfg.scheduler_warm_ratio
        )
        self.model = self.deepspeed_initialize()
        if self.training_required:
            restored = resume_distributed_checkpoint(
                self.model,
                self.campaign,
            )
            if restored is not None:
                (
                    self.epoch,
                    self.best_eval_loss,
                    self.train_step_counter,
                ) = restored
                self.logger.info(
                    "Resumed exact campaign at epoch %s from %s.",
                    self.epoch,
                    (self.campaign.checkpoint_root / "latest").resolve(),
                )

    def main(self):
        """Train or reuse the exact campaign, export, and evaluate."""
        if self.training_required:
            self._train_epochs()
            self._load_best_checkpoint()
            self._export_completed_checkpoint()
        else:
            load_completed_portable(self.model, self.campaign)
        self.model.eval()
        del self.train_loader
        del self.val_loader
        self._evaluate_requested_datasets()
        self.writer.close()
        dist.destroy_process_group()

    def _train_epochs(self):
        self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
        while self.epoch < self.total_epoch:
            self.count_epoch()
            self.before_epoch()
            self.model.train()
            loader = self.train_loader
            if self.rank == 0:
                loader = tqdm(loader, unit="batch")
            for self.input_dict in loader:
                if self.rank == 0:
                    loader.set_description(f"Epoch {self.epoch}")
                    loader.set_postfix(self.train_step())
                else:
                    self.train_step()
            self.model.eval()
            loader = self.val_loader
            if self.rank == 0:
                loader = tqdm(loader, unit="batch")
            for self.input_dict in loader:
                if self.rank == 0:
                    loader.set_description(f"Epoch {self.epoch}")
                    loader.set_postfix(self.eval_step())
                else:
                    self.eval_step()
            self.after_epoch()
        self.logger.info(">>>>>>>>>>>>>>>> Finish Training >>>>>>>>>>>>>>>>")

    def _load_best_checkpoint(self):
        validate_checkpoint(self.campaign.root, "best")
        load_path, _ = self.model.load_checkpoint(
            load_dir=self.ckpt_path,
            tag="best",
        )
        if load_path is None:
            raise RuntimeError(
                "DeepSpeed could not load the verified best checkpoint at "
                f"{(self.campaign.checkpoint_root / 'best').resolve()}."
            )

    def _export_completed_checkpoint(self):
        dist.barrier()
        failed = torch.zeros(
            (1,),
            device=self.local_rank,
            dtype=torch.int32,
        )
        if self.rank == 0:
            try:
                health = export_completed_weights(
                    self.campaign,
                    expected_state=self.model.module.state_dict(),
                )
            except Exception as error:
                failed.fill_(1)
                self.logger.info(
                    "Failed to export verified BrainTokenizer weights: %s",
                    error,
                )
            else:
                self.logger.info(
                    "Verified portable BrainTokenizer weights: %s",
                    health.portable_path,
                )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError(
                "Rank zero could not export and validate BrainTokenizer.pt."
            )
        dist.barrier()

    def _evaluate_requested_datasets(self):
        self.logger.info("=> Start Testing ...")
        evaluator_path = Path(__file__).resolve()
        for mode in self.cfg.evaluation_datasets:
            if not evaluation_metadata_available(
                self.campaign,
                self.cfg.pretrain_metadata_path,
                mode,
                self.rank,
            ):
                dist.barrier()
                continue
            metadata_path = evaluation_metadata_path(
                self.cfg.pretrain_metadata_path,
                mode,
            )
            if existing_evaluation_matches(
                self.campaign,
                mode,
                evaluator_path,
                metadata_path,
            ):
                self.logger.info("Verified existing evaluation for %s.", mode)
                dist.barrier()
                continue
            self.test_loader = self.build_dataloader(mode=mode, ratio=1.0)
            if len(self.test_loader) == 0:
                raise RuntimeError(
                    f"Evaluation loader for {mode!r} is empty despite "
                    "non-empty metadata. Check distributed batch sizing."
                )
            self.metrics_computer = MetricsComputer()
            for index, self.input_dict in enumerate(self.test_loader):
                input_dict = self.fetch_input_dict()
                output_dict = self.model.visualize(**input_dict)
                self.metrics_computer.step(
                    output_dict["x_rec"],
                    output_dict["x"],
                    output_dict["sensor_type"],
                )
                if index % 10 == 0:
                    self.write_visualize_result(
                        output_dict["x"],
                        output_dict["x_rec"],
                        tag=mode,
                        global_step=index,
                    )
            metrics = self.metrics_computer.get_metrics()
            for metric_group in metrics.values():
                for key in metric_group:
                    metric_group[key] = self.scalar_comm_reduce(
                        metric_group[key]
                    )
            if self.rank == 0:
                path = write_evaluation_metrics(
                    self.campaign,
                    mode,
                    metrics,
                    evaluator_path,
                    metadata_path,
                )
                self.logger.info("Saved evaluation metrics to %s.", path)
            dist.barrier()

    def count_epoch(self):
        self.epoch += 1

    def before_epoch(self):
        self.logger.info(
            f">>>>>>>>>>>>>>>> Epoch {self.epoch} >>>>>>>>>>>>>>>>"
        )
        self.eval_running_indices = []
        self.train_running_dict = {
            "loss": 0.0,
            "time_loss": 0.0,
            "pcc": 0.0,
            "amp_loss": 0.0,
            "phase_loss": 0.0,
            "commitment_loss": 0.0,
            "judge_loss": 0.0,
        }
        self.eval_running_dict = {
            "loss": 0.0,
            "time_loss": 0.0,
            "pcc": 0.0,
            "amp_loss": 0.0,
            "phase_loss": 0.0,
            "commitment_loss": 0.0,
            "judge_loss": 0.0,
        }

    def fetch_input_dict(self):
        input_dict = self.input_dict
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].to(
                    device=self.local_rank, non_blocking=True
                )
        return input_dict

    def write_visualize_result(
        self,
        raw: torch.Tensor,
        rec: torch.Tensor,
        tag: str = "reconstruction_comparison",
        global_step: int = None,
    ):
        raw = raw.detach().cpu().float()
        rec = rec.detach().cpu().float()
        raw = rearrange(raw, "... D -> (...) D")
        rec = rearrange(rec, "... D -> (...) D")
        random_select_indices = torch.randperm(raw.shape[0])[:4]
        plt.figure(figsize=(12, 12))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            x = raw[random_select_indices[i]]
            plt.plot(x, label="raw")
            plt.plot(rec[random_select_indices[i]], label="rec")
            plt.legend()
        self.writer.add_figure(
            tag,
            plt.gcf(),
            global_step=(
                self.train_step_counter if global_step is None else global_step
            ),
        )
        plt.close()

    def train_step(self):
        input_dict = self.fetch_input_dict()
        output_dict, _ = self.model(**input_dict)
        tqdm_dict = {k: v.item() for k, v in output_dict.items()}
        for key in self.train_running_dict.keys():
            self.train_running_dict[key] += output_dict[key].item()
        loss = output_dict["loss"]
        self.model.backward(loss)
        self.model.step()
        if self.train_step_counter % self.cfg.visualization_interval_steps == 0:
            self.model.eval()
            output_dict = self.model.visualize(**input_dict)
            self.write_visualize_result(
                output_dict["x"],
                output_dict["x_rec"],
            )
            self.model.train()
            torch.cuda.empty_cache()
        self.train_step_counter += 1
        return tqdm_dict

    @torch.no_grad()
    def eval_step(self):
        input_dict = self.fetch_input_dict()
        output_dict, indices = self.model(**input_dict)
        tqdm_dict = {k: v.item() for k, v in output_dict.items()}
        for key in self.eval_running_dict.keys():
            self.eval_running_dict[key] += output_dict[key].item()
        self.eval_running_indices.append(
            indices.cpu().view(-1, self.cfg.num_quantizers)
        )
        return tqdm_dict

    def scalar_comm_reduce(self, scalar, op=dist.ReduceOp.AVG):
        tensor_scalar = torch.tensor(
            [scalar], device=self.local_rank, dtype=torch.float32
        )
        dist.all_reduce(tensor_scalar, op=op)
        return tensor_scalar.item()

    def after_epoch(self):
        torch.cuda.empty_cache()
        indices = (
            torch.vstack(self.eval_running_indices)
            .transpose(0, 1)
            .to(self.local_rank)
        )
        codebook_count = batched_bincount(indices, self.cfg.codebook_size, -1)
        dist.all_reduce(codebook_count, op=dist.ReduceOp.SUM)
        codebook_count = codebook_count / codebook_count.sum(
            dim=-1, keepdim=True
        )
        codebook_utilize_entropy = -torch.sum(
            codebook_count * torch.log2(codebook_count + 1e-6), dim=-1
        )
        codebook_utilize_entropy /= math.log2(self.cfg.codebook_size)
        for i in range(self.cfg.num_quantizers):
            self.writer.add_scalar(
                tag=f"eval_codebook_utilize_entropy_{i}",
                scalar_value=codebook_utilize_entropy[i].item(),
                global_step=self.epoch,
            )

        self.writer.add_scalar(
            tag=f"eval_codebook_utilize_entropy_mean",
            scalar_value=codebook_utilize_entropy.mean().item(),
            global_step=self.epoch,
        )

        for key in self.train_running_dict.keys():
            self.train_running_dict[key] = self.train_running_dict[key] / len(
                self.train_loader
            )
            self.train_running_dict[key] = self.scalar_comm_reduce(
                self.train_running_dict[key]
            )
            self.writer.add_scalar(
                tag=f"train_{key}",
                scalar_value=self.train_running_dict[key],
                global_step=self.epoch,
            )
        for key in self.eval_running_dict.keys():
            self.eval_running_dict[key] = self.eval_running_dict[key] / len(
                self.val_loader
            )
            self.eval_running_dict[key] = self.scalar_comm_reduce(
                self.eval_running_dict[key]
            )
            self.writer.add_scalar(
                tag=f"eval_{key}",
                scalar_value=self.eval_running_dict[key],
                global_step=self.epoch,
            )
            self.logger.info(
                f"train {key}:{self.train_running_dict[key]} "
                f"eval {key}:{self.eval_running_dict[key]}"
            )
        self.logger.info(
            f"code utilize entropy:{codebook_utilize_entropy.cpu()}"
        )
        self.logger.info("")

        if self.eval_running_dict["judge_loss"] < self.best_eval_loss:
            self.best_eval_loss = self.eval_running_dict["judge_loss"]
            self.save_ckpt(tag="best")
        if self.epoch % self.cfg.checkpoint_interval_epochs == 0:
            self.save_ckpt(tag=f"epoch_{self.epoch}")
        self.save_ckpt(tag="latest")

    def save_ckpt(self, tag: str):
        """Save one manifested collective checkpoint with recovery state."""
        save_distributed_checkpoint(
            self.model,
            self.campaign,
            tag,
            self.epoch,
            self.best_eval_loss,
            self.train_step_counter,
            self.rank,
        )

    def deepspeed_initialize(self):
        model = BrainTokenizer(
            **self.cfg.get_model_cfg(),
            channel_mask_ratio=self.cfg.channel_mask_ratio,
        )
        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.get_parameters_groups(
                lr=self.cfg.lr,
                codebook_lr=self.cfg.codebook_lr,
                weight_decay=self.cfg.weight_decay,
            ),
            config=self.cfg.ds_config,
        )
        n_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        self.logger.info(f"Num params: {n_parameters/1.0e9} B")
        return model

    def build_writer(self):
        if self.rank != 0:
            return EmptyWriter()
        writer_path = self.campaign.attempt_root / "tensorboard"
        writer = SummaryWriter(writer_path)
        self.logger.info(
            "TensorBoard writer logging directory: %s",
            writer_path.resolve(),
        )
        return writer

    def build_dataloader(self, mode, ratio, persistent_workers=False):
        return build_brain_bucket_dataloader(
            mode=mode,
            ratio=ratio,
            metadata_path=self.cfg.pretrain_metadata_path,
            accessor=self.accessor,
            rank=self.rank,
            world_size=self.world_size,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            persistent_workers=persistent_workers,
        )

    def build_logger(self):
        if self.rank != 0:
            return EmptyLogger()
        logger = logging.getLogger(name="BrainTokenizer")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
            "%H:%M:%S",
        )

        log_directory = repository_attempt_log_directory(self.campaign)
        log_directory.mkdir(parents=True, exist_ok=True)
        fileHandler = logging.FileHandler(
            log_directory / "logs.txt",
            encoding="utf-8",
        )
        fileHandler.setLevel(logging.INFO)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        screenHandler = logging.StreamHandler()
        screenHandler.setLevel(logging.INFO)
        screenHandler.setFormatter(formatter)
        logger.addHandler(screenHandler)

        logger.info(f"Save experiment in {self.exp_path}")
        return logger

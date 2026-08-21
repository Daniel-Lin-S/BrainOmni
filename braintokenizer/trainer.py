import os
from pathlib import Path

import torch
import logging
from urllib.parse import quote
import deepspeed
import matplotlib.pyplot as plt
import deepspeed.comm as dist
from tqdm import tqdm
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from accessor import DataAccessor
from pretrain_dataset import (
    build_brain_bucket_dataloader,
    build_fixed_monitor_batch,
)
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
from factory.lr_scheduler import warmup_cosine_scheduler_factory
from braintokenizer.config import BrainTokenizerTrainerConfig
from braintokenizer.model import BrainTokenizer
from braintokenizer.metrics import MetricsComputer
from factory.pretraining_monitor_runtime import StageOneAccumulator
from factory.pretraining_monitors import (
    MONITOR_EPSILON,
    attention_similarity_statistics,
    canonical_tag,
    checked_ratio,
    distributed_update_to_weight_ratio,
    level_name,
    monitor_due,
    optimizer_step_values,
    successful_optimizer_steps,
    write_scalars,
    zero_partition_snapshot,
)


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
        self.model = self.deepspeed_initialize(train_total_steps)
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
        self.step_monitor = StageOneAccumulator(self.cfg.codebook_size)
        self.fixed_monitor_batch = None
        self.codebook_monitor_snapshot = None
        if self.training_required:
            self.fixed_monitor_batch = build_fixed_monitor_batch(
                metadata_path=self.cfg.pretrain_metadata_path,
                accessor=self.accessor,
                rank=self.rank,
                world_size=self.world_size,
                batch_size=self.cfg.batch_size,
            )
            self.codebook_monitor_snapshot = self._codebook_snapshot()

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
                        tag=(
                            "evaluation/visualization/reconstruction/"
                            f"{quote(mode, safe='._-')}"
                        ),
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
        self.train_epoch_monitor = StageOneAccumulator(
            self.cfg.codebook_size
        )
        self.validation_epoch_monitor = StageOneAccumulator(
            self.cfg.codebook_size
        )
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
        tag: str = (
            "train/micro_step/visualization/reconstruction"
        ),
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
        boundary = self.model.is_gradient_accumulation_boundary()
        successful_before = successful_optimizer_steps(self.model)
        lightweight_due = monitor_due(
            self.model,
            self.cfg.lightweight_monitor_interval_steps,
            boundary,
        )
        diagnostic_due = monitor_due(
            self.model,
            self.cfg.diagnostic_monitor_interval_steps,
            boundary,
        )
        learning_rates = self.model.get_lr() if lightweight_due else None
        weight_snapshot = (
            zero_partition_snapshot(self.model) if diagnostic_due else None
        )
        output_dict, _, monitor_data = self.model(
            **input_dict,
            return_monitor_data=True,
        )
        tqdm_dict = {k: v.item() for k, v in output_dict.items()}
        for key in self.train_running_dict.keys():
            self.train_running_dict[key] += output_dict[key].item()
        self.step_monitor.update(output_dict, monitor_data)
        self.train_epoch_monitor.update(output_dict, monitor_data)
        loss = output_dict["loss"]
        self.model.backward(loss)
        self.model.step()
        if boundary:
            successful_after = successful_optimizer_steps(self.model)
            if successful_after > successful_before:
                if lightweight_due:
                    self._write_lightweight_monitors(
                        successful_after,
                        learning_rates,
                    )
                if diagnostic_due:
                    self._write_diagnostic_monitors(
                        successful_after,
                        weight_snapshot,
                    )
            self.step_monitor = StageOneAccumulator(
                self.cfg.codebook_size
            )
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
        output_dict, _, monitor_data = self.model(
            **input_dict,
            return_monitor_data=True,
        )
        tqdm_dict = {k: v.item() for k, v in output_dict.items()}
        for key in self.eval_running_dict.keys():
            self.eval_running_dict[key] += output_dict[key].item()
        self.validation_epoch_monitor.update(output_dict, monitor_data)
        return tqdm_dict

    def scalar_comm_reduce(self, scalar, op=dist.ReduceOp.AVG):
        tensor_scalar = torch.tensor(
            [scalar], device=self.local_rank, dtype=torch.float32
        )
        dist.all_reduce(tensor_scalar, op=op)
        return tensor_scalar.item()

    def _reduce_sum(self, value: torch.Tensor) -> None:
        """Sum one monitor sufficient statistic across all ranks."""
        dist.all_reduce(value, op=dist.ReduceOp.SUM)

    def _codebook_snapshot(self) -> torch.Tensor:
        """Return an in-memory snapshot of all effective RVQ codebooks."""
        return (
            self.model.module.quantizer.rvq.codebooks
            .detach()
            .double()
            .clone()
        )

    def _write_lightweight_monitors(
        self,
        optimizer_step: int,
        learning_rates,
    ) -> None:
        """Write Stage-1 and optimization scalars at lightweight cadence."""
        if learning_rates is None:
            raise RuntimeError(
                "Learning rates were not captured before the optimizer step."
            )
        self.step_monitor.reduce_(self._reduce_sum)
        values = self.step_monitor.training_values("step")
        values.update(
            optimizer_step_values(
                self.model,
                learning_rates,
                self.optimizer_group_names,
            )
        )
        if self.rank == 0:
            write_scalars(self.writer, values, optimizer_step)

    def _write_diagnostic_monitors(
        self,
        optimizer_step: int,
        weight_snapshot: list[torch.Tensor] | None,
    ) -> None:
        """Write Stage-1 diagnostics at the configured sparse cadence."""
        if weight_snapshot is None:
            raise RuntimeError(
                "Update-to-weight monitoring has no pre-update snapshot."
            )
        update_ratio = distributed_update_to_weight_ratio(
            weight_snapshot,
            self.model,
            self._reduce_sum,
            self.local_rank,
        )

        if self.codebook_monitor_snapshot is None:
            raise RuntimeError(
                "Codebook monitoring has no previous in-memory snapshot."
            )
        current_codebooks = self._codebook_snapshot()
        codebook_update = torch.sqrt(
            torch.square(current_codebooks - self.codebook_monitor_snapshot)
            .sum(dim=(1, 2))
        )
        codebook_norm = torch.sqrt(
            torch.square(self.codebook_monitor_snapshot).sum(dim=(1, 2))
        )
        codebook_ratio = codebook_update / (
            codebook_norm + MONITOR_EPSILON
        )
        self.codebook_monitor_snapshot = current_codebooks

        if self.fixed_monitor_batch is None:
            raise RuntimeError(
                "Inter-query attention monitoring has no fixed validation "
                "batch."
            )
        fixed_batch = {
            key: self.fixed_monitor_batch[key].to(
                device=self.local_rank,
                non_blocking=True,
            )
            for key in ("x", "pos", "sensor_type")
        }
        was_training = self.model.training
        self.model.eval()
        attention = self.model.module.monitor_attention(**fixed_batch)
        similarity_sum, similarity_count = (
            attention_similarity_statistics(attention)
        )
        similarity_statistics = torch.stack(
            (similarity_sum, similarity_count)
        ).to(device=self.local_rank)
        self._reduce_sum(similarity_statistics)
        similarity_mean = checked_ratio(
            similarity_statistics[0],
            similarity_statistics[1],
            "distributed inter-query attention similarity",
        )
        if was_training:
            self.model.train()

        values = {
            canonical_tag(
                "train",
                "step",
                "optimization",
                "update_to_weight_ratio",
                "global",
            ): update_ratio,
            canonical_tag(
                "validation",
                "step",
                "latent_source",
                "inter_query_attention_similarity",
            ): similarity_mean,
        }
        for level in range(codebook_ratio.numel()):
            values[
                canonical_tag(
                    "train",
                    "step",
                    "rvq",
                    "codebook_update_magnitude",
                    level_name(level),
                )
            ] = codebook_ratio[level]
        if self.rank == 0:
            write_scalars(self.writer, values, optimizer_step)

    def after_epoch(self):
        torch.cuda.empty_cache()
        for key in self.train_running_dict.keys():
            self.train_running_dict[key] = self.train_running_dict[key] / len(
                self.train_loader
            )
            self.train_running_dict[key] = self.scalar_comm_reduce(
                self.train_running_dict[key]
            )
        for key in self.eval_running_dict.keys():
            self.eval_running_dict[key] = self.eval_running_dict[key] / len(
                self.val_loader
            )
            self.eval_running_dict[key] = self.scalar_comm_reduce(
                self.eval_running_dict[key]
            )
            self.logger.info(
                f"train {key}:{self.train_running_dict[key]} "
                f"eval {key}:{self.eval_running_dict[key]}"
            )
        self.train_epoch_monitor.reduce_(self._reduce_sum)
        self.validation_epoch_monitor.reduce_(self._reduce_sum)
        if self.rank == 0:
            write_scalars(
                self.writer,
                self.train_epoch_monitor.training_values("epoch"),
                self.epoch,
            )
            write_scalars(
                self.writer,
                self.validation_epoch_monitor.validation_values(),
                self.epoch,
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

    def deepspeed_initialize(self, train_total_steps: int):
        """Build the model, optimizer groups, scheduler, and engine."""
        model = BrainTokenizer(
            **self.cfg.get_model_cfg(),
            channel_mask_ratio=self.cfg.channel_mask_ratio,
            noise_std=self.cfg.noise_std,
        )
        parameter_groups = model.get_named_parameter_groups(
            lr=self.cfg.lr,
            codebook_lr=self.cfg.codebook_lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.optimizer_group_names = tuple(parameter_groups)
        scheduler_factory = warmup_cosine_scheduler_factory(
            total_num_steps=train_total_steps,
            warmup_ratio=self.cfg.scheduler_warm_ratio,
            warmup_min_lr_ratio=(
                self.cfg.scheduler_warmup_min_lr_ratio
            ),
            cosine_min_ratio=self.cfg.scheduler_cosine_min_ratio,
        )
        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=list(parameter_groups.values()),
            lr_scheduler=scheduler_factory,
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

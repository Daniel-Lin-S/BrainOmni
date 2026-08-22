import os
from pathlib import Path

import torch
import logging
import deepspeed
import deepspeed.comm as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from accessor import DataAccessor
from brainomni.config import BrainOmniTrainerConfig
from brainomni.model import BrainOmni
from pretrain_dataset import (
    build_brain_bucket_dataloader,
    training_dataset_ids,
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
from factory.pretraining_integrity import (
    nonfinite_gradient_names,
    nonfinite_tensor_names,
    warn_and_raise_distributed_failure,
)
from factory.pretraining_monitor_runtime import (
    ExposureAccumulator,
    StageTwoAccumulator,
)
from factory.pretraining_monitors import (
    canonical_tag,
    distributed_update_to_weight_ratio,
    modality_channel_groups,
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
        cfg: BrainOmniTrainerConfig,
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
            mode="val", ratio=self.cfg.valid_data_ratio
        )
        self.training_dataset_ids = training_dataset_ids(
            self.cfg.pretrain_metadata_path
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
        self.step_monitor = StageTwoAccumulator()

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
                    "Failed to export verified BrainOmni weights: %s",
                    error,
                )
            else:
                self.logger.info(
                    "Verified portable BrainOmni weights: %s",
                    health.portable_path,
                )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError(
                "Rank zero could not export and validate BrainOmni.pt."
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
            self.test_running_dict = {"loss": 0.0, "acc_all": 0.0}
            for index in range(self.cfg.num_quantizers_used):
                self.test_running_dict[f"acc_{index}"] = 0.0
            with torch.no_grad():
                for self.input_dict in self.test_loader:
                    output_dict = self.model(**self.fetch_input_dict())
                    for key in self.test_running_dict:
                        self.test_running_dict[key] += output_dict[key].item()
            torch.cuda.empty_cache()
            for key in self.test_running_dict:
                value = self.test_running_dict[key] / len(self.test_loader)
                self.test_running_dict[key] = self.scalar_comm_reduce(value)
            if self.rank == 0:
                path = write_evaluation_metrics(
                    self.campaign,
                    mode,
                    self.test_running_dict,
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
        self.train_running_dict = {"loss": 0.0, "acc_all": 0.0}
        self.eval_running_dict = {"loss": 0.0, "acc_all": 0.0}
        for i in range(self.cfg.num_quantizers_used):
            self.train_running_dict[f"acc_{i}"] = 0.0
            self.eval_running_dict[f"acc_{i}"] = 0.0
        self.train_epoch_monitor = StageTwoAccumulator()
        self.validation_epoch_monitor = StageTwoAccumulator()
        self.train_exposure_monitor = ExposureAccumulator(
            self.training_dataset_ids
        )

    def fetch_input_dict(self):
        input_dict = self.input_dict
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].to(
                    device=self.local_rank, non_blocking=True
                )
        return input_dict

    def train_step(self):
        input_dict = self.fetch_input_dict()
        self._check_finite_inputs(input_dict)
        self.train_exposure_monitor.update(
            input_dict["dataset"],
            input_dict["sensor_type"],
        )
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
        output_dict, monitor_data = self.model(
            **input_dict,
            return_monitor_data=True,
        )
        tqdm_dict = {k: v.item() for k, v in output_dict.items()}
        for key in self.train_running_dict.keys():
            self.train_running_dict[key] += output_dict[key].item()
        self.step_monitor.update(output_dict, monitor_data)
        self.train_epoch_monitor.update(output_dict, monitor_data)
        loss = output_dict["loss"]
        self._check_finite_training_loss(loss)
        self.model.backward(loss)
        self._check_finite_gradients()
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
            self.step_monitor = StageTwoAccumulator()
        self.train_step_counter += 1
        return tqdm_dict

    @torch.no_grad()
    def eval_step(self):
        input_dict = self.fetch_input_dict()
        self._check_finite_inputs(input_dict)
        output_dict, monitor_data = self.model(
            **input_dict,
            return_monitor_data=True,
        )
        tqdm_dict = {k: v.item() for k, v in output_dict.items()}
        for key in self.eval_running_dict.keys():
            self.eval_running_dict[key] += output_dict[key].item()
        self.validation_epoch_monitor.update(output_dict, monitor_data)
        self._update_modality_validation(input_dict)
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

    def _reduce_max(self, value: torch.Tensor) -> None:
        """Take the maximum of one integrity failure flag across ranks."""
        dist.all_reduce(value, op=dist.ReduceOp.MAX)

    def _check_finite_inputs(self, input_dict) -> None:
        """Stop collectively when a training or validation input is invalid."""
        warn_and_raise_distributed_failure(
            nonfinite_tensor_names(input_dict),
            "pre-training inputs",
            self.rank,
            self.local_rank,
            self._reduce_max,
        )

    def _check_finite_training_loss(self, loss: torch.Tensor) -> None:
        """Stop collectively before backward propagation of invalid loss."""
        warn_and_raise_distributed_failure(
            nonfinite_tensor_names({"loss": loss}),
            "training loss",
            self.rank,
            self.local_rank,
            self._reduce_max,
        )

    def _check_finite_gradients(self) -> None:
        """Stop collectively before an update with invalid gradients."""
        module = getattr(self.model, "module", self.model)
        warn_and_raise_distributed_failure(
            nonfinite_gradient_names(module),
            "training gradients",
            self.rank,
            self.local_rank,
            self._reduce_max,
        )

    @torch.no_grad()
    def _update_modality_validation(self, input_dict) -> None:
        """Evaluate EEG and MEG channel subsets, including mixed EMEG."""
        sensor_type = input_dict["sensor_type"]
        for name, groups in modality_channel_groups(sensor_type).items():
            for sample_tensor, channel_mask in groups:
                modality_input = {
                    "x": input_dict["x"].index_select(
                        0,
                        sample_tensor,
                    )[:, channel_mask],
                    "pos": input_dict["pos"].index_select(
                        0,
                        sample_tensor,
                    )[:, channel_mask],
                    "sensor_type": sensor_type.index_select(
                        0,
                        sample_tensor,
                    )[:, channel_mask],
                }
                _, monitor_data = self.model(
                    **modality_input,
                    return_monitor_data=True,
                )
                self.validation_epoch_monitor.add_modality(
                    name,
                    monitor_data,
                )

    def _write_lightweight_monitors(
        self,
        optimizer_step: int,
        learning_rates,
    ) -> None:
        """Write Stage-2 and optimization scalars at lightweight cadence."""
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
        """Write the Stage-2 update-to-weight diagnostic."""
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
        values = {
            canonical_tag(
                "train",
                "step",
                "optimization",
                "update_to_weight_ratio",
                "global",
            ): update_ratio,
        }
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
        self.train_exposure_monitor.reduce_(self._reduce_sum)
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
            write_scalars(
                self.writer,
                self.train_exposure_monitor.values(),
                self.epoch,
            )
        self.logger.info("")
        if self.eval_running_dict["loss"] < self.best_eval_loss:
            self.best_eval_loss = self.eval_running_dict["loss"]
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
        model = BrainOmni(**self.cfg.get_model_cfg())
        model.load_frozen_tokenizer_ckpt(
            tokenizer_ckpt_path=self.cfg.tokenizer_ckpt_path
        )
        parameter_groups = model.get_named_parameter_groups(
            lr=self.cfg.lr,
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
            p.numel()
            for n, p in model.named_parameters()
            if p.requires_grad and "predict_head" not in n
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

    def build_dataloader(
        self, mode, ratio, batch_size=None, persistent_workers=False
    ):
        return build_brain_bucket_dataloader(
            mode=mode,
            ratio=ratio,
            metadata_path=self.cfg.pretrain_metadata_path,
            accessor=self.accessor,
            rank=self.rank,
            world_size=self.world_size,
            batch_size=batch_size if batch_size else self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            persistent_workers=persistent_workers,
        )

    def build_logger(self):
        if self.rank != 0:
            return EmptyLogger()
        logger = logging.getLogger(name="BrainGPT")
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

"""Accumulate stage-specific pre-training monitor statistics in memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from urllib.parse import quote

import torch

from factory.brain_constant import SENSOR_TYPE_DICT
from factory.pretraining_monitors import (
    TensorSums,
    assignment_metrics,
    canonical_tag,
    checked_ratio,
    covariance_from_sums,
    effective_rank,
    level_name,
    mean_absolute_off_diagonal_correlation,
    reconstruction_statistics,
    residual_energy_reduction,
    source_name,
    source_statistics,
    source_variance,
)

STAGE_ONE_LOSS_KEYS = (
    "loss",
    "time_loss",
    "pcc",
    "amp_loss",
    "phase_loss",
    "commitment_loss",
)
CORRUPTION_NAMES = ("dedicated_mask", "random_token")
MODALITY_NAMES = ("eeg", "meg")
EEG_SENSOR_TYPE = SENSOR_TYPE_DICT["EEG"]
DATA_EXPOSURE_FAMILY = "data"
DATASET_EXPOSURE_METRIC = "dataset_exposure_ratio"
MODALITY_SAMPLE_EXPOSURE_METRIC = "modality_sample_exposure_ratio"
MODALITY_CHANNEL_EXPOSURE_METRIC = "modality_channel_exposure_ratio"
EXPOSURE_MODALITIES = (
    ("eeg", SENSOR_TYPE_DICT["EEG"]),
    ("mag", SENSOR_TYPE_DICT["MAG"]),
    ("grad", SENSOR_TYPE_DICT["GRAD"]),
)


class ExposureAccumulator:
    """Accumulate actual training dataset and modality exposure in memory."""

    def __init__(self, dataset_ids: Sequence[str]) -> None:
        if not dataset_ids:
            raise ValueError(
                "Exposure monitoring requires at least one dataset."
            )
        if any(
            not isinstance(dataset, str) or not dataset
            for dataset in dataset_ids
        ):
            raise ValueError(
                "Exposure monitoring requires non-empty dataset identifiers."
            )
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError(
                "Exposure monitoring received duplicate dataset identifiers."
            )
        self.dataset_ids = tuple(dataset_ids)
        self.sums = TensorSums()

    def update(
        self,
        datasets: Sequence[str],
        sensor_type: torch.Tensor,
    ) -> None:
        """Accumulate one actual training batch.

        Parameters
        ----------
        datasets : Sequence[str]
            Dataset identifier for each batch element.
        sensor_type : torch.Tensor
            Sensor categories with shape ``(batch, channels)``.
        """
        if sensor_type.ndim != 2:
            raise ValueError(
                "Expected sensor types with shape (batch, channels), got "
                f"{tuple(sensor_type.shape)}."
            )
        if len(datasets) != sensor_type.shape[0]:
            raise ValueError(
                "Dataset identifiers must match the batch dimension: "
                f"expected {sensor_type.shape[0]}, got {len(datasets)}."
            )
        counts = torch.zeros(
            len(self.dataset_ids),
            dtype=torch.float64,
            device=sensor_type.device,
        )
        for dataset in datasets:
            if dataset not in self.dataset_ids:
                raise ValueError(
                    "Training batch contains dataset outside the configured "
                    f"exposure set: {dataset!r}."
                )
            counts[self.dataset_ids.index(dataset)] += 1
        self.sums.add("dataset_count", counts)
        self.sums.add("sample_count", counts.sum())
        for modality, sensor_code in EXPOSURE_MODALITIES:
            sensor_mask = sensor_type == sensor_code
            self.sums.add(
                f"{modality}_sample_count",
                sensor_mask.any(dim=1).sum(),
            )
            self.sums.add(
                f"{modality}_channel_count",
                sensor_mask.sum(),
            )

    def reduce_(self, reduce_sum) -> None:
        """Reduce actual-exposure sufficient statistics across ranks."""
        self.sums.reduce_(reduce_sum)

    def values(self) -> dict[str, torch.Tensor]:
        """Return epoch-level TensorBoard scalars for actual exposure ratios."""
        dataset_ratio = checked_ratio(
            self.sums.require("dataset_count"),
            self.sums.require("sample_count"),
            "dataset exposure ratio",
        )
        values = {
            canonical_tag(
                "train",
                "epoch",
                DATA_EXPOSURE_FAMILY,
                DATASET_EXPOSURE_METRIC,
                quote(dataset, safe="._-"),
            ): dataset_ratio[index]
            for index, dataset in enumerate(self.dataset_ids)
        }
        eeg_channel_count = self.sums.require("eeg_channel_count")
        meg_channel_count = (
            self.sums.require("mag_channel_count")
            + self.sums.require("grad_channel_count")
        )
        if eeg_channel_count > 0 and meg_channel_count > 0:
            sample_total = self.sums.require("sample_count")
            channel_total = sum(
                self.sums.require(f"{modality}_channel_count")
                for modality, _ in EXPOSURE_MODALITIES
            )
            for modality, _ in EXPOSURE_MODALITIES:
                sample_count = self.sums.require(
                    f"{modality}_sample_count"
                )
                channel_count = self.sums.require(
                    f"{modality}_channel_count"
                )
                values[canonical_tag(
                    "train",
                    "epoch",
                    DATA_EXPOSURE_FAMILY,
                    MODALITY_SAMPLE_EXPOSURE_METRIC,
                    modality,
                )] = checked_ratio(
                    sample_count,
                    sample_total,
                    f"{modality} sample exposure ratio",
                )
                values[canonical_tag(
                    "train",
                    "epoch",
                    DATA_EXPOSURE_FAMILY,
                    MODALITY_CHANNEL_EXPOSURE_METRIC,
                    modality,
                )] = checked_ratio(
                    channel_count,
                    channel_total,
                    f"{modality} channel exposure ratio",
                )
        return values


class StageOneAccumulator:
    """Accumulate BrainTokenizer statistics without retaining raw tensors."""

    def __init__(self, codebook_size: int) -> None:
        self.codebook_size = codebook_size
        self.sums = TensorSums()

    def update(
        self,
        output: Mapping[str, torch.Tensor],
        monitor: Mapping[str, torch.Tensor],
    ) -> None:
        """Accumulate one BrainTokenizer micro-batch."""
        target = monitor["target"]
        reconstruction = monitor["reconstruction"]
        batch_size = target.shape[0]
        batch_count = target.new_tensor(
            batch_size,
            dtype=torch.float64,
        )
        trace_count = target.new_tensor(
            target.numel() // target.shape[-1],
            dtype=torch.float64,
        )
        weights = {
            "loss": batch_count,
            "time_loss": trace_count,
            "pcc": trace_count,
            "amp_loss": trace_count,
            "phase_loss": trace_count,
            "commitment_loss": monitor["quantization_count"][0],
        }
        for key in STAGE_ONE_LOSS_KEYS:
            self.sums.add(f"objective_{key}_count", weights[key])
            self.sums.add(
                f"objective_{key}_sum",
                output[key] * weights[key],
            )

        dropped = monitor["dropped_channel_mask"].bool()
        sensor_type = monitor["sensor_type"]
        masks = {
            "all": torch.ones_like(dropped),
            "dropped": dropped,
            "visible": ~dropped,
            "eeg": sensor_type == EEG_SENSOR_TYPE,
            "meg": sensor_type != EEG_SENSOR_TYPE,
        }
        for name, value in reconstruction_statistics(
            reconstruction,
            target,
            masks,
        ).items():
            self.sums.add(name, value)
        for name, value in source_statistics(monitor["source_latent"]).items():
            self.sums.add(name, value)

        indices = monitor["indices"]
        if indices.ndim < 2:
            raise ValueError(
                f"Expected RVQ indices with a level axis, got {indices.shape}."
            )
        level_count = indices.shape[-1]
        assignment_counts = []
        for level in range(level_count):
            assignment_counts.append(
                torch.bincount(
                    indices[..., level].reshape(-1),
                    minlength=self.codebook_size,
                )
            )
        self.sums.add(
            "assignment_counts",
            torch.stack(assignment_counts),
        )
        for key in (
            "quantization_error_sum",
            "quantization_count",
            "residual_input_energy_sum",
            "residual_output_energy_sum",
        ):
            self.sums.add(key, monitor[key])

    def reduce_(self, reduce_sum) -> None:
        """Reduce accumulated sufficient statistics across ranks."""
        self.sums.reduce_(reduce_sum)

    def _objective_values(
        self,
        split: str,
        cadence: str,
    ) -> dict[str, torch.Tensor]:
        means = {
            key: checked_ratio(
                self.sums.require(f"objective_{key}_sum"),
                self.sums.require(f"objective_{key}_count"),
                f"BrainTokenizer {key}",
            )
            for key in STAGE_ONE_LOSS_KEYS
        }
        pcc_loss = torch.exp(-means["pcc"])
        specification_loss = (
            means["time_loss"]
            + means["amp_loss"]
            + means["phase_loss"]
            + pcc_loss
            + means["commitment_loss"]
        )
        return {
            canonical_tag(
                split,
                cadence,
                "objective",
                "optimized_loss",
            ): means["loss"],
            canonical_tag(
                split,
                cadence,
                "objective",
                "specification_loss",
            ): specification_loss,
            canonical_tag(
                split,
                cadence,
                "reconstruction",
                "time_loss",
            ): means["time_loss"],
            canonical_tag(
                split,
                cadence,
                "reconstruction",
                "amplitude_loss",
            ): means["amp_loss"],
            canonical_tag(
                split,
                cadence,
                "reconstruction",
                "phase_loss",
            ): means["phase_loss"],
            canonical_tag(
                split,
                cadence,
                "reconstruction",
                "pcc",
            ): means["pcc"],
            canonical_tag(
                split,
                cadence,
                "reconstruction",
                "pcc_loss",
            ): pcc_loss,
            canonical_tag(
                split,
                cadence,
                "rvq",
                "commitment_loss",
            ): means["commitment_loss"],
        }

    def _quantization_values(
        self,
        split: str,
        cadence: str,
    ) -> dict[str, torch.Tensor]:
        errors = checked_ratio(
            self.sums.require("quantization_error_sum"),
            self.sums.require("quantization_count"),
            "per-level RVQ quantization error",
        )
        return {
            canonical_tag(
                split,
                cadence,
                "rvq",
                "quantization_error",
                level_name(level),
            ): errors[level]
            for level in range(errors.numel())
        }

    def _source_variance_values(
        self,
        split: str,
        cadence: str,
    ) -> dict[str, torch.Tensor]:
        variances = source_variance(
            self.sums.require("source_count"),
            self.sums.require("source_sum"),
            self.sums.require("source_square_sum"),
        )
        return {
            canonical_tag(
                split,
                cadence,
                "latent_source",
                "activation_variance",
                source_name(source),
            ): variances[source]
            for source in range(variances.numel())
        }

    def training_values(self, cadence: str) -> dict[str, torch.Tensor]:
        """Return Stage-1 training values for one step or epoch."""
        values = self._objective_values("train", cadence)
        values.update(self._quantization_values("train", cadence))
        values.update(self._source_variance_values("train", cadence))
        if cadence != "epoch":
            return values
        assignment = assignment_metrics(
            self.sums.require("assignment_counts")
        )
        reduction = residual_energy_reduction(
            self.sums.require("residual_input_energy_sum"),
            self.sums.require("residual_output_energy_sum"),
        )
        for level in range(reduction.numel()):
            dimension = level_name(level)
            for metric in (
                "utilization",
                "perplexity",
                "perplexity_normalized",
            ):
                values[
                    canonical_tag(
                        "train",
                        "epoch",
                        "rvq",
                        f"assignment_{metric}",
                        dimension,
                    )
                ] = assignment[metric][level]
            values[
                canonical_tag(
                    "train",
                    "epoch",
                    "rvq",
                    "residual_energy_reduction",
                    dimension,
                )
            ] = reduction[level]
        return values

    def validation_values(self) -> dict[str, torch.Tensor]:
        """Return Stage-1 validation metrics and compact strata."""
        values = self._objective_values("validation", "epoch")
        values.update(self._quantization_values("validation", "epoch"))
        all_count = self.sums.require("all_element_count")
        values[
            canonical_tag(
                "validation",
                "epoch",
                "reconstruction",
                "mae",
            )
        ] = checked_ratio(
            self.sums.require("all_absolute_sum"),
            all_count,
            "validation reconstruction MAE",
        )
        values[
            canonical_tag(
                "validation",
                "epoch",
                "reconstruction",
                "mse",
            )
        ] = checked_ratio(
            self.sums.require("all_squared_sum"),
            all_count,
            "validation reconstruction MSE",
        )
        values[
            canonical_tag(
                "validation",
                "epoch",
                "reconstruction",
                "pcc",
            )
        ] = checked_ratio(
            self.sums.require("all_pcc_sum"),
            self.sums.require("all_pcc_count"),
            "validation reconstruction PCC",
        )
        for stratum in ("dropped", "visible"):
            self._add_reconstruction_stratum(values, stratum)
        eeg_count = self.sums.require("eeg_element_count")
        meg_count = self.sums.require("meg_element_count")
        if eeg_count > 0 and meg_count > 0:
            for stratum in MODALITY_NAMES:
                self._add_reconstruction_stratum(values, stratum)

        covariance = covariance_from_sums(
            self.sums.require("source_count"),
            self.sums.require("source_sum"),
            self.sums.require("source_cross_product"),
        )
        values[
            canonical_tag(
                "validation",
                "epoch",
                "latent_source",
                "mean_absolute_correlation",
            )
        ] = mean_absolute_off_diagonal_correlation(covariance)
        values[
            canonical_tag(
                "validation",
                "epoch",
                "latent_source",
                "effective_rank",
            )
        ] = effective_rank(covariance)

        entropy = assignment_metrics(
            self.sums.require("assignment_counts")
        )["entropy_normalized"]
        for level in range(entropy.numel()):
            values[
                canonical_tag(
                    "validation",
                    "epoch",
                    "rvq",
                    "assignment_entropy_normalized",
                    level_name(level),
                )
            ] = entropy[level]
        values[
            canonical_tag(
                "validation",
                "epoch",
                "rvq",
                "assignment_entropy_normalized",
                "mean",
            )
        ] = entropy.mean()
        return values

    def _add_reconstruction_stratum(
        self,
        values: dict[str, torch.Tensor],
        stratum: str,
    ) -> None:
        element_count = self.sums.require(f"{stratum}_element_count")
        pcc_count = self.sums.require(f"{stratum}_pcc_count")
        values[
            canonical_tag(
                "validation",
                "epoch",
                "reconstruction",
                "time_loss",
                stratum,
            )
        ] = checked_ratio(
            self.sums.require(f"{stratum}_absolute_sum"),
            element_count,
            f"{stratum} reconstruction time loss",
        )
        values[
            canonical_tag(
                "validation",
                "epoch",
                "reconstruction",
                "pcc",
                stratum,
            )
        ] = checked_ratio(
            self.sums.require(f"{stratum}_pcc_sum"),
            pcc_count,
            f"{stratum} reconstruction PCC",
        )


class StageTwoAccumulator:
    """Accumulate BrainOmni masked-token sufficient statistics."""

    def __init__(self) -> None:
        self.sums = TensorSums()

    def update(
        self,
        output: Mapping[str, torch.Tensor],
        monitor: Mapping[str, torch.Tensor],
    ) -> None:
        """Accumulate one BrainOmni micro-batch."""
        del output
        masked_count = monitor["masked_count"]
        level_count = monitor["cross_entropy_sum"].numel()
        self.sums.add(
            "optimized_loss_sum",
            monitor["cross_entropy_sum"].sum(),
        )
        self.sums.add(
            "optimized_loss_count",
            masked_count * level_count,
        )
        for key in (
            "cross_entropy_sum",
            "correct_sum",
            "masked_count",
            "label_counts",
            "corruption_cross_entropy_sum",
            "corruption_count",
            "source_cross_entropy_sum",
            "source_count",
        ):
            self.sums.add(key, monitor[key])
        for modality in MODALITY_NAMES:
            self.sums.add(
                f"{modality}_cross_entropy_sum",
                torch.zeros_like(monitor["cross_entropy_sum"]),
            )
            self.sums.add(
                f"{modality}_count",
                torch.zeros_like(masked_count),
            )

    def add_modality(
        self,
        modality: str,
        monitor: Mapping[str, torch.Tensor],
    ) -> None:
        """Accumulate an additional modality-specific validation forward."""
        if modality not in MODALITY_NAMES:
            raise ValueError(f"Unknown modality monitor {modality!r}.")
        self.sums.add(
            f"{modality}_cross_entropy_sum",
            monitor["cross_entropy_sum"],
        )
        self.sums.add(f"{modality}_count", monitor["masked_count"])

    def reduce_(self, reduce_sum) -> None:
        """Reduce accumulated sufficient statistics across ranks."""
        self.sums.reduce_(reduce_sum)

    def _base_values(
        self,
        split: str,
        cadence: str,
    ) -> dict[str, torch.Tensor]:
        optimized_loss = checked_ratio(
            self.sums.require("optimized_loss_sum"),
            self.sums.require("optimized_loss_count"),
            "BrainOmni optimized loss",
        )
        cross_entropy = checked_ratio(
            self.sums.require("cross_entropy_sum"),
            self.sums.require("masked_count"),
            "per-level masked-token cross entropy",
        )
        accuracy = checked_ratio(
            self.sums.require("correct_sum"),
            self.sums.require("masked_count"),
            "per-level masked-token accuracy",
        )
        values = {
            canonical_tag(
                split,
                cadence,
                "objective",
                "optimized_loss",
            ): optimized_loss,
            canonical_tag(
                split,
                cadence,
                "masked_token",
                "cross_entropy",
                "total",
            ): cross_entropy.sum(),
            canonical_tag(
                split,
                cadence,
                "masked_token",
                "accuracy",
                "mean",
            ): accuracy.mean(),
        }
        for level in range(cross_entropy.numel()):
            dimension = level_name(level)
            values[
                canonical_tag(
                    split,
                    cadence,
                    "masked_token",
                    "cross_entropy",
                    dimension,
                )
            ] = cross_entropy[level]
            values[
                canonical_tag(
                    split,
                    cadence,
                    "masked_token",
                    "accuracy",
                    dimension,
                )
            ] = accuracy[level]
        return values

    def training_values(self, cadence: str) -> dict[str, torch.Tensor]:
        """Return Stage-2 training values for one step or epoch."""
        return self._base_values("train", cadence)

    def validation_values(self) -> dict[str, torch.Tensor]:
        """Return Stage-2 validation metrics and compact strata."""
        values = self._base_values("validation", "epoch")
        counts = self.sums.require("label_counts")
        probabilities = checked_ratio(
            counts,
            counts.sum(dim=-1, keepdim=True),
            "validation token unigram probabilities",
        )
        unigram_ce = -torch.where(
            probabilities > 0,
            probabilities * torch.log(probabilities),
            torch.zeros_like(probabilities),
        ).sum(dim=-1)
        majority_accuracy = probabilities.max(dim=-1).values
        model_ce = checked_ratio(
            self.sums.require("cross_entropy_sum"),
            self.sums.require("masked_count"),
            "validation model cross entropy",
        )
        model_accuracy = checked_ratio(
            self.sums.require("correct_sum"),
            self.sums.require("masked_count"),
            "validation model accuracy",
        )
        for level in range(model_ce.numel()):
            dimension = level_name(level)
            level_values = {
                "unigram_cross_entropy": unigram_ce[level],
                "cross_entropy_improvement": (
                    unigram_ce[level] - model_ce[level]
                ),
                "majority_accuracy": majority_accuracy[level],
                "accuracy_improvement": (
                    model_accuracy[level] - majority_accuracy[level]
                ),
            }
            for metric, value in level_values.items():
                values[
                    canonical_tag(
                        "validation",
                        "epoch",
                        "masked_token",
                        metric,
                        dimension,
                    )
                ] = value

        corruption_ce = checked_ratio(
            self.sums.require("corruption_cross_entropy_sum"),
            self.sums.require("corruption_count").unsqueeze(-1),
            "corruption-specific cross entropy",
        )
        for corruption, name in enumerate(CORRUPTION_NAMES):
            for level in range(corruption_ce.shape[1]):
                values[
                    canonical_tag(
                        "validation",
                        "epoch",
                        "masked_token",
                        "cross_entropy",
                        f"{level_name(level)}_{name}",
                    )
                ] = corruption_ce[corruption, level]

        source_ce = checked_ratio(
            self.sums.require("source_cross_entropy_sum"),
            self.sums.require("source_count").unsqueeze(0),
            "source-specific cross entropy",
        )
        for level in range(source_ce.shape[0]):
            for source in range(source_ce.shape[1]):
                values[
                    canonical_tag(
                        "validation",
                        "epoch",
                        "masked_token",
                        "cross_entropy",
                        (
                            f"{level_name(level)}_"
                            f"{source_name(source)}"
                        ),
                    )
                ] = source_ce[level, source]

        modality_available = all(
            f"{modality}_count" in self.sums.values
            and self.sums.require(f"{modality}_count") > 0
            for modality in MODALITY_NAMES
        )
        if modality_available:
            for modality in MODALITY_NAMES:
                modality_ce = checked_ratio(
                    self.sums.require(
                        f"{modality}_cross_entropy_sum"
                    ),
                    self.sums.require(f"{modality}_count"),
                    f"{modality} masked-token cross entropy",
                )
                for level in range(modality_ce.numel()):
                    values[
                        canonical_tag(
                            "validation",
                            "epoch",
                            "masked_token",
                            "cross_entropy",
                            f"{level_name(level)}_{modality}",
                        )
                    ] = modality_ce[level]
        return values

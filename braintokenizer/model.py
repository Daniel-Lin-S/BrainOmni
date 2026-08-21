import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from braintokenizer.constants import DEFAULT_NOISE_STD
from model_utils.attn import RMSNorm
from model_utils.loss import get_time_loss, get_pcc, get_frequency_domain_loss
from model_utils.module import (
    BrainSensorModule,
    BrainTokenizerEncoder,
    BrainQuantizer,
    BrainTokenizerDecoder,
)


class BrainTokenizer(nn.Module):
    """
    BrainTokenizer model for compressing multi-channel brain signals into discrete tokens.

    This model uses a VQ-VAE architecture with a SEANet-based encoder and decoder,
    supplemented by a sensor embedding module to handle varied sensor types and positions.

    Attributes
    ----------
    window_length : int
        The temporal length of each signal window.
    n_dim : int
        The dimension of the latent feature space.
    sensor_embed : BrainSensorModule
        Module for encoding sensor positions and types.
    mask_ratio : float
        Ratio of channels to mask during training to improve robustness.
    noise_std : float
        Gaussian input-noise standard deviation used only during training.
    encoder : BrainTokenizerEncoder
        Spatial-temporal encoder for brain signals.
    quantizer : BrainQuantizer
        Vector quantization module for discretizing features.
    decoder : BrainTokenizerDecoder
        Spatial-temporal decoder for signal reconstruction.
    """

    def __init__(
        self,
        window_length,
        n_filters,
        ratios,
        kernel_size,
        last_kernel_size,
        n_dim,
        n_neuro,
        n_head,
        dropout,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        rotation_trick: bool,
        quantize_optimize_method: str,
        **kwargs,
    ):
        """
        Parameters
        ----------
        codebook_dim : int
            Dimension of the codebook vectors, if different to n_dim,
            latent features will be projected linearly into codewords,
            and projected linearly out before decoding.
        codebook_size : int
            Number of codewords in the codebook.
        num_quantizers : int
            Number of quantizers to use in the model.
        """
        super().__init__()
        self.window_length = window_length
        self.n_dim = n_dim
        self.mask_ratio = kwargs.get("channel_mask_ratio", 0.25)
        self.noise_std = kwargs.get("noise_std", DEFAULT_NOISE_STD)
        if (
            not isinstance(self.mask_ratio, (int, float))
            or isinstance(self.mask_ratio, bool)
            or not math.isfinite(self.mask_ratio)
            or not 0 <= self.mask_ratio < 1
        ):
            raise ValueError(
                "Expected channel_mask_ratio in [0, 1), got "
                f"{self.mask_ratio!r}."
            )
        if (
            not isinstance(self.noise_std, (int, float))
            or isinstance(self.noise_std, bool)
            or not math.isfinite(self.noise_std)
            or self.noise_std < 0
        ):
            raise ValueError(
                "Expected finite noise_std >= 0, got "
                f"{self.noise_std!r}."
            )

        self.sensor_embed = BrainSensorModule(n_dim)
        self.encoder = BrainTokenizerEncoder(
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            n_dim=n_dim,
            n_neuro=n_neuro,
            n_head=n_head,
            dropout=dropout,
        )
        self.quantizer = BrainQuantizer(
            n_dim=n_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
            quantize_optimize_method=quantize_optimize_method,
        )
        self.decoder = BrainTokenizerDecoder(
            n_dim=n_dim,
            n_head=n_head,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            dropout=dropout,
        )
        # --------------------------------------------------------------------------
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            if isinstance(m.weight, nn.Parameter):
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)

    @torch.jit.ignore
    def get_parameters_groups(self, lr: float, codebook_lr: float, weight_decay: float):
        normal_params = []
        no_decay_params = []
        codebook_params = []
        for n, p in self.named_parameters():
            if p.requires_grad:
                if "norm" in n or n in [
                    "sensor_embed.sensor_embedding_layer.weight",
                ]:
                    no_decay_params.append(p)
                elif "quantizer" in n:
                    codebook_params.append(p)
                else:
                    normal_params.append(p)
        return [
            {"params": normal_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
            {"params": codebook_params, "lr": codebook_lr, "weight_decay": 0.0},
        ]

    def unfold(self, x: torch.Tensor, overlap_ratio: float = 0.0):
        if x.shape[-1] < self.window_length:
            x = F.pad(x, pad=(0, self.window_length - x.shape[-1]))
        if overlap_ratio > 0.0:
            stride = int(self.window_length * (1 - overlap_ratio))
            right_remain = (x.shape[-1] - self.window_length) % stride
            if right_remain > 0:
                x = F.pad(x, pad=(0, stride - right_remain))
        return x.unfold(
            dimension=-1,
            size=self.window_length,
            step=int(self.window_length * (1 - overlap_ratio)),
        )

    def norm_target(self, x: torch.Tensor):
        """
        x: B C N L
        """
        x = x.float()
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1,keepdim=True)+1e-6)
        return x

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add configured Gaussian noise only while training.

        Parameters
        ----------
        x : torch.Tensor
            Visible input windows of shape
            ``(batch, channels, windows, samples)``.

        Returns
        -------
        torch.Tensor
            Training input with Gaussian noise, or the unchanged input during
            evaluation and when ``noise_std`` is zero.
        """
        if not self.training or self.noise_std == 0:
            return x
        return x + torch.randn_like(x) * self.noise_std

    def masked_channel_count(self, channel_count: int) -> int:
        """Return a safe number of input channels to hide.

        Parameters
        ----------
        channel_count : int
            Number of channels in the current batch.

        Returns
        -------
        int
            Number of channels to hide before encoding.
        """
        if channel_count <= 0:
            raise ValueError(
                "BrainTokenizer requires at least one input channel, got "
                f"{channel_count}."
            )
        if self.mask_ratio == 0:
            return 0
        if channel_count == 1:
            raise ValueError(
                "Positive channel masking requires at least two input "
                f"channels, got {channel_count}."
            )
        return max(int(channel_count * self.mask_ratio), 1)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        return_monitor_data: bool = False,
        **kwargs,
    ):
        """
        Forward pass for training the VQ-VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input brain signal of shape (Batch, Channels, Time).
        pos : torch.Tensor
            Sensor coordinates and orientations of shape (Batch, Channels, 6).
        sensor_type : torch.Tensor
            Sensor category indices of shape (Batch, Channels).
        return_monitor_data : bool, optional
            Return transient sufficient-statistic inputs, by default False.
        **kwargs
            Additional arguments.

        Returns
        -------
        dict[str, torch.Tensor]
            The optimisation metrics:
            - loss: Total weighted loss for backpropagation.
            - time_loss: Temporal reconstruction L1 loss.
            - pcc: Waveform correlation coefficient.
            - amp_loss: Frequency domain amplitude match.
            - phase_loss: Frequency domain phase match.
            - commitment_loss: RVQ codebook stability loss.
            - judge_loss: Detached total loss for performance monitoring.
        torch.Tensor
            Discrete indices with shape
            (Batch, Channels, Window, num_quantizers).
        dict[str, torch.Tensor], optional
            Monitoring payload returned only when ``return_monitor_data`` is
            True. The default two-element return contract is unchanged.
        """
        x = self.unfold(x)

        sensor_embedding = self.sensor_embed(pos, sensor_type)
        random_index = torch.randperm(x.shape[1], device=x.device)
        x = x.index_select(dim=1, index=random_index)
        sensor_embedding = sensor_embedding.index_select(dim=1, index=random_index)
        n_mask_channel = self.masked_channel_count(x.shape[1])
        source_latent = self.encoder(
            self.add_noise(x[:, n_mask_channel:]),
            sensor_embedding[:, n_mask_channel:],
        )

        if return_monitor_data:
            feature, indices, commitment_loss, rvq_monitor = self.quantizer(
                source_latent,
                return_monitor_data=True,
            )
        else:
            feature, indices, commitment_loss = self.quantizer(source_latent)

        x_rec = self.decoder(feature, sensor_embedding)

        x_rec = x_rec.float()
        x = self.norm_target(x)

        time_loss = get_time_loss(x_rec, x)
        pcc = get_pcc(x_rec, x)
        amp_loss, phase_loss = get_frequency_domain_loss(x_rec, x)
        output = {
            "loss": time_loss
            + torch.exp(-pcc)
            + commitment_loss
            + amp_loss
            + 0.5 * phase_loss,
            "time_loss": time_loss.detach(),
            "pcc": pcc.detach(),
            "amp_loss": amp_loss.detach(),
            "phase_loss": phase_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
            "judge_loss": (
                time_loss
                + torch.exp(-pcc)
                + commitment_loss
                + amp_loss
                + 0.5 * phase_loss
            ).detach(),
        }
        if return_monitor_data:
            dropped_mask = torch.zeros(
                x.shape[:2],
                dtype=torch.bool,
                device=x.device,
            )
            dropped_mask[:, :n_mask_channel] = True
            monitor_data = {
                "target": x.detach(),
                "reconstruction": x_rec.detach(),
                "dropped_channel_mask": dropped_mask,
                "sensor_type": sensor_type.index_select(
                    dim=1,
                    index=random_index,
                ).detach(),
                "source_latent": source_latent.detach(),
                "indices": indices.detach(),
                **rvq_monitor,
            }
            return output, indices, monitor_data
        return output, indices

    @torch.no_grad()
    def monitor_attention(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> torch.Tensor:
        """Return source-query sensor attention for one validation batch.

        Parameters
        ----------
        x : torch.Tensor
            Brain signals of shape ``(batch, channels, samples)``.
        pos : torch.Tensor
            Sensor geometry of shape ``(batch, channels, 6)``.
        sensor_type : torch.Tensor
            Sensor categories of shape ``(batch, channels)``.

        Returns
        -------
        torch.Tensor
            Attention weights with shape
            ``(batch, windows, heads, sources, channels)``.
        """
        x = self.unfold(x)
        sensor_embedding = self.sensor_embed(pos, sensor_type)
        _, attention = self.encoder(
            x,
            sensor_embedding,
            return_attention=True,
        )
        return attention

    @torch.no_grad()
    def visualize(
        self, x: torch.Tensor, pos: torch.Tensor, sensor_type: torch.Tensor, **kwargs
    ):
        """
        A function that reconstructs input signal for visualisation purpose.

        Parameters
        ----------
        x : torch.Tensor
            Input brain signal of shape (Batch, Channels, Time).
        pos : torch.Tensor
            Sensor coordinates and orientations of shape (Batch, Channels, 6).
        sensor_type : torch.Tensor
            Sensor category indices of shape (Batch, Channels).
        
        Return
        ------
        Dict[str, torch.Tensor]
            key-value pairs:
            - x : torch.Tensor, the normalised input signal of shape (Batch, Channels, Time).
            - x_rec : torch.Tensor, the reconstructed signal of the same shape as x.
            - sensor_type : torch.Tensor, the sensor category indices of shape (Batch, Channels).
        """
        x = self.unfold(x)
        sensor_embedding = self.sensor_embed(pos, sensor_type)
        feature = self.encoder(x, sensor_embedding)
        feature, indices, commitment_loss = self.quantizer(feature)
        x_rec = self.decoder(feature, sensor_embedding)
        return {
            "x": self.norm_target(x),
            "x_rec": x_rec.float(),
            "sensor_type": sensor_type,
        }

    @torch.no_grad()
    def tokenize(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        overlap_ratio: float,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert brain signals into discrete latent tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input brain signal of shape (Batch, Channels, Time).
        pos : torch.Tensor
            Sensor coordinates and orientations of shape (Batch, Channels, 6).
        sensor_type : torch.Tensor
            Sensor category indices of shape (Batch, Channels).
        overlap_ratio : float
            Overlap ratio between consecutive temporal windows.
        **kwargs
            Additional arguments.

        Returns
        -------
        features : torch.Tensor
            Continuous latent representations of shape (Batch, Channels, Window, n_dim).
        indices : torch.Tensor
            Discrete codebook indices of shape (Batch, Channels, Window, num_quantizers).
        """
        self.eval()
        x = self.unfold(x, overlap_ratio=overlap_ratio)
        sensor_embedding = self.sensor_embed(pos, sensor_type)
        feature = self.encoder(x, sensor_embedding)
        feature, indices, commitment_loss = self.quantizer(feature)
        feature = rearrange(feature, "B C N T D->B C (N T) D")
        indices = rearrange(indices, "B C N T Q -> B C (N T) Q")
        return feature, indices

    def get_finetune_parameter_groups(self, weight_decay, layer_decay):
        del self.decoder
        del self.quantizer
        parameter_groups = {}

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue

            this_weight_decay = weight_decay
            group_name = "decay"

            # Create group if it doesn't exist
            if group_name not in parameter_groups:
                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": layer_decay,
                }

            parameter_groups[group_name]["params"].append(p)

        return list(parameter_groups.values())

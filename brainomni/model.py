import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from braintokenizer.model import BrainTokenizer
from model_utils.attn import RMSNorm, SpatialTemporalAttentionBlock

DEDICATED_MASK_PROBABILITY = 0.8


class BrainOmni(nn.Module):
    """
    BrainOmni model for generative pre-training on brain signals.

    This model combines a VQ-VAE based BrainTokenizer with a Transformer-based Large Latent Model (LLM)
    to perform masked modeling on discrete latent representations of multi-channel brain signals.

    Attributes
    ----------
    lm_dim : int
        Dimension of the latent model embeddings.
    window_length : int
        Length of the window used for signal tokenization.
    overlap_ratio : float
        Overlap ratio between consecutive windows during tokenization.
    mask_ratio : float
        Ratio of tokens to be masked or perturbed during pre-training.
    num_quantizers_used : int
        Number of codebook levels used for prediction.
    tokenizer : BrainTokenizer
        The tokenizer module used to map raw signals to discrete latent tokens.
    mask_token : nn.Parameter
        Trainable parameter used to represent masked positions.
    projection : nn.Module
        Linear layer to project tokenizer dimensions to latent model dimensions.
    blocks : nn.ModuleList
        List of spatial-temporal attention blocks.
    predict_head : nn.Linear
        Linear layer to predict discrete tokens from latent representations.
    """

    def __init__(
        self,
        # tokenizer parameter
        window_length: int,
        n_filters: int,
        ratios: List[int],
        kernel_size: int,
        last_kernel_size: int,
        n_dim: int,
        n_head: int,
        n_neuro: int,
        dropout: float,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        rotation_trick: bool,
        quantize_optimize_method: str,
        # lm model parameter
        overlap_ratio: float,
        lm_dim: int,
        lm_head: int,
        lm_depth: int,
        lm_dropout: float,
        mask_ratio: float,
        num_quantizers_used: int,
        **kwargs,
    ):
        """
        Initialize the BrainOmni model.

        Parameters
        ----------
        window_length : int
            The length of the temporal window for each token.
        n_filters : int
            Number of filters in the tokenizer's encoder.
        ratios : List[int]
            Downsampling ratios for the tokenizer.
        kernel_size : int
            Kernel size for tokenizer convolutions.
        last_kernel_size : int
            Kernel size for the last layer of the tokenizer.
        n_dim : int
            Dimension of the tokenizer's latent space.
        n_head : int
            Number of attention heads in the tokenizer.
        n_neuro : int
            Number of neurons/channels expected by the tokenizer.
        dropout : float
            Dropout rate for the tokenizer.
        codebook_dim : int
            Dimension of the discrete codebook entries.
        codebook_size : int
            Total number of entries in each codebook.
        num_quantizers : int
            Total number of quantizers (levels) in the tokenizer.
        rotation_trick : bool
            Whether to use the rotation trick in quantization.
        quantize_optimize_method : str
            Optimization method for the codebook.
        overlap_ratio : float
            Overlap ratio between windows during encoding.
        lm_dim : int
            Embedding dimension of the Transformer blocks.
        lm_head : int
            Number of attention heads in the Transformer blocks.
        lm_depth : int
            Number of Transformer blocks.
        lm_dropout : float
            Dropout rate for the Transformer blocks.
        mask_ratio : float
            Probability of masking a token during training.
        num_quantizers_used : int
            How many quantizer levels to predict in the sequence modeling objective.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__()
        self.lm_dim = lm_dim
        self.window_length = window_length
        self.overlap_ratio = overlap_ratio
        self.mask_ratio = mask_ratio
        self.num_quantizers_used = (
            num_quantizers_used if num_quantizers_used != None else num_quantizers
        )
        # B C T -> unfold -> B C T' -> tokenizer -> (B C) W D -> next predict
        self.tokenizer = BrainTokenizer(
            window_length,
            n_filters,
            ratios,
            kernel_size,
            last_kernel_size,
            n_dim,
            n_neuro,
            n_head,
            dropout,
            codebook_dim,
            codebook_size,
            num_quantizers,
            rotation_trick,
            quantize_optimize_method,
        )
        self.mask_token = nn.Parameter(torch.randn(n_dim))
        self.projection = nn.Linear(n_dim, lm_dim) if n_dim != lm_dim else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                SpatialTemporalAttentionBlock(lm_dim, lm_head, lm_dropout, causal=False)
                for _ in range(lm_depth)
            ]
        )
        self.predict_head = nn.Linear(lm_dim, num_quantizers_used * codebook_size)
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
    def load_frozen_tokenizer_ckpt(self, tokenizer_ckpt_path: str):
        """
        Load a pre-trained tokenizer checkpoint and freeze its parameters.

        Parameters
        ----------
        tokenizer_ckpt_path : str
            Path to the saved state dictionary of the BrainTokenizer.
        """
        self.tokenizer.load_state_dict(
            torch.load(tokenizer_ckpt_path, weights_only=True)
        )
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        return None

    @torch.jit.ignore
    def get_named_parameter_groups(
        self,
        lr: float,
        weight_decay: float,
    ) -> dict[str, dict[str, object]]:
        """Return ordered, named, non-empty optimizer parameter groups.

        Parameters
        ----------
        lr : float
            Learning rate for every trainable parameter.
        weight_decay : float
            Weight decay for main parameters.

        Returns
        -------
        dict[str, dict[str, object]]
            Ordered mapping from monitor name to DeepSpeed group settings.
        """
        no_decay_params = []
        normal_params = []
        for n, p in self.named_parameters():
            if p.requires_grad:
                if (
                    "norm" in n
                    or "predict_head" in n
                    or n in ["projection.weight", "projection.bias", "mask_token"]
                ):
                    no_decay_params.append(p)
                else:
                    normal_params.append(p)

        candidates = {
            "main": {
                "params": normal_params,
                "lr": lr,
                "weight_decay": weight_decay,
            },
            "no_decay": {
                "params": no_decay_params,
                "lr": lr,
                "weight_decay": 0.0,
            },
        }
        return {
            name: group
            for name, group in candidates.items()
            if group["params"]
        }

    @torch.jit.ignore
    def get_parameters_groups(
        self,
        lr: float,
        weight_decay: float,
    ) -> list[dict[str, object]]:
        """Return non-empty parameter groups for external optimizers."""
        groups = self.get_named_parameter_groups(
            lr=lr,
            weight_decay=weight_decay,
        )
        return list(groups.values())

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        return_monitor_data: bool = False,
        **kwargs,
    ):
        """
        Forward pass for pre-training using masked modeling.

        Parameters
        ----------
        x : torch.Tensor
            Brain signal tensor of shape (Batch, Channels, Time).
        pos : torch.Tensor
            Sensor position coordinates of shape (Batch, Channels, 6).
            2 sets of Cartesian coordinates, 3 for location, 3 for orientation.
        sensor_type : torch.Tensor
            Sensor type indicators of shape (Batch, Channels).
        return_monitor_data : bool, optional
            Return transient sufficient statistics, by default False.
        **kwargs
            Additional arguments.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing total loss and per-level accuracies.
        dict[str, torch.Tensor], optional
            Monitoring payload returned only when ``return_monitor_data`` is
            True. The default dictionary return contract is unchanged.
        """
        x, label_indices = self.tokenizer.tokenize(
            x, pos, sensor_type, self.overlap_ratio
        )

        B, C, W, D = x.shape

        mask = (
            torch.rand(size=(B, C, W), device=x.device) > self.mask_ratio
        )  # true in mask will be preserve, false in mask will be masked
        # 20% random select a token from the minibatch
        x = torch.where(
            mask.unsqueeze(-1).repeat(1, 1, 1, D),
            x,
            rearrange(
                x.view(-1, D)[torch.randperm(B * C * W, device=x.device)],
                "(B C W) D -> B C W D",
                B=B,
                C=C,
            ),
        )
        # 80% use mask token
        tmp_mask = (
            mask.float() + torch.rand(size=(B, C, W), device=x.device)
        ) > DEDICATED_MASK_PROBABILITY
        random_replacement_mask = ~mask & tmp_mask
        tmp_mask_values = tmp_mask.unsqueeze(-1).type_as(x)
        mask_token = self.mask_token.type_as(x)
        x = x * tmp_mask_values + mask_token * (1 - tmp_mask_values)

        neuro = self.tokenizer.encoder.neuros.type_as(x).detach().view(1, C, 1, -1)
        x = x + neuro

        x = self.projection(x)

        for block in self.blocks:
            x = block(x)

        # (batch channel) window (num_quant logit_dim)  -> batch channel window num_quant logit_dim
        logits = rearrange(
            self.predict_head(x),
            "B C W (N D) -> B C W N D",
            N=self.num_quantizers_used,
        )
        if return_monitor_data:
            loss, acc, token_loss, correct = self._compute_cross_entropy(
                logits,
                label_indices,
                mask,
                return_details=True,
            )
        else:
            loss, acc = self._compute_cross_entropy(
                logits,
                label_indices,
                mask,
            )
        output_dict = {"loss": loss, "acc_all": acc.mean()}
        for i in range(self.num_quantizers_used):
            output_dict[f"acc_{i}"] = acc[i]
        if return_monitor_data:
            monitor_data = self._monitor_statistics(
                token_loss,
                correct,
                label_indices[..., : self.num_quantizers_used],
                ~mask,
                random_replacement_mask,
            )
            return output_dict, monitor_data
        return output_dict

    def _monitor_statistics(
        self,
        token_loss: torch.Tensor,
        correct: torch.Tensor,
        labels: torch.Tensor,
        selected_mask: torch.Tensor,
        random_replacement_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return masked-token sufficient statistics for monitoring."""
        level_count = token_loss.shape[-1]
        selected_loss = token_loss[selected_mask]
        selected_correct = correct[selected_mask]
        selected_labels = labels[selected_mask]
        selected_count = selected_mask.sum().double()
        if selected_count <= 0:
            raise ValueError(
                "Masked-token monitoring found no masked positions."
            )
        label_counts = []
        for level in range(level_count):
            label_counts.append(
                torch.bincount(
                    selected_labels[:, level],
                    minlength=self.predict_head.out_features // level_count,
                )
            )

        dedicated_mask = selected_mask & ~random_replacement_mask
        corruption_masks = (dedicated_mask, random_replacement_mask)
        corruption_loss_sums = []
        corruption_counts = []
        for corruption_mask in corruption_masks:
            corruption_loss_sums.append(
                token_loss[corruption_mask].double().sum(dim=0)
            )
            corruption_counts.append(corruption_mask.sum().double())

        source_loss_sum = (
            token_loss.double()
            * selected_mask.unsqueeze(-1)
        ).sum(dim=(0, 2)).transpose(0, 1)
        source_count = selected_mask.sum(dim=(0, 2)).double()
        return {
            "cross_entropy_sum": selected_loss.double().sum(dim=0),
            "correct_sum": selected_correct.double().sum(dim=0),
            "masked_count": selected_count,
            "label_counts": torch.stack(label_counts).double(),
            "corruption_cross_entropy_sum": torch.stack(
                corruption_loss_sums
            ),
            "corruption_count": torch.stack(corruption_counts),
            "source_cross_entropy_sum": source_loss_sum,
            "source_count": source_count,
        }

    def encode(
            self, x: torch.Tensor, pos: torch.Tensor, sensor_type: torch.Tensor
        ) -> torch.Tensor:
        """
        Extract normalised latent features from brain signals.
        Used for evaluation.

        Parameters
        ----------
        x : torch.Tensor
            Brain signal tensor of shape (Batch, Channels, Time).
        pos : torch.Tensor
            Sensor position coordinates of shape (Batch, Channels, 6).
        sensor_type : torch.Tensor
            Sensor type indicators of shape (Batch, Channels).
            Each indicator should be one of the following 3 types:
            0 - EEG, 1 - gradiometer, 2 - magnetometer.

        Returns
        -------
        torch.Tensor
            Normalised latent representations of shape (Batch, Channels, Window, lm_dim).
            Window - sequence length of latent tokens (temporal dimension),
            lm_dim - embedding size of the model.
        """
        x, label_indices = self.tokenizer.tokenize(
            x, pos, sensor_type, self.overlap_ratio
        )

        B, C, W, _ = x.shape
        neuro = self.tokenizer.encoder.neuros.type_as(x).detach().view(1, C, 1, -1)
        x = x + neuro
        x = self.projection(x)

        for block in self.blocks[:-1]:
            x = block(x)

        return F.normalize(
            x,
            p=2.0,
            dim=-1,
            eps=1e-6,
        )

    def _compute_cross_entropy(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor,
        return_details: bool = False,
    ):
        """
        Compute cross-entropy loss for masked tokens and calculate level-wise accuracy.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits of shape (Batch, Channels, Window, num_quantizers, codebook_size).
        label : torch.Tensor
            Target codebook indices of shape (Batch, Channels, Window, num_quantizers).
        mask : torch.Tensor
            Boolean mask indicating which tokens are NOT masked (True = preserved, False = masked).
            Note: The loss is computed on the elements where mask is False (standard for MAE style).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing (mean cross-entropy loss, level-wise accuracy tensor).
        """
        if not return_details:
            level_count = label.shape[-1]
            selected_logits = logits[~mask]
            selected_label = label[~mask]
            selected_logits = rearrange(
                selected_logits,
                "X N M -> (X N) M",
            )
            selected_label = selected_label.view(-1)
            if selected_label.numel() == 0:
                raise ValueError(
                    "Masked-token objective requires at least one masked "
                    "position."
                )
            loss = F.cross_entropy(
                selected_logits.float(),
                selected_label,
                reduction="mean",
            )
            accuracy = rearrange(
                selected_logits.argmax(dim=-1) == selected_label,
                "(X N) -> N X",
                N=level_count,
            ).float().mean(dim=-1)
            return loss, accuracy

        level_count = logits.shape[-2]
        label = label[..., :level_count]
        flat_logits = rearrange(logits, "B C W N M -> (B C W N) M")
        flat_label = rearrange(label, "B C W N -> (B C W N)")
        token_loss = F.cross_entropy(
            flat_logits.float(),
            flat_label,
            reduction="none",
        ).view(*label.shape)
        correct = logits.argmax(dim=-1) == label
        selected_loss = token_loss[~mask]
        selected_correct = correct[~mask]
        if selected_loss.numel() == 0:
            raise ValueError(
                "Masked-token objective requires at least one masked position."
            )
        loss = selected_loss.mean()
        acc = selected_correct.float().mean(dim=0)
        if return_details:
            return loss, acc, token_loss, correct
        return loss, acc

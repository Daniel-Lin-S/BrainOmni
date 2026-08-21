# Pre-training configuration reference

This is the public, authoritative schema reference for BrainTokenizer and BrainOmni. Update it with every schema, validation, default, or artifact change.

## 1. Configuration precedence and launch flow

```bash
bash script/train_braintokenizer.sh --num-gpus N --config configs/pretrain/braintokenizer.yaml LOCAL_FILE
```

All `--config` files merge left-to-right; repeatable `--set` overrides apply
last. Validation runs only after the final complete merge.

The required local dataset catalog is auto-loaded from
`configs/data/datasets.local.yaml`; users do not pass it at launch. Copy the
tracked `configs/data/datasets.yaml` template and set each dataset's `path` and
`signal_type` (`eeg`, `meg`, or `both`). The catalog is the sole modality
source. `included_datasets: ["*"]` selects every catalog entry; an explicit
list selects only those entries. Each selected root is independently scanned
and every recording must match its catalog modality.

```yaml
datasets:
  MEG-MASC:
    path: <local path>
    signal_type: meg
```
There is no generic Stage-2 default: select `brainomni_tiny.yaml` or
`brainomni_base.yaml` explicitly.

Stage 2 uses an explicit paper-aligned architecture choice: tiny is 256 hidden
dimensions, 8 heads, 12 layers, and `5e-4`; base is 512 dimensions, 16 heads,
12 layers, and `4e-4`.

```bash
bash script/train_brainomni.sh --num-gpus N --config configs/pretrain/brainomni_tiny.yaml LOCAL_FILE
```

```bash
bash script/train_brainomni.sh --num-gpus N --config configs/pretrain/brainomni_base.yaml LOCAL_FILE
```

## 2. Campaign-wide settings

Campaign settings are checkpoint-semantic and are saved in `pretrain_setting.yaml`.

| Key | Type/default | Effect |
| --- | --- | --- |
| `schema_version` | integer, `1` | Selects this strict schema. |
| `campaign.stage` | enum; stage-specific | Selects the Stage-1 or Stage-2 contract. |
| `campaign.seed` | integer >= 0, `42` | Seeds Python, NumPy, and Torch. |
| `campaign.data.included_datasets` | non-empty string list, `["*"]` | `["*"]` selects all datasets; an explicit list filters source data. Artifacts save the concrete result. |
| `campaign.data.split_ratios.train` | fraction, `.85` | Metadata fraction assigned to training. |
| `campaign.data.split_ratios.validation` | fraction, `.10` | Metadata fraction assigned to validation. |
| `campaign.data.split_ratios.test` | fraction, `.05` | Metadata fraction assigned to testing. |
| `campaign.data.preprocessing.sample_rate_hz` | positive Hz, `256` | Resampling rate. |
| `campaign.data.preprocessing.low_frequency_hz` | non-negative Hz, `.1` | Band-pass lower cutoff. |
| `campaign.data.preprocessing.high_frequency_hz` | Hz below Nyquist, `96` | Band-pass upper cutoff. |
| `campaign.data.preprocessing.segment_seconds` | positive seconds; `10`/`30` | Raw preprocessing block length. |
| `campaign.data.preprocessing.stride_seconds` | positive seconds; `10`/`30` | Spacing between raw blocks. |
| `campaign.training.epochs` | positive integer; `16`/`32` | Number of optimization epochs. |
| `campaign.training.global_batch_size` | positive integer; `512`/`256` | Effective distributed batch. |

## 3. Stage-1 BrainTokenizer settings

| Key | Type/default | Effect |
| --- | --- | --- |
| `campaign.model.window_length` | positive samples, `512` | Length of each tokenizer window. |
| `campaign.model.n_filters` | positive integer, `32` | Initial convolution width. |
| `campaign.model.ratios` | positive integer list, `[8,4,2]` | Encoder/decoder downsampling ratios. |
| `campaign.model.kernel_size` | positive integer, `5` | Encoder convolution kernel length. |
| `campaign.model.last_kernel_size` | positive integer, `5` | Final encoder/decoder kernel length. |
| `campaign.model.n_dim` | positive integer, `256` | Latent embedding width. |
| `campaign.model.n_neuro` | positive integer, `16` | Learned sensor-source embedding count. |
| `campaign.model.n_head` | positive integer, `4` | Tokenizer attention heads. |
| `campaign.model.dropout` | fraction, `0.0` | Tokenizer dropout probability. |
| `campaign.model.codebook_dim` | positive integer, `256` | RVQ vector width. |
| `campaign.model.codebook_size` | positive integer, `512` | Entries per RVQ codebook. |
| `campaign.model.num_quantizers` | positive integer, `4` | Residual vector-quantizer levels. |
| `campaign.model.rotation_trick` | boolean, `true` | Uses the rotation-gradient estimator. |
| `campaign.model.quantize_optimize_method` | `ema`, `ema` | Codebook update method. |
| `campaign.objective.channel_mask_ratio` | fraction [0,1), `.25` | Randomly hidden input-channel share; `0` hides none, while positive values require at least two channels. |
| `campaign.objective.noise_std` | non-negative float, `.1` | Gaussian input-noise standard deviation used only during training. |
| `campaign.optimizer.type` | `AdamW`, `AdamW` | Optimizer implementation. |
| `campaign.optimizer.lr` | positive float, `2e-4` | Main parameter learning rate. |
| `campaign.optimizer.codebook_lr` | positive float, `3e-4` | Learning rate for trainable quantizer projections; EMA codebook buffers are not optimized. |
| `campaign.optimizer.betas` | two fractions [0,1), `[.5,.9]` | AdamW moment coefficients. |
| `campaign.optimizer.eps` | positive float, `1e-5` | AdamW numerical stabilizer. |
| `campaign.optimizer.weight_decay` | non-negative float, `.01` | Main-parameter L2 regularization. |
| `campaign.scheduler.warmup_ratio` | fraction [0,1], `.1` | Fraction of optimizer steps spent warming up. |
| `campaign.scheduler.warmup_min_lr_ratio` | fraction [0,1], `0` | Learning-rate multiplier at the start of warmup. |
| `campaign.scheduler.cosine_min_ratio` | fraction [0,1], `.05` | Final cosine LR fraction. |

## 4. Stage-2 BrainOmni settings

| Key | Type/default | Effect |
| --- | --- | --- |
| `campaign.model.lm_dim` | positive integer, `256` tiny / `512` base | Transformer hidden width. |
| `campaign.model.lm_head` | positive integer, `8` tiny / `16` base | Transformer attention-head count. |
| `campaign.model.lm_depth` | positive integer, `12` | Transformer block count. |
| `campaign.model.lm_dropout` | fraction, `.1` | Transformer dropout probability. |
| `campaign.objective.overlap_ratio` | fraction [0,1], `.25` | Overlap between tokenizer windows. |
| `campaign.objective.mask_ratio` | fraction [0,1], `.5` | Positions selected for masked-token prediction. |
| `campaign.objective.num_quantizers_used` | positive integer, `4` | RVQ levels predicted by the token head. |
| `campaign.optimizer.type` | `AdamW`, `AdamW` | Optimizer implementation. |
| `campaign.optimizer.lr` | positive float, `5e-4` tiny / `4e-4` base | Transformer learning rate. |
| `campaign.optimizer.betas` | two fractions [0,1), `[.9,.95]` | AdamW moment coefficients. |
| `campaign.optimizer.eps` | positive float, `1e-6` | AdamW numerical stabilizer. |
| `campaign.optimizer.weight_decay` | non-negative float, `.05` | Transformer L2 regularization. |
| `campaign.scheduler.warmup_ratio` | fraction [0,1], `.1` | Fraction of optimizer steps spent warming up. |
| `campaign.scheduler.warmup_min_lr_ratio` | fraction [0,1], `0` | Learning-rate multiplier at the start of warmup. |
| `campaign.scheduler.cosine_min_ratio` | fraction [0,1], `.1` | Final cosine LR fraction. |

The Stage-2 architecture is defined solely by the four `lm_*` fields, rather
than by a redundant preset label. The parent tokenizer architecture comes from
its `model_cfg.json`; its weights and configuration digests identify the parent.

## 5. Invocation settings

Invocation settings affect execution, not model semantics. They are saved only
inside the current attempt's `invocation.yaml`.

| Key | Type/default | Effect |
| --- | --- | --- |
| `invocation.data_catalog` | local mapping | Auto-loaded dataset paths and signal types. |
| `invocation.processed_root` | local path, `null` | Processed tensor destination. |
| `invocation.metadata_root` | local path, `null` | Preprocessing metadata destination. |
| `invocation.output_root` | local path, `null` | Campaign artifact destination. |
| `invocation.run_name` | string; stage-specific | Human-readable label stored with each invocation. |
| `invocation.tokenizer_path` | campaign root, `null` (Stage 2) | Health-checked frozen BrainTokenizer campaign. |
| `invocation.expected_tokenizer_model_config_sha256` | SHA-256 or `null` | Optional parent architecture check. |
| `invocation.expected_tokenizer_weights_sha256` | SHA-256 or `null` | Optional canonical parent tensor-state check. |
| `invocation.batch_size_per_gpu` | positive integer, `16`/`8` | Per-rank micro-batch. |
| `invocation.num_workers` | positive integer, `32` | Data-loader workers. |
| `invocation.preprocess_workers` | positive integer, `32` | Raw-recording workers. |
| `invocation.held_out_evaluation_datasets` | dataset list, `[]` | Catalog datasets excluded from training and evaluated after completion. |
| `invocation.checkpoint_interval_epochs` | positive integer, `20` | Retained epoch-checkpoint interval. |
| `invocation.visualization_interval_steps` | positive integer, `200` (Stage 1) | Reconstruction visualization interval. |
| `invocation.monitoring.lightweight_interval_steps` | positive integer, `100` | Successful optimizer-update interval for lightweight scalar monitors. |
| `invocation.monitoring.diagnostic_interval_steps` | positive integer, `500` | Successful optimizer-update interval for sparse diagnostics. |
| `invocation.deepspeed.bf16.enabled` | boolean, `true` | Enables bfloat16 runtime. |
| `invocation.deepspeed.bf16.auto_cast` | boolean, `true` | Enables automatic bfloat16 casting. |
| `invocation.deepspeed.bf16.loss_scale` | float, `0` | Dynamic loss-scale control. |
| `invocation.deepspeed.bf16.initial_scale_power` | integer, `16` | Initial dynamic-scale exponent. |
| `invocation.deepspeed.bf16.loss_scale_window` | integer, `1000` | Stable-step scale-update window. |
| `invocation.deepspeed.bf16.hysteresis` | integer, `2` | Overflows tolerated before scale reduction. |
| `invocation.deepspeed.bf16.min_loss_scale` | float, `1` | Minimum dynamic loss scale. |
| `invocation.deepspeed.zero_optimization.stage` | integer, `2` | Selects ZeRO stage 2. |
| `invocation.deepspeed.zero_optimization.offload_optimizer_device` | `cpu` | Optimizer-state placement. |
| `invocation.deepspeed.zero_optimization.offload_optimizer_pin_memory` | boolean | Pins offloaded optimizer memory. |
| `invocation.deepspeed.zero_optimization.overlap_comm` | boolean | Overlaps communication and computation. |
| `invocation.deepspeed.zero_optimization.allgather_partitions` | boolean | All-gathers parameter partitions. |
| `invocation.deepspeed.zero_optimization.allgather_bucket_size` | integer or `auto` | All-gather bucket size. |
| `invocation.deepspeed.zero_optimization.reduce_scatter` | boolean | Uses reduce-scatter gradients. |
| `invocation.deepspeed.zero_optimization.reduce_bucket_size` | integer or `auto` | Reduce bucket size. |

The random `test` split is always evaluated. It is in-distribution data held
out from optimization and validation-based checkpoint selection. Optional
held-out datasets are entire catalog datasets absent from train, validation,
and test. Missing compatible held-out preprocessing is recorded and skipped;
the warning gives the exact preprocessing command.

Each preprocessing worker runs its MNE filtering and resampling steps with
`n_jobs=1`. Parallelism comes only from `invocation.preprocess_workers`, which
avoids nested joblib worker pools and their temporary-resource cleanup warnings.

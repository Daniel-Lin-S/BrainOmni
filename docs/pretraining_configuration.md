# Pre-training configuration reference

This is the public, authoritative schema reference for BrainTokenizer and
BrainOmni. Update it with every schema, validation, default, or artifact change.

Machine paths, dataset selections, hardware choices, tokens, paper replication,
and machine-specific launch instructions belong only in ignored `*.local.yaml`
and `*.local.md` files. They are intentionally not documented here.

## 1. Configuration precedence and launch flow

```bash
deepspeed --num_gpus=N braintokenizer/launcher.py --config configs/pretrain/braintokenizer.yaml LOCAL_FILE
```

All `--config` files merge left-to-right; pass an ignored local overlay after
the tracked layers. Repeated `--set section.key=<JSON value>` overrides merge
last. Validation runs only after the final complete merge.
There is no generic Stage-2 default: select `brainomni_tiny.yaml` or
`brainomni_base.yaml` explicitly.

Stage 2 uses an explicit paper-aligned architecture choice: tiny is 256 hidden
dimensions, 8 heads, 12 layers, and `5e-4`; base is 512 dimensions, 16 heads,
12 layers, and `4e-4`.

```bash
deepspeed --num_gpus=N brainomni/launcher.py --config configs/pretrain/brainomni_tiny.yaml LOCAL_FILE
```

```bash
deepspeed --num_gpus=N brainomni/launcher.py --config configs/pretrain/brainomni_base.yaml LOCAL_FILE
```

## 2. Campaign-wide settings

Campaign settings are checkpoint-semantic and are saved in `pretrain_setting.yaml`.

| Key | Type/default | Effect |
| --- | --- | --- |
| `schema_version` | integer, `1` | Selects this strict schema. |
| `campaign.stage` | enum; stage-specific | Selects the Stage-1 or Stage-2 contract. |
| `campaign.seed` | integer >= 0, `42` | Seeds Python, NumPy, and Torch. |
| `campaign.data.signal_type` | `eeg`, `meg`, or `both`; `both` | Filters metadata by modality. |
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
| `campaign.objective.channel_mask_ratio` | fraction [0,1], `.25` | Randomly masked input-channel share. |
| `campaign.optimizer.type` | `AdamW`, `AdamW` | Optimizer implementation. |
| `campaign.optimizer.lr` | positive float, `2e-4` | Main parameter learning rate. |
| `campaign.optimizer.codebook_lr` | positive float, `3e-4` | RVQ codebook learning rate. |
| `campaign.optimizer.betas` | two fractions [0,1), `[.5,.9]` | AdamW moment coefficients. |
| `campaign.optimizer.eps` | positive float, `1e-5` | AdamW numerical stabilizer. |
| `campaign.optimizer.weight_decay` | non-negative float, `.01` | Main-parameter L2 regularization. |
| `campaign.scheduler.warmup_ratio` | fraction [0,1], `.1` | Initial warmup fraction. |
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
| `campaign.scheduler.warmup_ratio` | fraction [0,1], `.1` | Initial warmup fraction. |
| `campaign.scheduler.cosine_min_ratio` | fraction [0,1], `.1` | Final cosine LR fraction. |

The Stage-2 architecture is defined solely by the four `lm_*` fields, rather
than by a redundant preset label. The parent tokenizer architecture comes from
its `model_cfg.json`; its weights and configuration digests identify the parent.

## 5. Invocation settings

Invocation settings affect execution only and are saved in a run's `invocation.yaml`.

| Key | Type/default | Effect |
| --- | --- | --- |
| `invocation.raw_root` | non-empty local string, `null` | Source recordings; required in a local config layer. |
| `invocation.processed_root` | non-empty local string, `null` | Processed tensor destination. |
| `invocation.metadata_root` | non-empty local string, `null` | Preprocessing metadata destination. |
| `invocation.output_root` | non-empty local string, `null` | Run-artifact destination. |
| `invocation.run_name` | non-empty string; stage-specific | Run-directory prefix. |
| `invocation.tokenizer_path` | non-empty local string, `null` (Stage 2) | Frozen tokenizer artifact directory. |
| `invocation.expected_tokenizer_model_config_sha256` | SHA-256 or `null`, `null` | Optional parent architecture integrity check. |
| `invocation.expected_tokenizer_weights_sha256` | SHA-256 or `null`, `null` | Optional parent weight integrity check. |
| `invocation.batch_size_per_gpu` | positive integer, `16`/`8` | Per-rank micro-batch. |
| `invocation.num_workers` | positive integer, `32` | Data-loader worker count. |
| `invocation.preprocess_workers` | positive integer, `32` | Concurrent raw-file processors. |
| `invocation.evaluation_modes` | string list, stage-specific | Metadata partitions evaluated by the trainer. |
| `invocation.checkpoint_interval_epochs` | positive integer, `20` | Checkpoint save interval. |
| `invocation.visualization_interval_steps` | positive integer, `200` (Stage 1) | Reconstruction visualization interval. |
| `invocation.deepspeed.bf16.enabled` | boolean, `true` | Enables bfloat16 runtime. |
| `invocation.deepspeed.bf16.auto_cast` | boolean, `true` | Enables automatic bfloat16 casting. |
| `invocation.deepspeed.bf16.loss_scale` | float, `0` | Dynamic loss-scale control. |
| `invocation.deepspeed.bf16.initial_scale_power` | integer, `16` | Initial dynamic-scale exponent. |
| `invocation.deepspeed.bf16.loss_scale_window` | positive integer, `1000` | Stable-step scale-update window. |
| `invocation.deepspeed.bf16.hysteresis` | non-negative integer, `2` | Overflows tolerated before scale reduction. |
| `invocation.deepspeed.bf16.min_loss_scale` | positive float, `1` | Minimum dynamic loss scale. |
| `invocation.deepspeed.zero_optimization.stage` | integer, `2` | Selects ZeRO stage 2. |
| `invocation.deepspeed.zero_optimization.offload_optimizer_device` | `cpu`, `cpu` | Optimizer-state placement. |
| `invocation.deepspeed.zero_optimization.offload_optimizer_pin_memory` | boolean, `true` | Pins offloaded optimizer memory. |
| `invocation.deepspeed.zero_optimization.overlap_comm` | boolean, `false` | Overlaps communication and computation. |
| `invocation.deepspeed.zero_optimization.allgather_partitions` | boolean, `true` | All-gathers parameter partitions. |
| `invocation.deepspeed.zero_optimization.allgather_bucket_size` | integer or `auto`, `auto` | All-gather communication bucket. |
| `invocation.deepspeed.zero_optimization.reduce_scatter` | boolean, `true` | Uses reduce-scatter gradients. |
| `invocation.deepspeed.zero_optimization.reduce_bucket_size` | integer or `auto`, `auto` | Reduce communication bucket. |

## 6. Resolution, artifacts, and validation

All YAML layers merge left-to-right, then repeatable `--set` overrides apply.
Unknown or missing keys, invalid ranges, non-integer
dimensions or batch values, non-boolean flags, malformed lowercase SHA-256
values, invalid split fractions, and incompatible global batches fail early.

`model_cfg.json` contains model constructor architecture. `pretrain_setting.yaml`
contains resolved checkpoint-semantic settings without duplicated architecture.
`campaign.data.included_datasets: ["*"]` selects all datasets during
preprocessing. An explicit list selects only those dataset identities.
Launchers replace either source selection with sorted identities from
`info.json`; every run artifact then records that concrete list.

`pretrain_setting.json` contains integrity digests and, for Stage 2, parent
tokenizer identity. `invocation.yaml` is written only in the run directory.

Portable export copies weights, `model_cfg.json`, and `pretrain_setting.*`; it
excludes `invocation.yaml`. Operational commands belong only in an ignored
local guide.

`model_cfg.json`, `pretrain_setting.yaml`, and `invocation.yaml` are written
from the fully merged configuration. Before a run artifact is written, the
dataset source selection is replaced with the sorted, non-empty identities in
preprocessing `info.json`; `included_datasets` can never be `["*"]`, `null`,
or `['']` in an artifact. The two optional tokenizer integrity checks may be
`null` to disable those checks.

At successful Stage-1 completion, all ranks synchronize and rank zero converts
the best DeepSpeed checkpoint to `BrainTokenizer.pt` in the run directory.
That run directory and its `model_cfg.json` are the direct Stage-2 handoff.

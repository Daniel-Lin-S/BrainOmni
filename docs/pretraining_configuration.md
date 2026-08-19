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

## 6. Campaign artifacts and recovery

A semantic hash identifies the resolved model, data split, preprocessing,
objective, optimizer, scheduler, global batch, epochs, seed, and Stage-2 parent.
Changing `campaign.training.epochs` creates a new campaign trained from scratch.
Only an interrupted exact identity can resume.

```text
<output_root>/<stage>/<semantic-hash>/
├── campaign_identity.json      full semantic identity and artifact schema
├── model_cfg.json              model constructor architecture
├── pretrain_setting.yaml       portable resolved training semantics
├── pretrain_setting.json       sidecar and Stage-2 parent integrity digests
├── split_manifest.json         exact train, validation, and test digests
├── campaign_status.json        completion, portable digest, and repair history
├── checkpoint/
│   ├── manifest.json           checkpoint file sizes and integrity digests
│   ├── latest/                 exact interruption-recovery state
│   ├── best/                   best validation model and optimizer state
│   ├── epoch_<n>/              retained scheduled checkpoint
│   └── failed_recovery_<id>/   temporary unusable state during retraining
├── BrainTokenizer.pt | BrainOmni.pt  completed stage portable weights
├── evaluations/
│   ├── index.json              completed and skipped evaluation records
│   ├── metrics_test_set.json   mandatory in-distribution test metrics
│   └── metrics_heldout_<dataset>.json  one optional held-out result
└── attempts/<attempt-id>/
    ├── invocation.yaml         local runtime settings and launch provenance
    ├── status.json             invocation outcome and repair action
    └── tensorboard/            invocation-specific TensorBoard events

logs/<stage>/<semantic-hash>/<attempt-id>/
├── terminal.log                launcher standard output and error
└── logs.txt                    rank-zero trainer text messages
```

Both portable weight files are health-checked against their architecture and
canonical tensor-state digest. A damaged file is atomically reconstructed from
the verified best checkpoint. If training-side repair fails, unusable state is
retained while the exact campaign restarts; it is removed only after new best
and portable checkpoints validate successfully.

BrainOmni automatically health-checks `invocation.tokenizer_path`. Downstream
loading automatically health-checks a BrainOmni campaign root. Consumer repair
may reconstruct portable weights, but never starts training. Use the CPU-only
repair command when directed:

```bash
bash script/repair_pretrain_campaign.sh --campaign-root CAMPAIGN
```

Preprocessing reuses recording work while isolating semantic splits:

```text
<metadata_root>/<preprocessing-id>/
├── info.json                   processed segment identities and modalities
├── finish.json                 raw recordings completed successfully
└── splits_<split-id>/
    ├── train.json              optimization segment metadata
    ├── val.json                checkpoint-selection segment metadata
    ├── test.json               mandatory in-distribution test metadata
    └── <dataset>.json          whole-dataset held-out metadata
```

The preprocessing identity excludes local paths. The split identity includes
training dataset IDs and signal types, seed, and split ratios, but excludes the
invocation-only held-out list. Adding held-out evaluations therefore reuses the
same training splits without admitting those datasets into training.

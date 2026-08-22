# Campaign artifacts and recovery

This reference describes the artifacts produced by a resolved pre-training
campaign and the lifecycle of its terminal logs. Configuration fields and
defaults remain in [Pre-training configuration](pretraining_configuration.md).

## Campaign identity and artifacts

A semantic hash identifies the resolved model, data split, preprocessing,
objective, optimizer, scheduler, global batch, epochs, seed, and Stage-2
parent. Changing any of those settings creates a new campaign. Only an
interrupted campaign with the same identity can resume.

```text
<output_root>/<stage>/<semantic-hash>/
├── campaign_identity.json      full semantic identity and artifact schema
├── model_cfg.json              model constructor architecture
├── pretrain_setting.yaml       portable resolved training semantics
├── pretrain_setting.json       sidecar and parent integrity digests
├── split_manifest.json         train, validation, and test digests
├── campaign_status.json        completion, portable digest, and repairs
├── checkpoint/
│   ├── manifest.json           checkpoint file sizes and integrity digests
│   ├── latest/                 exact interruption-recovery state
│   ├── best/                   best validation model and optimizer state
│   ├── epoch_<n>/              retained scheduled checkpoint
│   └── failed_recovery_<id>/   unusable state retained for inspection
├── BrainTokenizer.pt | BrainOmni.pt  completed portable weights
├── evaluations/
│   ├── index.json              completed and skipped evaluation records
│   ├── metrics_test_set.json   mandatory in-distribution test metrics
│   └── metrics_heldout_<dataset>.json  optional held-out result
└── attempts/<attempt-id>/
    ├── invocation.yaml         local runtime settings and launch provenance
    ├── status.json             invocation outcome and repair action
    └── tensorboard/            invocation-specific TensorBoard events
```

Both portable weight files are checked against their architecture and canonical
tensor-state digest. A damaged file can be reconstructed atomically from the
verified best checkpoint. Consolidation writes the exact portable file directly
and does not require model-hub or sharded-serialization dependencies. If
training-side repair fails, the unusable checkpoint state remains under
`failed_recovery_<id>` until a newly trained best checkpoint and portable weight
validate.

BrainOmni checks `invocation.tokenizer_path` before Stage 2, and downstream
loading checks a BrainOmni campaign root. Consumer repair never starts
training. To validate or repair a completed campaign, run:

```bash
bash script/repair_pretrain_campaign.sh --campaign-root CAMPAIGN
```

## Terminal-log lifecycle

Terminal logs are repository-local operational diagnostics. They are NOT
portable checkpoint artifacts. The launcher creates a unique attempt log before
it can know a semantic campaign hash, so it first writes to:

```text
logs/<stage>/pending/<attempt-id>/terminal.log
```

`pending` is only a short-lived staging location. It is never the final state
of a completed or failed launcher command. The empty staging directory is
removed when its terminal log moves.

Before the launched command starts, its terminal log records the ordered,
absolute configuration paths. Interactive progress updates contain carriage
returns and remain visible in the terminal, but are excluded from terminal logs.

| Outcome | Final terminal-log location | Meaning |
| --- | --- | --- |
| Stage succeeds | `logs/<stage>/complete/<attempt-id>/terminal.log` | Training completed successfully. |
| Stage fails | `logs/<stage>/failed/<attempt-id>/terminal.log` | Training stopped with a nonzero exit status. |

`<stage>` can be one of preproces, braintokenizer, brainomni.

The parent launcher moves the complete attempt log directory only after
DeepSpeed exits, so no pending attempt directory remains after either outcome. Training
`logs.txt`, when present, moves alongside `terminal.log`. The campaign artifact
`attempts/<attempt-id>/status.json` remains the authoritative structured
record of campaign identity and outcome.

## Preprocessing metadata

Preprocessing reuses recording work while isolating semantic splits:

```text
<metadata_root>/<preprocessing-id>/
├── info.json                   processed segment identities and modalities
├── finish.json                 raw recordings completed successfully
├── dataset_snapshots.json      aggregate and per-dataset recording snapshots
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

## Monitoring artifacts

Each attempt's `tensorboard/` directory contains the monitoring record. New
scalar tags follow
`<split>/<cadence>/<family>/<metric>[/<dimension>]`.

The training workflow does not create monitor CSV files. It stores only
TensorBoard scalars and the existing reconstruction figures. Use the on-demand exporter `script/export_pretraining_monitors.py` when a tabular input (csv) is useful for another visualisation tool.
Integrity warnings for invalid samples or non-finite training values are not
written as TensorBoard, CSV, or campaign-status artifacts.

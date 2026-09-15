# BrainOmni Pre-training Quick Start

There are 2 stages for BrainOmni pre-training -- BrainTokenizer, and BrainOmni. They use different preprocessing pipelines.

## Step 1. Preprocessing

Please keep local paths to your datasets in a `.local.yaml` file. Preprocessing scans the union of `campaign.data.included_datasets` and `invocation.held_out_evaluation_datasets`. Nested or hidden directories like `derivatives` are ignored. MNE reads split FIF recordings through their first part. Every retained recording must match its catalog modality after applying the configured channel exclusions. Held-out datasets receive the same preprocessing and separate whole-dataset JSON metadata. Please see `pretraining_configuration.md` for more details on how to set the configuration parameters.

Prepare the recordings before training, use the following command to start pre-processing:

```bash
bash script/pretrain_preprocess.sh \
  --config configs/pretrain/braintokenizer.yaml LOCAL_FILE
```

Cache reuse rejects changed or unrecorded exclusions; use fresh `processed_root` and `metadata_root` directories for changed settings.

For Stage 2, pass the selected BrainOmni configuration and its local overlay
to the same preprocessing command.

## Step 2. Training

Train Stage 1 with the prepared BrainTokenizer configuration `configs/pretrain/braintokenizer.yaml`, and place paths to preprocessed data and checkpoints in a `.local.yaml` file. Run the following command:

```bash
bash script/train_braintokenizer.sh --num-gpus N --config configs/pretrain/braintokenizer.yaml LOCAL_FILE
```

There is no generic Stage-2 default: select `brainomni_tiny.yaml` or
`brainomni_base.yaml` explicitly.

```bash
bash script/train_brainomni.sh --num-gpus N --config configs/pretrain/brainomni_tiny.yaml LOCAL_FILE
```

```bash
bash script/train_brainomni.sh --num-gpus N --config configs/pretrain/brainomni_base.yaml LOCAL_FILE
```

Training results are saved under `output_root`, in a folder named by a hash key generated from the semantic configurations (not including non-semantic invocation configurations like number of GPUs used).

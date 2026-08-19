# AGENTS.md

BrainOmni is a multi-modal foundation model for neural recordings EEG and MEG.

Pre-training: there are 2 stages, `braintokenizer` is first trained to convert brain signals into codewords, and then the second stage trains on the frozen BrainTokenizer using masked latent token reconstruction to learn spatiotemporal semantic in the latent space.

Fine-tuning: A pre-trained BrainOmni model can be used for downstream classification tasks (as see module `downstream`)

## Boundaries and Constraints

NEVER reveal local paths on public configuration files or scripts (this include tests). You should ALWAYS keep local paths, tokens in untracked files like `.local.yaml`.

## Training artifacts and provenance

ALL functional semantic configurations should be saved along with the outputs. (for both pre-training and fine-tuning)

The configurations saved should be the ACTUAL parameter used, NOT the user input. For example, if user did not include a parameter with default, then the default value should be saved, rather than saving 'none' or not saving the parameter.

Invocation parameters (e.g., `batch_size_per_gpu`, log directory, dataset path) that doesn't change the ultimate model should NOT be saved with the portable checkpoints. And invocation parameters should ALWAYS be separated from semantic parameters.

## Documentation

If any implementation of pre-training is changed -- configuration structure, workflow, artifact structure, please remember to update `docs/pretraining_configuration.md`.

Documentation should be CONCISE, only exposing features that users should know rather than full technical details.

When necessary, guide user to the specific class or function's documentation rather than stacking all details in a single documentation under `docs/`.
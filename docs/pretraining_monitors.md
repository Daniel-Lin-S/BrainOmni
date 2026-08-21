# BrainOmni Training Monitors

## Stored format and tag grammar

Training writes scalar summaries to the existing attempt-specific TensorBoard
directory. Scalar tags written by new runs use
`<split>/<cadence>/<family>/<metric>[/<dimension>]`, where split is `train`,
`validation`, or `evaluation`, and cadence is `step`, `epoch`, or
`micro_step`. Indexed dimensions are zero padded, for example `level_00` and
`source_00`. Historical event files are not changed; the extraction interface
maps their flat tags to this grammar at read time.

Only scalar sufficient statistics and the existing reconstruction figures are
persisted. Activations, attention matrices, code assignments, optimizer
partitions, and parameter snapshots exist only transiently in memory.
Training reconstruction figures use
`train/micro_step/visualization/reconstruction`; final figures use
`evaluation/visualization/reconstruction/<encoded-dataset>`.

Cadenced steps count successful DeepSpeed optimizer updates. Gradient
accumulation micro-batches and skipped updates do not advance that count.
The defaults are configured under `invocation.monitoring` and are saved in the
attempt's `invocation.yaml`; they do not affect campaign identity.

## Extracting scalar events

Export one or more campaign or TensorBoard directories to long-form CSV only
when needed:

```bash
python -m script.export_pretraining_monitors \
  --event-dir /absolute/path/to/campaign-a \
  --event-dir /absolute/path/to/campaign-b \
  --output /absolute/path/to/monitors.csv
```

Python callers can use
`factory.pretraining_monitor_events.load_monitor_events` directly. It returns
rows with the run, original and normalized tags, tag components, optimizer or
epoch step, wall time, and scalar value. It recursively discovers event files,
and prefers a legacy `loss` event over a same-step `judge_loss` alias.

For fully custom plotting, TensorBoard's API is also straightforward:

```python
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

events = EventAccumulator(
    "/absolute/path/to/events.out.tfevents.example",
    size_guidance={"scalars": 0},
)
events.Reload()
points = events.Scalars("train/step/objective/optimized_loss")
steps = [point.step for point in points]
values = [point.value for point in points]
```

## Cadence conventions

The tables use explicit recommended cadences, however the exact interval should be configurable.

- **Every 100 optimiser steps**: lightweight quantities useful for following the training trajectory.
- **Every 500 optimiser steps**: more expensive diagnostic quantities that do not need dense sampling.
- **validation: once per epoch**: metrics requiring a representative held-out estimate on validation data, once per epoch.
- **aggregate per epoch** (aggregated): statistics that require many samples for a meaningful estimate.

The exact step intervals can be changed according to run length and computational budget; the important distinction is their relative frequency.

---

# 1. Cross-stage shared monitors

These monitors characterise optimisation dynamics and apply to both **Stage 1: BrainTokenizer** and **Stage 2: BrainOmni**.

## 1.1 Optimisation

| Monitor | Purpose | Mathematical definition | Cadence | Incremental cost |
|---|---|---|---|---|
| **Learning rate** | Relate changes in learning behaviour to the warm-up and cosine-decay schedule | Current optimiser learning rate \(\eta_t\) | Every 100 optimiser steps | Negligible |
| **Global gradient norm** | Detect exploding gradients, unusually weak gradients, or abrupt optimisation instability | \(G_t=\sqrt{\sum_p \lVert\nabla_{\theta_p}L_t\rVert_2^2}\), where \(p\) indexes trainable parameter tensors | Every 100 optimiser steps | Low–moderate; requires reduction over all gradients |
| **Update-to-weight ratio** | Measure effective parameter movement rather than gradient magnitude alone | \(R_t=\lVert\theta_{t+1}-\theta_t\rVert_2 / (\lVert\theta_t\rVert_2+\epsilon)\) | Every 500 optimiser steps | Moderate; requires parameter/update reductions |

---

# 2. Stage 1 — BrainTokenizer

BrainTokenizer is trained through signal reconstruction while compressing arbitrary sensor configurations into latent-source variables and quantising them using four-layer RVQ.

The total objective is

\[
L_{\mathrm{token}}
=
L_{\mathrm{time}}
+
L_{\mathrm{freq}}
+
L_{\mathrm{pcc}}
+
L_{\mathrm{rvq}},
\]

where

\[
L_{\mathrm{freq}}
=
L_{\mathrm{amp}}
+
L_{\mathrm{phase}}.
\]

Stage-1 monitoring is divided into:

1. reconstruction learning;
2. latent-source representation;
3. RVQ learning.

---

## 2.1 Reconstruction learning

| Monitor | Purpose | Mathematical definition | Cadence | Incremental cost |
|---|---|---|---|---|
| **Total tokenizer loss** | Monitor overall Stage-1 convergence | \(L_{\mathrm{token}}=L_{\mathrm{time}}+L_{\mathrm{amp}}+L_{\mathrm{phase}}+L_{\mathrm{pcc}}+L_{\mathrm{rvq}}\) | Training: every 100 optimiser steps; validation: once per epoch | Negligible; already required for optimisation |
| **Time-domain loss** | Monitor reconstruction of the original waveform | \(L_{\mathrm{time}}=\lVert X-\hat X\rVert_1\) | Training: every 100 optimiser steps; validation: once per epoch | Negligible |
| **Amplitude-spectrum loss** | Monitor reconstruction of spectral amplitudes independently of phase | \(L_{\mathrm{amp}}=\lVert A-\hat A\rVert_1\) | Training: every 100 optimiser steps; validation: once per epoch | Negligible if available before constructing \(L_{\mathrm{freq}}\) |
| **Phase-spectrum loss** | Monitor reconstruction of spectral phase independently of amplitude | \(L_{\mathrm{phase}}=\lVert\Phi-\hat\Phi\rVert_1\) | Training: every 100 optimiser steps; validation: once per epoch | Negligible if available before constructing \(L_{\mathrm{freq}}\) |
| **PCC loss** | Monitor the waveform-trend component that directly contributes to the training objective | \(L_{\mathrm{pcc}}=\exp[-\rho(X,\hat X)]\) | Training: every 100 optimiser steps; validation: once per epoch | Negligible |
| **Raw PCC** | Express waveform-shape agreement on an interpretable scale rather than through the transformed PCC loss | \(\rho(X,\hat X)=\operatorname{Cov}(X,\hat X) / [\sigma_X\sigma_{\hat X}]\) | Validation: once per epoch | Low |
| **MAE** | Measure absolute reconstruction error directly | \(\mathrm{MAE}=N^{-1}\sum_{j=1}^{N}\operatorname{abs}(X_j-\hat X_j)\) | Validation: once per epoch | Low |
| **MSE** | Complement L1 reconstruction with stronger sensitivity to large errors | \(\mathrm{MSE}=N^{-1}\sum_{j=1}^{N}(X_j-\hat X_j)^2\) | Validation: once per epoch | Low |

If a loss is turned off by user, please DONT add a monitor for that.

### Reconstruction stratification

Partition selected reconstruction metrics so that different learning behaviours are not hidden by a global average.

| Monitor | Purpose | Mathematical definition | Cadence | Incremental cost |
|---|---|---|---|---|
| **Dropped-vs-visible channel reconstruction** | Distinguish reconstruction requiring cross-channel inference from reconstruction of channels already available to the encoder | For reconstruction metric \(M\), compute \(M_{\mathrm{drop}}=M(X_{\mathcal D},\hat X_{\mathcal D})\) and \(M_{\mathrm{vis}}=M(X_{\mathcal V},\hat X_{\mathcal V})\), where \(\mathcal D\) and \(\mathcal V\) are dropped and visible channel sets | Validation: once per epoch | Low; no additional model forward pass |
| **EEG-vs-MEG reconstruction** | Detect asymmetric learning between the two modalities during joint EMEG training | Compute selected reconstruction metric \(M\) independently over EEG validation samples and MEG validation samples | Validation: once per epoch | Negligible–low; no additional forward pass if modality labels are retained |

For the stratifications, it is not necessary to duplicate every reconstruction metric. A compact implementation could stratify the primary reconstruction loss and raw PCC.

For EEG-only or MEG-only pre-training, the second metric should be SKIPPED.

---

## 2.2 Latent-source representation

Let the pre-RVQ latent representation be

\[
Z_{\mathrm{src}}
\in
\mathbb{R}^{B\times C'\times W\times D},
\]

where \(C'=16\) is the number of BrainOmni latent-source variables.

The following monitors test whether these source variables remain active and non-redundant as training proceeds.

| Monitor | Purpose | Mathematical definition | Cadence | Incremental cost |
|---|---|---|---|---|
| **Per-source activation variance** | Detect latent-source variables that become nearly constant or inactive | For source \(s\), \(v_s=\operatorname{Var}_{b,w,d}(Z_{\mathrm{src},b,s,w,d})\) | Every 100 optimiser steps; aggregate per epoch | Low |
| **Inter-source correlation** | Detect different source variables learning strongly redundant representations | Flatten observations for each source into \(z_s\); compute \(R_{ij}=\operatorname{corr}(z_i,z_j)\). Summarize by the mean absolute off-diagonal correlation | Validation: once per epoch | Low |
| **Latent-source effective rank** | Quantify global dimensional collapse that may not be obvious from individual pairwise correlations | Let \(\lambda_i\) be eigenvalues of the source covariance matrix and \(p_i=\lambda_i/\sum_j\lambda_j\). Then \(r_{\mathrm{eff}}=\exp[-\sum_i p_i\log p_i]\) | Validation: once per epoch | Low; the source-level covariance is only \(16\times16\) |
| **Inter-query attention similarity** | Detect different latent-source queries converging to nearly identical sensor aggregation patterns | For attention vectors \(a_i,a_j\), \(S_{ij}=a_i^\top a_j / [\lVert a_i\rVert_2\lVert a_j\rVert_2]\); summarise off-diagonal similarities | Every 500 optimiser steps on a fixed small validation subset | Moderate |

---

## 2.3 RVQ learning

BrainTokenizer uses sequential RVQ levels. Let:

- integer \(l\) denote RVQ level;
- \(K\) denote the codebook size;
- \(n_{lk}\) denote the number of assignments to code \(k\) at level \(l\) during the aggregation interval;
- \(r_l\) denote the residual entering RVQ level \(l\);
- \(e_{q_l}\) denote the selected code vector.

All RVQ monitors should be reported **separately for the four RVQ levels**.

| Monitor | Purpose | Mathematical definition | Cadence | Incremental cost |
|---|---|---|---|---|
| **Codebook utilisation** | Detect codebooks where substantial portions of the vocabulary are never selected | Define \(K_l^{\mathrm{used}}=\#\{k:n_{lk}>0\}\). Then \(U_l=K_l^{\mathrm{used}}/K\) | Accumulate assignments over each epoch; report once per epoch | Low |
| **Assignment perplexity** | Detect highly imbalanced code usage even when most codes are technically active | \(p_{lk}=n_{lk}/\sum_jn_{lj}\), \(H_l=-\sum_kp_{lk}\log p_{lk}\), and \(\mathrm{PPL}_l=\exp(H_l)\). Normalized perplexity is \(\mathrm{PPL}_l/K\) | Accumulate assignments over each epoch; report once per epoch | Low |
| **Per-level quantisation error** | Measure how accurately each RVQ codebook represents the residual supplied to it | \(Q_l=\mathbb E[\lVert r_l-e_{q_l}\rVert_2^2]\) | Training: every 100 optimiser steps; validation: once per epoch | Low |
| **Residual-energy reduction** | Measure whether each successive RVQ level materially improves representation fidelity | With \(r_{l+1}=r_l-e_{q_l}\), define \(G_l=1-\mathbb E[\lVert r_{l+1}\rVert_2^2]/\mathbb E[\lVert r_l\rVert_2^2]\) | Accumulate over each epoch; report once per epoch | Low |
| **Codebook update magnitude** | Track whether EMA-updated codebooks become effectively stationary or change unusually rapidly | \(\Delta_l^{(t)}=\lVert E_l^{(t)}-E_l^{(t-1)}\rVert_F / [\lVert E_l^{(t-1)}\rVert_F+\epsilon]\) | Every 500 optimiser steps | Low |

These RVQ quantities answer complementary questions:

- **utilization:** how much of the available vocabulary is ever used;
- **perplexity:** how balanced that usage is;
- **quantisation error:** how accurately each level represents its input residual;
- **residual-energy reduction:** how much additional representational value each level contributes.

---

# 3. Stage 2 — BrainOmni

BrainOmni operates on BrainTokenizer's discrete representations and predicts masked RVQ token indices.

For masked position \(i\) and RVQ level \(l\):

- \(q_{il}\) is the ground-truth code index;
- \(p_{il}(k)\) is the predicted probability of code \(k\);
- \(M\) is the number of masked positions.

Stage-2 learning is monitored primarily through masked-token prediction.

---

| Monitor | Purpose | Mathematical definition | Cadence | Incremental cost |
|---|---|---|---|---|
| **Total masked-token CE** | Main convergence monitor for BrainOmni pretraining | \(L_{\mathrm{model}}=M^{-1}\sum_{i=1}^{M}\sum_{l=1}^{4}-\log p_{il}(q_{il})\) | Training: every 100 optimiser steps; validation: once per epoch | Negligible; already required for optimisation |
| **Per-RVQ CE** | Determine whether prediction learning differs substantially across the four RVQ levels | \(CE_l=-M^{-1}\sum_{i=1}^{M}\log p_{il}(q_{il})\) | Training: every 100 optimiser steps; validation: once per epoch | Negligible |
| **CE improvement over unigram predictor** | Separate contextual prediction ability from gains obtainable purely from an imbalanced token distribution | Let \(u_l(k)\) be the empirical marginal probability of code \(k\). Define \(CE_l^{\mathrm{uni}}=-M^{-1}\sum_i\log u_l(q_{il})\), then \(\Delta CE_l=CE_l^{\mathrm{uni}}-CE_l^{\mathrm{model}}\) | Validation: once per epoch | Negligible after the marginal token distribution has been estimated |
| **Per-RVQ top-1 accuracy** | Provide an interpretable measure of token prediction and reproduce the type of analysis shown in BrainOmni's codebook-layer learning curves | \(A_l=M^{-1}\sum_{i=1}^{M}\mathbf 1[\operatorname{argmax}_k p_{il}(k)=q_{il}]\) | Training: every 100 optimiser steps; validation: once per epoch | Low |
| **Accuracy improvement over majority-token predictor** | Give the top-1 analogue of the unigram comparison | \(A_l^{\mathrm{maj}}=\max_k u_l(k)\); report \(A_l-A_l^{\mathrm{maj}}\) | Validation: once per epoch | Negligible |
| **Performance by corruption type** | Determine whether contextual prediction differs between positions represented by the dedicated mask token and positions replaced by random tokens | Compute per-RVQ \(CE_l^{\mathrm{mask}}\), \(CE_l^{\mathrm{random}}\), and optionally corresponding accuracies on the two subsets | Validation: once per epoch | Negligible; uses predictions from the existing validation forward pass |
| **EEG-vs-MEG masked-token performance** | Detect whether joint EMEG pretraining learns substantially differently across modalities | Compute per-RVQ CE, and optionally accuracy, independently over EEG and MEG validation samples | Validation: once per epoch | Negligible–low |
| **Performance by latent-source index** | Detect specific latent-source token streams that remain systematically easier or harder to model | For source index \(s\), compute \(CE_{l,s}\) using only masked positions belonging to source \(s\) | Validation: once per epoch | Low |

Again, for EEG-only or MEG-only pretraining, EEG-vs-MEG masked-token performance doesn't need to be reported.

Stage-2 predictions are indexed by latent source rather than original sensor
channel. The modality stratification therefore re-tokenizes EEG and MEG sensor
subsets separately; a mixed EMEG sample contributes to both strata. The
latent-source stratification uses the original joint forward pass.

The unigram and majority-token comparisons are particularly important for interpreting differences between RVQ levels: higher raw accuracy for one codebook does not necessarily imply that its tokens contain more predictable contextual structure if its marginal token distribution is also substantially more imbalanced.

# GENCO-PTDF: Auditing and Improving Local AC Power-Flow Sensitivities

**Entrant:** PowerUp — individual entry, Independent.  
**Contact:** bin@smu.edu.  
**GitHub:** [BinHuangScut](https://github.com/BinHuangScut). **Hugging Face:** [banw](https://huggingface.co/banw).

## Idea and contribution

Can a useful power-flow predictor also supply accurate local derivatives? We audit
official GENCO Small models for AC power-transfer distribution factors (AC-PTDF),
then test whether local transfer-response supervision improves them beyond ordinary
PF fine-tuning. The deliverable is a common derivative API and reproducible benchmark
with classical references, numerical checks, models, data and negative results.

The result is directional: response training improves three of four tested
grid/load-distribution comparisons on selected transfer queries. It does **not**
consistently improve complete injection matrices and provides no general speedup.

## GENCO, OpenGridFM and method

We use official **GENCO Small/seed0 IEEE14 and IEEE118 checkpoints**, including their
per-grid normalizers, with pinned OpenGridFM **GraphKit** and **DataKit** revisions.
GraphKit's upstream implementation is unchanged; a separate PF adapter preserves
given voltage setpoints and marks reference voltage magnitude as known for every
neural method. Solved voltage labels are never model inputs. DataKit provides
corrected networks and admittance utilities; our load sampler and proportional
generator schedule are a fixed-dispatch extension rather than the official OPF data.

For transfer a→b, AC-PTDF is the derivative of sending-end branch MW with respect
to transfer MW: injection increases at a and decreases at b. PQ/PV types, Q loads,
PV/reference voltage setpoints, reference angle, taps and phase shifts stay fixed;
reference generation balances incremental losses. A pandapower Newton solve is
followed by a rebuilt converged-point Jacobian, sharing one sparse LU across queries.

We compare **DC, classical AC, GENCO initialization, PF fine-tuning and PF+response
fine-tuning**. Both fine-tuning arms see identical base/perturbed state labels and
retain the official state and physics losses. The response arm additionally fits
branch-flow central secants at ±0.1 and ±1 MW, with weight 1. Final derivatives are
computed by automatic differentiation; full matrices use block reverse AD.

Each grid has **2,000/400/400 train/validation/test base points plus 200 higher-load
test points**. Each point and its eight perturbation states stay in one split.
Regular global load multipliers are 0.8–1.2, higher load 1.2–1.4, local multipliers
0.9–1.1. Three training seeds use AdamW, lr=1e-4, batch 8 groups, max 100 epochs,
patience 10. Checkpoints are selected by validation AD-PTDF error. Maximum budgets
are matched; independent early stopping changes actual update counts.

## Results

Transaction-query relative Frobenius errors, averaged per point and then across
three fine-tuning seeds (mean ± seed SD):

| Grid / test set | DC | GENCO initialization | PF fine-tune | PF + response |
| --- | ---: | ---: | ---: | ---: |
| IEEE14 regular | 6.931% | 14.371% | 0.755 ± 0.060% | **0.448 ± 0.144%** |
| IEEE14 higher load | 9.223% | 28.690% | **1.246 ± 0.169%** | 1.430 ± 0.400% |
| IEEE118 regular | 5.545% | 10.003% | 7.653 ± 0.383% | **2.418 ± 0.210%** |
| IEEE118 higher load | 7.416% | 13.249% | 10.565 ± 0.243% | **5.562 ± 0.982%** |

Response supervision reduces these errors relative to PF by **40.6%, −14.7%,
68.4% and 47.4%**, respectively. The negative IEEE14 higher-load result is retained.

![Transfer-query error distributions](https://raw.githubusercontent.com/BinHuangScut/genco-ptdf/v0.1.0/results/figures/01_error_distributions.png)

![Flow prediction versus derivative error](https://raw.githubusercontent.com/BinHuangScut/genco-ptdf/v0.1.0/results/figures/02_flow_vs_derivative.png)

![Three-seed fine-tuning comparison and common-update validation curves](https://raw.githubusercontent.com/BinHuangScut/genco-ptdf/v0.1.0/results/figures/03_finetuning_comparison.png)

**Coverage matters.** Queries cover 8 IEEE14 and 16 IEEE118 source-PV→load-PQ
directions; two directions per point supervise secants. Complete non-reference
matrices additionally evaluate 16 regular and 16 higher-load points per grid:

| Grid / subset | PF fine-tune | PF + response |
| --- | ---: | ---: |
| IEEE14 regular | 2.708 ± 0.067% | 2.876 ± 0.558% |
| IEEE14 higher load | 1.825 ± 0.154% | 2.239 ± 0.104% |
| IEEE118 regular | 25.902 ± 0.684% | 24.753 ± 1.052% |
| IEEE118 higher load | 27.609 ± 0.209% | 27.832 ± 0.721% |

Thus improved selected-transfer accuracy does not establish complete-basis accuracy.
We also report coefficient MAE, sign agreement, unseen +5 MW flow-increment error,
state/flow errors and descriptive correlations. Three seeds support no statistical
significance claim. Fixed-control limit violations are retained and flagged, so
these are conditional mathematical sensitivities, not OPF-certified dispatch.

## Numerical evidence and runtime

Classical Jacobians agree with high-precision central differences (relative errors
7.1e-11/5.3e-10). Neural AD agrees with model finite differences under step sweeps;
tests verify units, directions, multiple generators, known-input masks, target
isolation and block/batched AD. All 6,000 base states/full labels and 48,000 paired
state residuals are certified against the final pandapower solver; 11 tests pass.

On RTX 4080 SUPER/WSL, IEEE118 response-model queries take about **301 ms per single
point**, versus **10.9 ms for classical PF+Jacobian** and **3.6 ms for derivative-only
AC**. Classical static matrices and Jacobian factors are reused. Batch sizes
1/16/64, 20 synchronized repeats, full-matrix timings and GPU memory are reported.
GPU models use float32 and classical solvers float64; preprocessing/result transfers
are included, model loading excluded. Single-point neural queries are slower.

![Accuracy, single-point latency and batch throughput](https://raw.githubusercontent.com/BinHuangScut/genco-ptdf/v0.1.0/results/figures/04_accuracy_and_latency.png)

## Open science and reproduction

- **[Code, API, configuration and results](https://github.com/BinHuangScut/genco-ptdf/tree/v0.1.0)**
- **[Models, all grouped data, editable figures and SHA256 checksums](https://github.com/BinHuangScut/genco-ptdf/releases/tag/v0.1.0)**
- **[Detailed research report](https://github.com/BinHuangScut/genco-ptdf/blob/v0.1.0/docs/submission.md)**
- **[Model/data card and limitations](https://github.com/BinHuangScut/genco-ptdf/blob/v0.1.0/docs/model_card.md)**

The release includes 12 best/last fine-tuned models, official initialization and
normalizers, data/limit manifests, per-point metrics, training curves, a pinned
environment and numerical-validation logs. Project code is Apache-2.0; upstream
artifacts retain their attribution and applicable terms. Source revisions are pinned.

```bash
bash scripts/bootstrap.sh
VENV=${GENCO_VENV:-$HOME/.venvs/genco-ptdf}
"$VENV/bin/python" scripts/download_artifacts.py
"$VENV/bin/python" scripts/verify_artifacts.py
# Fresh checkout without downloaded artifacts: regenerate the full experiment
GENCO_VENV="$VENV" bash scripts/run_experiments.sh
```

GPU atomic reductions do not guarantee identical bits on reruns. We preserve
actual checkpoints, data hashes and all observed metrics for audit. This study
covers two fixed topologies and controls; it does not test topology generalization,
line outages or limit switching.

# GENCO-PTDF

Reproducible fixed-control AC-PTDF experiments on IEEE 14 and IEEE 118 with official
GENCO Small/seed0 weights, pinned OpenGridFM GraphKit and DataKit, and response-aware
fine-tuning. The experiment asks whether power-flow prediction accuracy transfers
to derivative accuracy, and whether local response supervision improves it.

**Entrant:** PowerUp (individual), Independent. **Contact:** bin@smu.edu.
[Community entry text](docs/challenge_entry.md) ·
[Detailed report](docs/submission.md) ·
[Models and data: v0.1.0](https://github.com/BinHuangScut/genco-ptdf/releases/tag/v0.1.0).

The completed run includes 6,000 base operating points and 12 fine-tunes. On the
evaluated transfer queries, response supervision reduces relative PTDF error by
40.6% (IEEE14 regular), 68.4% (IEEE118 regular) and 47.4% (IEEE118 higher load),
but increases it by 14.7% on IEEE14 higher load. Complete non-reference injection
matrices do not show consistent improvement. Single-point neural derivatives are
slower than the classical sparse baseline. Read the [English report](docs/submission.md)
for all results, fixed-control limit flags and the scope of these findings.

## Run in WSL Ubuntu

Python 3.12, CUDA-capable PyTorch, and an NVIDIA GPU are used for the full run.
The reference environment is RTX 4080 SUPER 16 GB. Run from the project folder:

```bash
bash scripts/bootstrap.sh
VENV=${GENCO_VENV:-$HOME/.venvs/genco-ptdf}
# Download the exact delivered models/data/figure exports, verifying SHA256:
"$VENV/bin/python" scripts/download_artifacts.py
# Check the delivered snapshot:
"$VENV/bin/python" scripts/verify_artifacts.py
# Or regenerate experiments in a fresh checkout, without downloading artifacts:
"$VENV/bin/python" -m genco_ptdf.cli fetch
"$VENV/bin/python" -m genco_ptdf.cli validate
"$VENV/bin/python" -m genco_ptdf.cli generate
"$VENV/bin/python" -m pytest -q
"$VENV/bin/python" -m genco_ptdf.cli train
"$VENV/bin/python" -m genco_ptdf.cli evaluate
"$VENV/bin/python" -m genco_ptdf.cli report
```

`python -m genco_ptdf.cli all` runs the pipeline. Completed training runs and
generated datasets are reused. To reproduce from scratch, use a fresh checkout
and the same pinned source/data configuration. `--grid`, `--mode`, `--seed`, and
`--config` select a subset. No Julia/OPF process is needed for this fixed-dispatch
DataKit extension. [Methods](docs/methods.md) explains the sampler and solvers.

## API

```python
import numpy as np
from genco_ptdf import Grid, compute_ac_ptdf

grid = Grid("case14_ieee")
point = grid.default_point()
D = np.zeros((grid.nb, 1))
index = {int(bus): row for row, bus in enumerate(grid.ids)}
D[index[2], 0], D[index[9], 0] = 1, -1  # +t MW at bus 2, -t MW at bus 9
result = compute_ac_ptdf(grid, point, D, method="ac_jacobian")
print(result["factors"])  # branch × transaction, MW/MW
full = compute_ac_ptdf(grid, point)  # PV then PQ columns, each to reference
neural = compute_ac_ptdf(grid, point, D, method="genco",
    checkpoint="checkpoints/finetuned/case14_ieee/response/seed0/best.pt")
```

Injection matrices follow `grid.ids` row order, not numeric bus-ID indexing.
`branch_ids` are zero-based original branch rows, with original from→to orientation.
All setpoints, reference convention and voltage/angle units are in the boundary
record. `method="dc"` provides the DC baseline. The derivative is conditional on
fixed PQ/PV types and fixed taps; reference generation absorbs incremental losses.

## Artifacts

- `requirements.lock.txt`, `configs/experiment.yaml`, `results/environment.json`:
  environment, experiment configuration and actual hardware.
- `data/<grid>/`: grouped NPZ data, hashes, split manifest, nonconvergence records.
- `checkpoints/official/`: original Small/seed0 model/config/normalizer and source revision.
- `checkpoints/finetuned/<grid>/<mode>/seed*/`: best/last weights, complete metadata,
  per-epoch history. Checkpoint selection uses held-out AD PTDF error.
- `results/<grid>/`: per-point/seed summaries, complete-basis subset, control-limit
  counts, timings. `results/figures/`: four figures, editable exports and source tables.
- `docs/submission.md`: generated English challenge write-up.
- `results/artifact_manifest.json`: file hashes for delivered data/models/results.

The two fine-tuning methods see identical base and perturbation state labels and
the same sampled transactions. Early stopping can yield different actual update
counts. GENCO initialization always uses the common PF boundary adapter. The
official per-grid models were trained on those grids; this experiment does not
test generalization to unseen topologies. Results, including negative findings
and speed disadvantages, are reported without assuming a benefit.

## Upstream provenance

- [GENCO challenge](https://huggingface.co/spaces/gridfm/genco-challenge)
- [Official PF reproduction guide](https://github.com/gridfm/gridfm-paper-genco/blob/main/repro_instructions/5.4.2_PF_on_datakit.md)
- [Official checkpoint repository](https://huggingface.co/gridfm/genco-pf-datakit)
- GraphKit `95b90eecc7f5c2c86a9dd776f0c3f40434def22f`
- DataKit `316c4afeb1b1d048b6ccaa6135fbc54e00549f64`
- [pandapower AC power flow](https://pandapower.readthedocs.io/en/stable/powerflow/ac.html)

Upstream code is retained unchanged in ignored `vendor/` clones. Project code is
Apache-2.0; upstream code/models/data retain their respective license terms.
Public artifact links, contents and extraction instructions are in [release notes](docs/release.md).

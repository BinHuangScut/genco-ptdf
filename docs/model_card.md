# GENCO-PTDF model and data card

## Intended use

Research on fixed-control local AC power-flow sensitivities. Two separate models
cover the fixed IEEE14 and IEEE118 corrected PGLib topologies. Models estimate
bus voltage state and derive sending-end branch MW through the official physics
decoder. AC-PTDF is computed by automatic differentiation in physical MW coordinates.

## Sources and training

The official GENCO Small/seed0 checkpoints and normalizers are pinned by
`configs/provenance.json`; upstream code remains in unchanged pinned clones.
Each model has 12 message-passing layers, hidden size24 and8 attention heads.
PF and PF+response arms fine-tune all parameters. Three seeds0,1,42 are used per arm.
See `complete.json` and `history.csv` for actual updates, selected epoch and
validation score. `best.pt` is the delivered checkpoint; `last.pt` supports auditing.

The project sampler provides 2000/400/400 train/validation/test grouped operating
points and200 higher-load points for each grid. All perturbations remain with their
base point. Loads, generator plans, exact state, branch flows, complete reference
derivatives and paired transfer states are in `dataset.npz`. Dataset and original
case hashes, split counts, controls and generator mappings are in `manifest.json`.

## Boundary adapter

Given PV/reference voltage magnitudes and reference angle are known. Before unknown
fields are zeroed, setpoints are retained; reference Vm is additionally unmasked.
This is common to official initialization and both fine-tunes. Solved voltage labels
cannot affect input features. The result therefore describes official initialization
with a PF adapter, rather than unchanged native upstream input masking.

## Limits and evaluation

Use `results/results_table.csv` and grid-specific per-point CSVs for actual metrics.
Data nonconvergence and control-limit violations are logged without switching bus
types. These models do not implement controller-limit switching, OPF redispatch,
line contingencies, topology generalization, or operational dispatch decisions.
Positive transfer directions and reference loss allocation are fixed conventions.
Serving dtype is float32; small-step derivative checks cast fixed weights to float64.
GPU atomics permit small non-bitwise run differences. Numerical validation,
independent solver tests, matched-data baselines and seed variability are delivered.

## License and release

Project code is Apache-2.0. Upstream code/model/data license terms remain applicable.
The public [v0.1.0 release](https://github.com/BinHuangScut/genco-ptdf/releases/tag/v0.1.0)
provides all delivered checkpoints, grouped datasets and figure exports.
See `docs/release.md` for artifact downloads and checksum verification.
Entrant and contact information are in `configs/submission.json`.

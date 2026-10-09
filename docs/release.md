# GENCO-PTDF v0.1.0 research snapshot

Entrant: PowerUp (individual), Independent. Contact: bin@smu.edu.

Code and tabular results: <https://github.com/BinHuangScut/genco-ptdf/tree/v0.1.0>.
Release assets: <https://github.com/BinHuangScut/genco-ptdf/releases/tag/v0.1.0>.

| Archive | Contents |
| --- | --- |
| `data-case14_ieee.tar.gz` | All 3,000 IEEE14 groups, paired states, full reference matrices, splits and limit records |
| `data-case118_ieee.tar.gz` | All 3,000 IEEE118 groups with the same data schema |
| `models.tar.gz` | Official Small/seed0 weights, normalization/configuration files, and best/last weights plus histories for all 12 fine-tunes |
| `figures.tar.gz` | Four PNG/SVG/PDF/TIFF figure bundles, source tables and export checks |
| `SHA256SUMS` | Archive checksums, also recorded in `release/manifest.json` |

Archive paths are relative to the repository root. With the pinned environment
installed, run `python scripts/download_artifacts.py` to download, verify and
extract all archives. It skips matching local archives. Then run
`python scripts/verify_artifacts.py` to verify the complete scientific snapshot.
To unpack manually, check `SHA256SUMS` and use `tar -xzf <archive> -C <repo-root>`.

The code/configuration/result snapshot is pinned by the release tag. The delivered
datasets and checkpoints retain their recorded hashes. Numerical reruns have fixed
seeds and source revisions; GPU atomic reductions do not guarantee identical bits.
`results/artifact_manifest.json` hashes scientific artifacts and source files;
the separate release manifest hashes distribution archives. Release-workflow
receipts and local browser screenshots are excluded from the scientific snapshot.

The study is conditional on fixed controls. Limit violations are retained and
reported; the sampler does not produce OPF-certified dispatch. Gains on the
selected transfer queries do not establish gains on all injection directions.
Three seeds and independent early stopping do not establish statistical
significance or exact actual-update matching. Single-point neural derivative
queries are slower than the cached classical method on the measured hardware.

Project code is Apache-2.0. Official models, OpenGridFM dependencies and corrected
PGLib/MATPOWER data retain their upstream attribution and applicable license terms;
see `LICENSE`, `NOTICE`, `configs/provenance.json` and the detailed model/data card.

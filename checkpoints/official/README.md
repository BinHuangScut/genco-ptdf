---
library_name: gridfm-graphkit
pipeline_tag: tabular-regression
license: apache-2.0
tags:
  - power-systems
  - power-flow
  - genco
  - datakit
---

GENCO power-flow checkpoints on datakit grids ([paper](https://arxiv.org/abs/2608.09921)). Base, Small, and Tiny (hidden size 48, 24, and 12) on IEEE 14, 30, 57, and 118 and GOC 500. Small and Tiny on GOC 2000 and GOC 10000.

Evaluate on the paper branch by following [GENCO §5.4.2](https://github.com/gridfm/gridfm-paper-genco/blob/main/repro_instructions/5.4.2_PF_on_datakit.md). That guide uses graphkit [genco-paper-repro](https://github.com/gridfm/gridfm-graphkit/tree/genco-paper-repro) and is how the paper numbers were produced.

Or evaluate the same checkpoints on graphkit [main](https://github.com/gridfm/gridfm-graphkit/tree/main). `main` is under development, so the same numbers are not guaranteed. They matched the paper-branch result at commit [`f104900`](https://github.com/gridfm/gridfm-graphkit/commit/f104900245cba26d1380037eea562cf633a97484) (1 October 2026).

```bash
git clone https://github.com/gridfm/gridfm-graphkit.git
cd gridfm-graphkit
pip install -e .
pip install "huggingface_hub[cli]"
TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.__version__ + ('+cpu' if torch.version.cuda is None else ''))")
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
export MLFLOW_ALLOW_FILE_STORE=true

mkdir -p data/case118_ieee/raw
hf download gridfm/pf_small_case118_ieee --repo-type dataset --local-dir data/case118_ieee/raw
hf download gridfm/genco-pf-datakit --include "case118_ieee/base/seed0/**" --local-dir genco-pf-datakit

gridfm_graphkit evaluate \
  --config_conversion auto_inline \
  --config genco-pf-datakit/case118_ieee/base/seed0/case118_ieee_base_seed0.yaml \
  --data_path data \
  --model_path genco-pf-datakit/case118_ieee/base/seed0/best_model_state_dict.pt \
  --normalizer_stats genco-pf-datakit/case118_ieee/base/seed0/normalizer_stats.pt
```

For GOC 10000, pass `last.pt` instead of `best_model_state_dict.pt`.


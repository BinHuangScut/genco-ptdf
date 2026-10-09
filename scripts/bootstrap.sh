#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${GENCO_VENV:-$HOME/.venvs/genco-ptdf}
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
mkdir -p vendor
if [ ! -d vendor/gridfm-graphkit/.git ]; then
  git clone https://github.com/gridfm/gridfm-graphkit.git vendor/gridfm-graphkit
fi
if [ ! -d vendor/gridfm-datakit/.git ]; then
  git clone https://github.com/gridfm/gridfm-datakit.git vendor/gridfm-datakit
fi
git -C vendor/gridfm-graphkit checkout 95b90eecc7f5c2c86a9dd776f0c3f40434def22f
git -C vendor/gridfm-datakit checkout 316c4afeb1b1d048b6ccaa6135fbc54e00549f64
"$VENV/bin/pip" install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/pip" install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html --only-binary=:all:
"$VENV/bin/pip" install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128 -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
"$VENV/bin/pip" install -e vendor/gridfm-datakit -e vendor/gridfm-graphkit -e . --no-deps
"$VENV/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.get_device_name())'

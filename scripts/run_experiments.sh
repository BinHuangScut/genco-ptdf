#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${GENCO_VENV:-$HOME/.venvs/genco-ptdf}
mkdir -p results/logs
"$VENV/bin/python" scripts/record_environment.py
"$VENV/bin/python" -m genco_ptdf.cli all 2>&1 | tee results/logs/pipeline.log
"$VENV/bin/python" -m pytest -q > results/validation/pytest.txt
"$VENV/bin/python" -m genco_ptdf.cli report
"$VENV/bin/python" scripts/verify_artifacts.py | tee results/verification.txt

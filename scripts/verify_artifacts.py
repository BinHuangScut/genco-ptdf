"""Verify that the finished release is complete and internally consistent."""
from hashlib import sha256
import json
import pandas as pd
import numpy as np
import yaml
from genco_ptdf.classical import ROOT
from genco_ptdf.data import Dataset

config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
for name in config["grids"]:
    dataset = Dataset(name)
    assert dataset.manifest["counts"] == config["counts"]
    assert dataset.manifest["dataset_sha256"] == sha256((ROOT / "data" / name / "dataset.npz").read_bytes()).hexdigest()
    perturbations = pd.read_csv(ROOT / "data" / name / "perturbation_limits.csv")
    assert len(perturbations) == sum(config["counts"].values()) * 8
    for mode in ("pf", "response"):
        for seed in config["training"]["seeds"]:
            folder = ROOT / "checkpoints/finetuned" / name / mode / f"seed{seed}"
            record = json.loads((folder / "complete.json").read_text())
            assert record["training_data_sha256"] == dataset.manifest["dataset_sha256"]
            assert 1 <= record["epochs"] <= config["training"]["epochs"]
            assert 0 <= record["best_epoch"] <= record["epochs"]
            assert (folder / "best.pt").stat().st_size > 1000
    points = pd.read_csv(ROOT / "results" / name / "per_point.csv")
    assert len(points) == (config["counts"]["test"] + config["counts"]["stress"]) * 9
    assert not points.duplicated(["method", "seed", "group"]).any()
    assert np.isfinite(points[["ptdf_mae", "ptdf_relative", "sign_agreement"]]).all().all()
    full = pd.read_csv(ROOT / "results" / name / "full_matrix_per_point.csv")
    assert len(full) == config["evaluation"]["full_matrix_examples"] * 2 * 9
    timing = pd.read_csv(ROOT / "results" / name / "timing.csv")
    assert len(timing) == 6 * (len(config["evaluation"]["batch_sizes"]) + 1)
    assert (timing.median_seconds > 0).all()
for record in json.loads((ROOT / "results/artifact_manifest.json").read_text()):
    path = ROOT / record["path"]
    assert path.stat().st_size == record["bytes"], path
    assert sha256(path.read_bytes()).hexdigest() == record["sha256"], path
checks = json.loads((ROOT / "results/validation/checkpoints.json").read_text())
assert len(checks) == 14
classical = json.loads((ROOT / "results/validation/pandapower_dataset.json").read_text())
assert len(classical) == 2
for record in classical:
    assert record["base_points"] == 3000 and record["paired_states"] == 24000
    assert record["dataset_sha256"] == Dataset(record["grid"]).manifest["dataset_sha256"]
    assert record["solver_source_sha256"] == sha256((ROOT / "genco_ptdf/classical.py").read_bytes()).hexdigest()
    assert record["base_voltage_max_pu"] < 1e-8
    assert record["base_ptdf_max_abs"] < 1e-6
    assert record["paired_residual_max_pu"] < 1e-9
for figure in range(1, 5):
    exports = list((ROOT / "results/figures").glob(f"0{figure}_*.svg"))
    assert len(exports) == 1
    for extension in ("png", "pdf", "tiff"):
        assert exports[0].with_suffix("." + extension).stat().st_size > 1000
assert (ROOT / "docs/submission.md").exists()
print("Verified: 6,000 grouped points, 12 fine-tunes, 10,800 test records, 576 complete matrices, 48 timing rows, 4 figure bundles.")

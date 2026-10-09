"""Held-out accuracy and explicit, cached-solver performance measurements."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .classical import Grid, ROOT
from .data import Dataset
from .neural import GencoSolver


def metrics(predicted, truth):
    difference = predicted - truth
    significant = np.abs(truth) >= 1e-3
    return {"ptdf_mae": float(np.mean(np.abs(difference))),
            "ptdf_rmse": float(np.sqrt(np.mean(difference ** 2))),
            "ptdf_relative": float(np.linalg.norm(difference) / max(np.linalg.norm(truth), 1e-12)),
            "sign_agreement": float(np.mean(np.sign(predicted[significant]) == np.sign(truth[significant]))),
            "max_absolute_error": float(np.max(np.abs(difference)))}


def model_specs(name, config):
    yield "official_init", 0, None
    for mode in ("pf", "response"):
        for seed in config["training"]["seeds"]:
            folder = ROOT / "checkpoints/finetuned" / name / mode / f"seed{seed}"
            if not (folder / "complete.json").exists():
                raise FileNotFoundError(f"Training unfinished: {folder}")
            yield mode, seed, folder / "best.pt"


def evaluate(name, config, run_timing=True):
    torch.set_num_threads(config["training"]["torch_threads"])
    grid, dataset = Grid(name), Dataset(name)
    folder = ROOT / "results" / name
    folder.mkdir(parents=True, exist_ok=True)
    a = dataset.arrays
    D = a["D"]
    ids = np.r_[dataset.indices("test"), dataset.indices("stress")]
    true_h = a["H"][ids] @ a["directions"]
    # An unseen +5 MW transfer tests the first-order approximation away from
    # the +/-0.1 and +/-1 MW training secants. Keep all controls fixed.
    delta_flow, delta_failures, delta_limits = {}, [], []
    for i in ids:
        try:
            changed, _ = grid.solve(grid.perturb(dataset.point(i), D[:, 0], 5.0))
            delta_flow[i] = grid.flows(changed)[0].real - a["flow"][i]
            delta_limits.append({"group": int(i), "split": dataset.groups.iloc[i]["split"],
                                 "query": 0, "step_mw": 5.0,
                                 **grid.limits(grid.perturb(dataset.point(i), D[:, 0], 5.0), changed)})
        except RuntimeError as error:
            delta_failures.append({"group": int(i), "error": str(error)})
    (folder / "delta_failures.json").write_text(json.dumps(delta_failures, indent=2))
    pd.DataFrame(delta_limits).to_csv(folder / "delta_control_limits.csv", index=False)
    rows, full_rows = [], []
    full_ids = np.r_[dataset.indices("test")[:config["evaluation"]["full_matrix_examples"]],
                     dataset.indices("stress")[:config["evaluation"]["full_matrix_examples"]]]
    for method in ("dc", "ac_jacobian"):
        for position, i in enumerate(ids):
            truth = true_h[position]
            prediction = grid.dc_factors(D) if method == "dc" else grid.ac_factors(a["V"][i], D)
            record = {"grid": name, "method": method, "seed": -1, "group": int(i),
                      "split": dataset.groups.iloc[i]["split"], **metrics(prediction, truth)}
            record["delta_mae_mw"] = float(np.mean(np.abs(5 * prediction[:, 0] - delta_flow[i]))) if i in delta_flow else np.nan
            rows.append(record)
        for i in full_ids:
            prediction = grid.dc_factors() if method == "dc" else grid.ac_factors(a["V"][i])
            full_rows.append({"grid": name, "method": method, "seed": -1, "group": int(i),
                              "split": dataset.groups.iloc[i]["split"], **metrics(prediction, a["H"][i])})
    for method, seed, checkpoint in model_specs(name, config):
        path = folder / f"{method}_seed{seed}.csv"
        full_path = folder / f"{method}_seed{seed}_full.csv"
        if path.exists() and full_path.exists():
            rows.extend(pd.read_csv(path).to_dict("records"))
            full_rows.extend(pd.read_csv(full_path).to_dict("records"))
            continue
        solver = GencoSolver(grid, checkpoint)
        method_rows = []
        for start in range(0, len(ids), 16):
            selected = ids[start:start + 16]
            points = [dataset.point(i) for i in selected]
            predicted = solver.directional_batch(points, D, capture=config["evaluation"].get("cuda_graph", False))
            with torch.no_grad():
                batch = solver.batch(a["pd"][selected], a["qd"][selected], a["pg"][selected])
                state, flows = solver.forward(batch)
                state = state["bus"][:, :2].reshape(len(selected), grid.nb, 2).cpu().numpy()
                flows = flows.cpu().numpy()
            for j, i in enumerate(selected):
                truth = true_h[start + j]
                error = metrics(predicted[j], truth)
                record = {"grid": name, "method": method, "seed": seed, "group": int(i),
                          "split": dataset.groups.iloc[i]["split"], **error,
                          "flow_mae_mw": float(np.mean(np.abs(flows[j] - a["flow"][i]))),
                          "vm_mae_pu": float(np.mean(np.abs(state[j, :, 0] - abs(a["V"][i])))),
                          "va_mae_rad": float(np.mean(np.abs(state[j, :, 1] - np.angle(a["V"][i])))),
                          "delta_mae_mw": float(np.mean(np.abs(5 * predicted[j, :, 0] - delta_flow[i]))) if i in delta_flow else np.nan}
                method_rows.append(record)
            if (start + 16) % 160 == 0:
                print(f"Evaluate {name} {method} seed{seed}: {min(start+16,len(ids))}/{len(ids)}", flush=True)
        pd.DataFrame(method_rows).to_csv(path, index=False)
        rows.extend(method_rows)
        method_full = []
        for i in full_ids:
            prediction = solver.factors(dataset.point(i), chunk=config["evaluation"]["derivative_chunk"])
            method_full.append({"grid": name, "method": method, "seed": seed, "group": int(i),
                                "split": dataset.groups.iloc[i]["split"], **metrics(prediction, a["H"][i])})
        pd.DataFrame(method_full).to_csv(full_path, index=False)
        full_rows.extend(method_full)
        del solver
        torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(folder / "per_point.csv", index=False)
    pd.DataFrame(full_rows).to_csv(folder / "full_matrix_per_point.csv", index=False)
    numeric = ["ptdf_mae", "ptdf_rmse", "ptdf_relative", "sign_agreement", "delta_mae_mw", "flow_mae_mw", "vm_mae_pu", "va_mae_rad"]
    summary = table.groupby(["grid", "method", "seed", "split"])[numeric].mean().reset_index()
    summary.to_csv(folder / "summary.csv", index=False)
    counts = dataset.groups.groupby("split")[["voltage_limit", "q_limit", "p_limit", "branch_limit"]].agg(["sum", "mean"])
    counts.to_csv(folder / "control_limits.csv")
    if run_timing:
        benchmark(name, dataset, config)


def benchmark(name, dataset, config, methods=None):
    """Same transfer queries for all methods; preprocessing included, load excluded.

    Classical PF+Jacobian starts at case voltage, not a saved solved voltage.
    The derivative-only row uses an already known operating voltage. Static
    Y matrices, DC LU and neural graph templates/weights are prepared once.
    """
    torch.set_num_threads(config["training"]["torch_threads"])
    grid = Grid(name)
    D = dataset.arrays["D"]
    ids = dataset.indices("test")
    settings = config["evaluation"]
    records = []
    specs = [("dc", None), ("ac_jacobian", None), ("ac_pf_jacobian", None)]
    specs += [(m, p) for m, s, p in model_specs(name, config) if s == 0]
    if methods is not None:
        specs = [(m, p) for m, p in specs if m in methods]
    for method, checkpoint in specs:
        solver = GencoSolver(grid, checkpoint) if method in ("official_init", "pf", "response") else None
        for size in settings["batch_sizes"]:
            points = [dataset.point(i) for i in ids[:size]]
            def operation():
                if solver:
                    return solver.directional_batch(points, D, capture=settings.get("cuda_graph", False))
                if method == "dc":
                    return np.stack([grid.dc_factors(D) for p in points])
                if method == "ac_jacobian":
                    return np.stack([grid.ac_factors(p.V, D) for p in points])
                result = []
                for point in points:
                    fresh = type(point)(point.pd, point.qd, point.pg)
                    voltage, _ = grid.solve(fresh)
                    result.append(grid.ac_factors(voltage, D))
                return np.stack(result)
            operation()  # warm up allocations and kernels
            if solver:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(settings["timing_repeats"]):
                start = time.perf_counter()
                operation()
                if solver:
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            records.append({"grid": name, "method": method, "scope": "transaction_matrix", "batch_size": size,
                            "transactions": D.shape[1], "branches": grid.nl, "dtype": "float32" if solver else "float64",
                            "device": solver.device if solver else "cpu", "repeats": len(times),
                            "median_seconds": float(np.median(times)), "p95_seconds": float(np.percentile(times, 95)),
                            "mean_seconds": float(np.mean(times)), "std_seconds": float(np.std(times)),
                            "seconds_per_point": float(np.median(times) / size),
                            "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20 if solver else 0})
            records[-1]["peak_reserved_gpu_mib"] = torch.cuda.max_memory_reserved() / 2**20 if solver else 0
            records[-1]["implementation"] = "CUDA graph JVP" if solver and settings.get("cuda_graph", False) else "eager JVP" if solver else "pandapower NR + rebuilt Jacobian" if method == "ac_pf_jacobian" else "cached sparse matrices, fresh per-point AC factorization"
            print(f"Timing {name} {method} batch{size}: {np.median(times)/size:.5f}s/point", flush=True)
        # Full sensitivity matrices are additionally measured at one point,
        # using block reverse AD rather than many scalar finite differences.
        point = dataset.point(ids[0])
        def full_operation():
            if solver:
                return solver.factors(point, chunk=settings["derivative_chunk"])
            if method == "dc":
                return grid.dc_factors()
            if method == "ac_jacobian":
                return grid.ac_factors(point.V)
            fresh = type(point)(point.pd, point.qd, point.pg)
            voltage, _ = grid.solve(fresh)
            return grid.ac_factors(voltage)
        full_operation()
        if solver:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(settings["timing_repeats"]):
            start = time.perf_counter()
            full_operation()
            if solver:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        records.append({"grid": name, "method": method, "scope": "full_matrix", "batch_size": 1,
                        "transactions": grid.nb - 1, "branches": grid.nl,
                        "dtype": "float32" if solver else "float64", "device": solver.device if solver else "cpu",
                        "repeats": len(times), "median_seconds": float(np.median(times)),
                        "p95_seconds": float(np.percentile(times, 95)), "mean_seconds": float(np.mean(times)),
                        "std_seconds": float(np.std(times)), "seconds_per_point": float(np.median(times)),
                        "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20 if solver else 0})
        records[-1]["peak_reserved_gpu_mib"] = torch.cuda.max_memory_reserved() / 2**20 if solver else 0
        records[-1]["implementation"] = "block reverse AD" if solver else "pandapower NR + rebuilt Jacobian" if method == "ac_pf_jacobian" else "cached sparse matrices, fresh per-point AC factorization"
        if solver:
            del solver
            torch.cuda.empty_cache()
    path = ROOT / "results" / name / "timing.csv"
    table = pd.DataFrame(records)
    if methods is not None and path.exists():
        previous = pd.read_csv(path)
        table = pd.concat([previous[~previous.method.isin(methods)], table], ignore_index=True)
    table.to_csv(path, index=False)

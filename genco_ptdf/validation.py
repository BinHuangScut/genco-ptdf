import json
import time
import numpy as np
import torch
from hashlib import sha256
from .classical import Grid, ROOT
from .data import make_transactions
from .neural import GencoSolver


def validate_dataset(grids):
    """Certify every stored state/label against the final pandapower solver."""
    from .classical import OperatingPoint
    from .data import Dataset
    path = ROOT / "results/validation/pandapower_dataset.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(path.read_text()) if path.exists() else []
    solver_hash = sha256((ROOT / "genco_ptdf/classical.py").read_bytes()).hexdigest()
    records = [r for r in previous if r["grid"] not in grids]
    for name in grids:
        grid, data = Grid(name), Dataset(name)
        cached = [r for r in previous if r["grid"] == name
                  and r.get("solver_source_sha256") == solver_hash
                  and r.get("dataset_sha256") == data.manifest["dataset_sha256"]]
        if cached:
            records.extend(cached)
            continue
        maxima = {"base_voltage_max_pu": 0., "base_ptdf_max_abs": 0.,
                  "paired_residual_max_pu": 0.}
        for i in range(len(data.groups)):
            point = data.point(i)
            voltage, _ = grid.solve(OperatingPoint(point.pd, point.qd, point.pg))
            maxima["base_voltage_max_pu"] = max(maxima["base_voltage_max_pu"],
                                                float(np.max(abs(voltage - point.V))))
            maxima["base_ptdf_max_abs"] = max(maxima["base_ptdf_max_abs"],
                                              float(np.max(abs(grid.ac_factors(voltage) - data.arrays["H"][i]))))
            for pair, query in enumerate(data.arrays["pair_index"][i]):
                for step_index, step in enumerate(data.arrays["steps"]):
                    for sign_index, sign in enumerate((1, -1)):
                        changed = grid.perturb(point, data.arrays["D"][:, query], sign * step)
                        v = data.arrays["pair_V"][i, pair, step_index, sign_index]
                        pg = np.bincount(grid.gb, weights=changed.pg, minlength=grid.nb)
                        qg = np.bincount(grid.gb, weights=grid.gen[:, 2], minlength=grid.nb)
                        mismatch = v*np.conj(grid.Ybus @ v) - (pg-changed.pd+1j*(qg-changed.qd))/grid.base
                        residual = np.r_[mismatch[grid.nonref].real, mismatch[grid.pq].imag]
                        maxima["paired_residual_max_pu"] = max(maxima["paired_residual_max_pu"],
                                                              float(np.max(abs(residual))))
            if (i + 1) % 500 == 0:
                print(f"{name}: verified {i+1}/{len(data.groups)} base groups", flush=True)
        assert maxima["base_voltage_max_pu"] < 1e-8, maxima
        assert maxima["base_ptdf_max_abs"] < 1e-6, maxima  # stored H is float32
        assert maxima["paired_residual_max_pu"] < 1e-9, maxima
        record = {"grid": name, "base_points": len(data.groups), "paired_states": len(data.groups)*8,
                  "solver": "pandapower.pypower.newtonpf", "solver_source_sha256": solver_hash,
                  "dataset_sha256": data.manifest["dataset_sha256"], **maxima}
        records.append(record)
        print(json.dumps(record), flush=True)
    path.write_text(json.dumps(records, indent=2))


def validate(grids):
    torch.set_num_threads(2)
    folder = ROOT / "results/validation"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for name in grids:
        grid = Grid(name)
        point = grid.default_point()
        point.V, info = grid.solve(point)
        D = make_transactions(grid, 3)
        analytic = grid.ac_factors(point.V, D)
        finite = []
        for direction in D.T:
            vp, _ = grid.solve(grid.perturb(point, direction, 0.01))
            vm, _ = grid.solve(grid.perturb(point, direction, -0.01))
            finite.append((grid.flows(vp)[0].real - grid.flows(vm)[0].real) / 0.02)
        classical_error = np.linalg.norm(np.column_stack(finite) - analytic) / np.linalg.norm(analytic)
        solver = GencoSolver(grid, dtype=torch.float64)
        start = time.perf_counter()
        ad = solver.factors(point, D, chunk=8)
        if solver.device == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - start
        fd_errors = {}
        # A finite step can cross a LeakyReLU boundary. Check convergence to
        # the local derivative as the step shrinks, in double precision.
        for step in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            fd = []
            for direction in D.T:
                plus = solver.predict(grid.perturb(point, direction, step))[1]
                minus = solver.predict(grid.perturb(point, direction, -step))[1]
                fd.append((plus - minus) / (2 * step))
            fd = np.column_stack(fd)
            fd_errors[str(step)] = float(np.linalg.norm(ad - fd) / max(np.linalg.norm(ad), 1e-12))
        neural_error = fd_errors['1e-06']
        jvp = solver.directional_factors(point, D)
        jvp_error = np.linalg.norm(ad - jvp) / max(np.linalg.norm(ad), 1e-12)
        # Physics decoder and normalized edge representation must give exactly
        # the same branch flows as the classical admittance formulation.
        batch = solver.batch(point.pd, point.qd, point.pg)
        truth = torch.tensor(np.column_stack([abs(point.V), np.angle(point.V)]), dtype=torch.float64, device=solver.device)
        physics, _ = solver.model.branch_flow_layer(truth, batch.edge_index_dict[("bus", "connects", "bus")],
                                                   batch.edge_attr_dict[("bus", "connects", "bus")])
        model_flow = physics[:grid.nl].detach().cpu().numpy() * solver.scale
        flow_error = float(np.max(np.abs(model_flow - grid.flows(point.V)[0].real)))
        record = {"grid": name, **info, "classical_fd_relative_error": float(classical_error),
                  "neural_ad_fd_relative_error": float(neural_error), "neural_jvp_relative_error": float(jvp_error),
                  "neural_fd_step_errors": fd_errors,
                  "physics_flow_max_mw_error": flow_error, "full_reverse_seconds": duration,
                  "dtype": "float64", "device": solver.device}
        print(json.dumps(record), flush=True)
        records.append(record)
        if classical_error > 1e-6 or neural_error > 1e-4 or jvp_error > 1e-6 or flow_error > 1e-3:
            (folder / "checks.json").write_text(json.dumps(records, indent=2))
            raise AssertionError("Sensitivity validation failed; inspect results/validation/checks.json")
    (folder / "checks.json").write_text(json.dumps(records, indent=2))


def validate_finetuned(config):
    """Validate every delivered checkpoint in double precision at held-out input.

    Casting the fixed weights allows small finite steps without float32 flow
    cancellation. Serving/evaluation dtype remains separately recorded float32.
    """
    from .data import Dataset
    from .evaluation import model_specs
    torch.set_num_threads(2)
    output_path = ROOT / "results/validation/checkpoints.json"
    existing = json.loads(output_path.read_text()) if output_path.exists() else []
    rows = [r for r in existing if r["grid"] not in config["grids"]]
    for name in config["grids"]:
        grid, data = Grid(name), Dataset(name)
        point = data.point(data.indices("val")[0])
        D = data.arrays["D"][:, :3]
        for method, seed, checkpoint in model_specs(name, config):
            solver = GencoSolver(grid, checkpoint, dtype=torch.float64)
            ad = solver.factors(point, D)
            steps = {}
            for step in (1e-4, 1e-5, 1e-6):
                fd = np.column_stack([(solver.predict(grid.perturb(point, d, step))[1] -
                                       solver.predict(grid.perturb(point, d, -step))[1]) / (2 * step) for d in D.T])
                steps[str(step)] = float(np.linalg.norm(ad-fd) / max(np.linalg.norm(ad), 1e-12))
            jvp = solver.directional_factors(point, D)
            serving = GencoSolver(grid, checkpoint)
            serving_ad = serving.factors(point, D)
            serving_jvp = serving.directional_batch([point, point], D, capture=config["evaluation"].get("cuda_graph", False))
            serving_error = float(np.linalg.norm(serving_ad-serving_jvp[0]) / np.linalg.norm(serving_ad))
            batch_error = float(np.linalg.norm(serving_jvp[0]-serving_jvp[1]) / np.linalg.norm(serving_jvp[0]))
            record = {"grid": name, "method": method, "seed": seed,
                      "operating_group": int(data.indices("val")[0]), "dtype": "float64 cast of delivered weights",
                      "ad_fd_step_errors": steps, "ad_jvp_relative": float(np.linalg.norm(ad-jvp)/np.linalg.norm(ad)),
                      "serving_float32_ad_jvp_relative": serving_error, "serving_batch_independence_relative": batch_error}
            rows.append(record)
            (ROOT / "results/validation/checkpoints.json").write_text(json.dumps(rows, indent=2))
            print(f"Derivative check {name} {method} seed{seed}: {min(steps.values()):.2g}", flush=True)
            if min(steps.values()) > 1e-4 or record["ad_jvp_relative"] > 1e-8 or serving_error > 1e-3 or batch_error > 1e-3:
                raise AssertionError(f"Delivered model derivative validation failed: {record}")
            del solver, serving
            import gc
            gc.collect()
            torch.cuda.empty_cache()

"""Fixed-topology load scenarios using DataKit network/physics utilities.

The sampler is project-specific, to make the requested ranges exact. Generator
scheduled shares balance demand plus a fixed loss allowance; reference generation is recovered
from AC balance. No OPF redispatch or control switching occurs in perturbations.
"""

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pandas

from .classical import Grid, OperatingPoint, ROOT


def make_transactions(grid, count=16, seed=913):
    rng = np.random.default_rng(seed)
    sources = np.flatnonzero(grid.injection_gen_weights.sum(axis=0) > 0)
    sinks = grid.pq[grid.bus[grid.pq, 2] > 0]
    pairs = [(int(a), int(b)) for a in sources for b in sinks if a != b]
    if not pairs:
        raise ValueError("No generator-to-load transfers available")
    selected = rng.permutation(len(pairs))[:min(count, len(pairs))]
    D = np.zeros((grid.nb, len(selected)))
    for i, j in enumerate(selected):
        a, b = pairs[j]
        D[a, i], D[b, i] = 1, -1
    return D


def generate_dataset(name, config, output=None):
    output = Path(output or ROOT / "data" / name)
    output.mkdir(parents=True, exist_ok=True)
    grid = Grid(name)
    rng = np.random.default_rng(config["data_seed"] + grid.nb)
    counts = config["counts"]
    D = make_transactions(grid, config["transactions"])
    full = grid.transfer_matrix()
    directions = np.linalg.lstsq(full, D, rcond=None)[0]
    steps = np.asarray(config["response_steps_mw"])
    pairs_per_point = 2
    records, pd_values, qd_values, pg_values, voltage, flow, H = [], [], [], [], [], [], []
    pair_voltage, pair_flow, pair_index = [], [], []
    failures = []
    perturbation_limits = []
    start = time.perf_counter()
    for split, count in counts.items():
        accepted = 0
        while accepted < count:
            lo, hi = config["stress_load"] if split == "stress" else config["global_load"]
            global_scale = rng.uniform(lo, hi)
            pd = grid.bus[:, 2] * global_scale * rng.uniform(*config["local_load"], grid.nb)
            qd = grid.bus[:, 3] * global_scale * rng.uniform(*config["local_load"], grid.nb)
            # PGLib Pg initializes each active generator at mid-capacity; it
            # does not balance total demand. Normalize the plan to demand plus
            # a fixed 3% loss allowance, then let the reference absorb the
            # difference between this allowance and actual AC losses.
            pg = grid.gen[:, 1] / grid.gen[:, 1].sum() * pd.sum() * (1 + config["dispatch_loss_margin"])
            point = OperatingPoint(pd, qd, pg)
            try:
                V, solved = grid.solve(point)
                point.V = V
                true_h = grid.ac_factors(V)
                selected = rng.choice(D.shape[1], pairs_per_point, replace=False)
                pv, pf = [], []
                for index in selected:
                    vv, ff = [], []
                    for step in steps:
                        sv, sf = [], []
                        for sign in (1, -1):
                            perturbed = grid.perturb(point, D[:, index], sign * step)
                            perturbed_v, _ = grid.solve(perturbed)
                            sv.append(perturbed_v)
                            sf.append(grid.flows(perturbed_v)[0].real)
                        vv.append(sv)
                        ff.append(sf)
                    pv.append(vv)
                    pf.append(ff)
            except RuntimeError as exc:
                failures.append({"split": split, "global_scale": global_scale, "error": str(exc)})
                if len(failures) > 1000:
                    raise RuntimeError("Too many nonconvergent samples") from exc
                continue
            group = len(records)
            for pair, query in enumerate(selected):
                for step_index, step in enumerate(steps):
                    for sign_index, sign in enumerate((1, -1)):
                        changed = grid.perturb(point, D[:, query], sign * step)
                        perturbation_limits.append({"group": group, "split": split, "query": int(query),
                                                    "step_mw": float(sign * step),
                                                    **grid.limits(changed, pv[pair][step_index][sign_index])})
            digest = sha256(np.r_[pd, qd, pg].astype("<f8").tobytes()).hexdigest()
            records.append({"group": group, "split": split, "global_scale": global_scale,
                            "fingerprint": digest, **solved, **grid.limits(point, V)})
            pd_values.append(pd); qd_values.append(qd); pg_values.append(pg)
            voltage.append(V); flow.append(grid.flows(V)[0].real); H.append(true_h)
            pair_voltage.append(pv); pair_flow.append(pf); pair_index.append(selected)
            accepted += 1
            if accepted % 200 == 0 or accepted == count:
                print(f"{name} {split} {accepted}/{count}, elapsed {time.perf_counter()-start:.1f}s", flush=True)
    arrays = dict(pd=np.asarray(pd_values), qd=np.asarray(qd_values), pg=np.asarray(pg_values),
                  V=np.asarray(voltage), flow=np.asarray(flow), H=np.asarray(H, dtype=np.float32),
                  pair_V=np.asarray(pair_voltage), pair_flow=np.asarray(pair_flow),
                  pair_index=np.asarray(pair_index), D=D, steps=steps, directions=directions)
    np.savez_compressed(output / "dataset.npz", **arrays)
    table = pandas.DataFrame(records)
    if table.fingerprint.duplicated().any():
        raise AssertionError("Duplicate base points")
    table.to_csv(output / "groups.csv", index=False)
    pandas.DataFrame(perturbation_limits).to_csv(output / "perturbation_limits.csv", index=False)
    (output / "failures.json").write_text(json.dumps(failures, indent=2))
    manifest = {"grid": name, "config": config, "counts": table.split.value_counts().to_dict(),
                "sampler": "project uniform global/local scaling on DataKit Network",
                "solver": "pandapower.pypower.newtonpf with DataKit Ybus",
                "dispatch": "base Pg shares normalized to total Pd times (1+fixed loss margin); reference balances actual losses",
                "fixed_vm": grid.vm_set.tolist(), "fixed_va_rad": grid.va_set[grid.ref].tolist(),
                "nonref_bus_ids": grid.ids[grid.nonref].tolist(),
                "injection_gen_weights": grid.injection_gen_weights.tolist(),
                "case_sha256": sha256(grid.path.read_bytes()).hexdigest(),
                "dataset_sha256": sha256((output / "dataset.npz").read_bytes()).hexdigest(),
                "generation_seconds": time.perf_counter() - start,
                "nonconvergent": len(failures), "control_mode_switching": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


class Dataset:
    def __init__(self, name, root=None):
        self.name = name
        folder = Path(root or ROOT / "data" / name)
        self.groups = pandas.read_csv(folder / "groups.csv")
        with np.load(folder / "dataset.npz") as arrays:
            self.arrays = {k: arrays[k] for k in arrays.files}
        self.manifest = json.loads((folder / "manifest.json").read_text())

    def indices(self, split):
        return np.flatnonzero(self.groups.split.to_numpy() == split)

    def point(self, index):
        a = self.arrays
        return OperatingPoint(a["pd"][index], a["qd"][index], a["pg"][index], a["V"][index])

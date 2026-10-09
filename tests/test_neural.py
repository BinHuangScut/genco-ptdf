import numpy as np
import torch
from genco_ptdf.classical import Grid
from genco_ptdf.data import make_transactions, Dataset
from genco_ptdf.neural import GencoSolver


def test_boundary_batching_and_analytic_derivatives():
    torch.set_num_threads(2)
    grid = Grid("case14_ieee")
    solver = GencoSolver(grid, dtype=torch.float64)
    point = grid.default_point()
    point.V, _ = grid.solve(point)
    D = make_transactions(grid, 2)
    analytic = solver.factors(point, D)
    together = solver.directional_batch([point, point], D)
    assert np.allclose(together[0], analytic, atol=1e-8)
    assert np.allclose(together[1], analytic, atol=1e-8)
    complete = solver.factors(point)
    assert np.allclose(complete, solver.factors(point, grid.transfer_matrix()), atol=1e-8)
    weights = np.linalg.lstsq(grid.transfer_matrix(), D, rcond=None)[0]
    assert np.allclose(complete @ weights, analytic, atol=1e-8)
    if solver.device == "cuda":
        captured = solver.directional_batch([point, point], D, capture=True)
        assert np.allclose(captured, together, atol=1e-8)
        with torch.no_grad():
            next(solver.model.parameters()).mul_(1.001)
        changed = [grid.perturb(point, D[:, 0], 0.5), point]
        assert np.allclose(solver.directional_batch(changed, D, capture=True),
                           solver.directional_batch(changed, D), atol=1e-8)
    batch = solver.batch(point.pd, point.qd, point.pg, point.V)
    assert not batch.mask_dict["bus"][grid.ref, 3].any()
    assert np.allclose(batch["bus"].x[grid.ref, 3].cpu(), grid.vm_set[grid.ref])
    changed = point.V * 1.03
    other = solver.batch(point.pd, point.qd, point.pg, changed)
    assert torch.equal(batch["bus"].x, other["bus"].x)  # targets cannot leak into inputs


def test_group_split_and_training_perturbation_identity():
    for name in ("case14_ieee", "case118_ieee"):
        dataset = Dataset(name)
        assert not dataset.groups.fingerprint.duplicated().any()
        assert dataset.groups.split.value_counts().to_dict() == {"train": 2000, "val": 400, "test": 400, "stress": 200}
        grid = Grid(name)
        ids = [dataset.indices(s)[0] for s in ("train", "val", "test", "stress")]
        for i in ids:
            point = dataset.point(i)
            j = dataset.arrays["pair_index"][i, 0]
            perturbation = grid.perturb(point, dataset.arrays["D"][:, j], 0.1)
            solved, _ = grid.solve(perturbation)
            assert np.allclose(solved, dataset.arrays["pair_V"][i, 0, 0, 0], atol=1e-10)

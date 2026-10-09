import numpy as np
import pytest
import torch
from genco_ptdf.classical import Grid
from genco_ptdf.data import Dataset
from genco_ptdf.neural import GencoSolver
from genco_ptdf.capture import CapturedStep, objective


@pytest.mark.parametrize("mode", ["pf", "response"])
def test_cuda_capture_preserves_objective_and_updates(mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA graph execution needs CUDA")
    torch.set_num_threads(2)
    grid = Grid("case14_ieee")
    data = Dataset(grid.name).arrays
    # Double precision isolates graph-execution errors from float32 atomic
    # reductions and cancellation in the response difference.
    eager, graph_solver = GencoSolver(grid, dtype=torch.float64), GencoSolver(grid, dtype=torch.float64)
    optimizers = [torch.optim.AdamW(s.model.parameters(), lr=1e-4, capturable=True, fused=True)
                  for s in (eager, graph_solver)]
    ids = np.arange(24)
    batches = [s.batch(data["pd"][ids], data["qd"][ids], data["pg"][ids], data["V"][ids])
               for s in (eager, graph_solver)]
    den = torch.ones((8, 1), device="cuda", dtype=torch.float64)
    truth = torch.zeros((8, grid.nl), device="cuda", dtype=torch.float64)
    selection = batches[0].mask_dict["bus"][:, 3:5].nonzero(as_tuple=True)
    out, _ = eager.forward(batches[0])
    original, state, physics = eager.pf_loss(out, batches[0])
    fast = objective(eager, batches[0], selection, den, truth, "pf", 1)
    assert torch.allclose(original, fast[0], rtol=1e-5, atol=1e-8)
    graph = CapturedStep(graph_solver, optimizers[1], batches[1], den, truth, mode, 1)
    for iteration in range(2):
        for batch in batches:
            batch["bus"].x[:, 0].mul_(1.01)
        optimizers[0].zero_grad(set_to_none=True)
        reference = objective(eager, batches[0], selection, den, truth, mode, 1)
        reference[0].backward()
        torch.nn.utils.clip_grad_norm_(eager.model.parameters(), 1.)
        optimizers[0].step()
        result = graph.run(batches[1], den, truth)
        torch.cuda.synchronize()
        assert torch.allclose(reference[0], result[0], rtol=1e-8, atol=1e-10)
        for (name, a), b in zip(eager.model.named_parameters(), graph_solver.model.parameters()):
            assert torch.allclose(a, b, rtol=1e-8, atol=1e-9), (iteration, name, float((a-b).detach().abs().max()),
                   float(torch.linalg.norm(a.grad-b.grad)), float(torch.linalg.norm(a.grad)),
                   float(optimizers[0].state[a]["step"]), float(optimizers[1].state[b]["step"]))

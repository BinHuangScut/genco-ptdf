"""Check whether the unchanged GENCO forward/backward can use CUDA graphs."""
import time
import torch
from genco_ptdf.classical import Grid
from genco_ptdf.data import Dataset
from genco_ptdf.neural import GencoSolver

torch.set_num_threads(2)
grid = Grid("case14_ieee")
dataset = Dataset(grid.name)
a = dataset.arrays
solver = GencoSolver(grid)
batch = solver.batch(a["pd"][:24], a["qd"][:24], a["pg"][:24], a["V"][:24])
mask = batch.mask_dict["bus"][:, 3:5]
selection = mask.nonzero(as_tuple=True)
target = batch["bus"].y[:, 3:5]
optimizer = torch.optim.AdamW(solver.model.parameters(), lr=1e-4, capturable=True, fused=True)
def step():
    optimizer.zero_grad(set_to_none=True)
    out, flow = solver.forward(batch)
    state = torch.mean((out["bus"][selection] - target[selection]) ** 2)
    residuals = solver.model.layer_residuals
    keys = sorted(residuals)
    weights = [0.5**(len(keys)-i-1) for i in range(len(keys))]
    physics = sum(residuals[k]*w for k,w in zip(keys,weights))/sum(weights)
    loss = 0.9*state + 0.1*physics
    loss.backward()
    torch.nn.utils.clip_grad_norm_(solver.model.parameters(), 1.0)
    optimizer.step()
    return loss
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3):
        step()
torch.cuda.current_stream().wait_stream(stream)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    loss = step()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(250):
    graph.replay()
torch.cuda.synchronize()
print("250 steps", time.perf_counter()-start, "loss", float(loss), flush=True)

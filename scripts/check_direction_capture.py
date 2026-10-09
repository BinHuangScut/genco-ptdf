import time
import numpy as np
import torch
from genco_ptdf.classical import Grid
from genco_ptdf.data import Dataset
from genco_ptdf.neural import GencoSolver

torch.set_num_threads(2)
g=Grid("case14_ieee")
d=Dataset(g.name)
s=GencoSolver(g)
D=d.arrays["D"][:,:4]
points=[d.point(i) for i in d.indices("val")[:64]]
reference=s.directional_batch(points,D)
start=time.perf_counter()
value=s.directional_batch(points,D,capture=True)
print("capture seconds", time.perf_counter()-start, "relative",np.linalg.norm(value-reference)/np.linalg.norm(reference),flush=True)
start=time.perf_counter()
for _ in range(7):
    s.directional_batch(points,D,capture=True)
print("7 replays seconds", time.perf_counter()-start,flush=True)
with torch.no_grad():
    next(s.model.parameters()).mul_(1.001)
points=points[1:]+points[:1]
reference=s.directional_batch(points,D)
value=s.directional_batch(points,D,capture=True)
print("changed weights/inputs relative",np.linalg.norm(value-reference)/np.linalg.norm(reference),flush=True)

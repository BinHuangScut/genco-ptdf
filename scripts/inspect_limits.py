import numpy as np
from genco_ptdf.classical import Grid
from genco_ptdf.data import Dataset

for name in ("case14_ieee", "case118_ieee"):
    g=Grid(name)
    d=Dataset(name)
    p=d.point(d.indices("test")[0])
    S=g.base*p.V*np.conj(g.Ybus@p.V)
    print(name, "sum Pd", p.pd.sum(), "sum scheduled Pg", p.pg.sum(), "actual slack", S.real[g.ref]+p.pd[g.ref],
          "slack bounds", g.gen[g.gb==g.ref[0],8:10], "zero Pmax generators", g.gen[g.gen[:,8]==0,1],
          "branch over count", ((abs(g.flows(p.V)[0])>g.branch[:,5]) & (g.branch[:,5]>0)).sum(),
          "PQ generators", g.ids[np.intersect1d(g.gb,g.pq)], flush=True)

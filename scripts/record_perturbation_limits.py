"""Add the same limit records as the generator to existing saved states."""
import pandas as pd
from genco_ptdf.classical import Grid, ROOT
from genco_ptdf.data import Dataset

for name in ("case14_ieee", "case118_ieee"):
    grid, data = Grid(name), Dataset(name)
    a=data.arrays
    rows=[]
    for i in range(len(data.groups)):
        point=data.point(i)
        for pair, query in enumerate(a["pair_index"][i]):
            for j, step in enumerate(a["steps"]):
                for k, sign in enumerate((1,-1)):
                    changed=grid.perturb(point,a["D"][:,query],sign*step)
                    rows.append({"group":i,"split":data.groups.iloc[i]["split"],"query":int(query),
                                 "step_mw":float(sign*step),**grid.limits(changed,a["pair_V"][i,pair,j,k])})
    pd.DataFrame(rows).to_csv(ROOT/"data"/name/"perturbation_limits.csv",index=False)
    print(name,len(rows),"perturbation limit records",flush=True)

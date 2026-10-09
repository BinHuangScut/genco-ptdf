"""Refresh metadata-only limits without changing grouped samples or labels."""
from genco_ptdf.classical import Grid, ROOT
from genco_ptdf.data import Dataset

for name in ("case14_ieee", "case118_ieee"):
    dataset, grid = Dataset(name), Grid(name)
    for i in range(len(dataset.groups)):
        for key, value in grid.limits(dataset.point(i), dataset.arrays["V"][i]).items():
            dataset.groups.loc[i, key] = value
    dataset.groups.to_csv(ROOT / "data" / name / "groups.csv", index=False)
    print(name, dataset.groups.groupby("split")[["voltage_limit", "q_limit", "p_limit", "branch_limit"]].mean().to_dict(), flush=True)

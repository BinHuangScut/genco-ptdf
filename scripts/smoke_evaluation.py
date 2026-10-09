"""Small real-data accuracy-path check; this is not a performance benchmark."""
import json
import numpy as np
import torch
from genco_ptdf.classical import Grid, ROOT
from genco_ptdf.data import Dataset
from genco_ptdf.neural import GencoSolver
from genco_ptdf.evaluation import metrics

torch.set_num_threads(2)
records=[]
for name in ("case14_ieee", "case118_ieee"):
    g,d=Grid(name),Dataset(name)
    s=GencoSolver(g)
    ids=d.indices("test")[:16]
    points=[d.point(i) for i in ids]
    prediction=s.directional_batch(points,d.arrays["D"])
    truth=d.arrays["H"][ids]@d.arrays["directions"]
    assert prediction.shape==truth.shape
    with torch.no_grad():
        out,flows=s.forward(s.batch(d.arrays["pd"][ids],d.arrays["qd"][ids],d.arrays["pg"][ids]))
    assert out["bus"][:,:2].reshape(len(ids),g.nb,2).shape==(16,g.nb,2)
    record={"grid":name,"n":len(ids),"transactions":prediction.shape[-1],**metrics(prediction,truth),
            "flow_mae_mw":float(np.mean(abs(flows.cpu().numpy()-d.arrays["flow"][ids])))}
    records.append(record)
    print(record,flush=True)
    del s
    torch.cuda.empty_cache()
(ROOT/"results/validation/evaluation_smoke.json").write_text(json.dumps(records,indent=2))

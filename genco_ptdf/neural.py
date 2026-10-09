"""Official GENCO model with an explicit fixed-reference-voltage PF adapter."""

from pathlib import Path
import json
import time

import numpy as np
import torch
import yaml
from torch_geometric.data import HeteroData, Batch
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.datasets.masking import AddPFHeteroMask
from gridfm_graphkit.training.loss import MaskedBusMSE, LayeredWeightedPhysicsLoss

from .classical import ROOT, OperatingPoint


def fetch_checkpoints(grids):
    from huggingface_hub import snapshot_download
    source = json.loads((ROOT / "configs/provenance.json").read_text())["checkpoint"]
    repo, revision = source["repo"], source["revision"]
    folder = ROOT / "checkpoints/official"
    snapshot_download(repo_id=repo, revision=revision, local_dir=folder,
                      allow_patterns=[f"{g}/small/seed0/*" for g in grids] + ["README.md", "LICENSE*"])
    (folder / "source.json").write_text(json.dumps({"repo": repo, "revision": revision}, indent=2))


class GencoSolver:
    def __init__(self, grid, checkpoint=None, device=None, dtype=torch.float32):
        self.grid = grid
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        folder = ROOT / "checkpoints/official" / grid.name / "small/seed0"
        config = yaml.safe_load((folder / f"{grid.name}_small_seed0.yaml").read_text())
        self.args = NestedNamespace(**config)
        self.model = GNS_heterogeneous(self.args).to(self.device, dtype=dtype)
        weights = torch.load(checkpoint or folder / "best_model_state_dict.pt", map_location="cpu", weights_only=True)
        if "state_dict" in weights:
            weights = weights["state_dict"]
        if weights and all(key.startswith("model.") for key in weights):
            weights = {key.removeprefix("model."): value for key, value in weights.items()}
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        self.normalizer = HeteroDataMVANormalizer(self.args)
        stats = torch.load(folder / "normalizer_stats.pt", weights_only=True)
        self.normalizer.fit_from_dict(stats.get(grid.name, stats))
        self.scale = float(self.normalizer.baseMVA)
        self.template = self._template()
        self._batches = {}
        self._direction_graphs = {}
        self.state_loss = MaskedBusMSE(NestedNamespace(), self.args)
        self.physics_loss = LayeredWeightedPhysicsLoss(NestedNamespace(base_weight=0.5), self.args)
        self.gen_weights = torch.tensor(grid.injection_gen_weights, device=self.device, dtype=dtype)
        self.load_weights = torch.tensor(grid.injection_load_weights, device=self.device, dtype=dtype)

    def _template(self):
        g = self.grid
        data = HeteroData()
        bx = np.zeros((g.nb, 15))
        bx[:, 0:2] = g.bus[:, 2:4]
        bx[:, 3] = g.vm_set
        bx[:, 4] = np.rad2deg(g.va_set)
        bx[:, 5:8] = np.column_stack([g.bus[:, 1] == t for t in (1, 2, 3)])
        bx[:, 8:10] = g.bus[:, [12, 11]]
        bx[:, 10] = np.bincount(g.gb, weights=g.gen[:, 4], minlength=g.nb)
        bx[:, 11] = np.bincount(g.gb, weights=g.gen[:, 3], minlength=g.nb)
        bx[:, 12:14] = g.bus[:, 4:6] / g.base
        bx[:, 14] = g.bus[:, 9]
        gx = np.zeros((g.ng, 6))
        gx[:, :3] = g.gen[:, [1, 9, 8]]
        for i, row in enumerate(g.cost):
            coefficients = list(row[4:4 + int(row[3])])[::-1]
            gx[i, 3:3 + min(3, len(coefficients))] = coefficients[:3]
        forward = np.column_stack([np.zeros((g.nl, 2)), g.Yff.real, g.Yff.imag,
                                   g.Yft.real, g.Yft.imag, g.branch[:, 8],
                                   g.branch[:, 11], g.branch[:, 12], g.branch[:, 5]])
        reverse = np.column_stack([np.zeros((g.nl, 2)), g.Ytt.real, g.Ytt.imag,
                                   g.Ytf.real, g.Ytf.imag, g.branch[:, 8],
                                   g.branch[:, 11], g.branch[:, 12], g.branch[:, 5]])
        data["bus"].x = torch.tensor(bx, dtype=torch.float32)
        data["bus"].y = data["bus"].x[:, :5].clone()
        data["gen"].x = torch.tensor(gx, dtype=torch.float32)
        data["gen"].y = data["gen"].x[:, :1].clone()
        data["bus", "connects", "bus"].edge_index = torch.tensor(np.vstack([np.r_[g.f, g.t], np.r_[g.t, g.f]]))
        data["bus", "connects", "bus"].edge_attr = torch.tensor(np.vstack([forward, reverse]), dtype=torch.float32)
        data["gen", "connected_to", "bus"].edge_index = torch.tensor(np.vstack([np.arange(g.ng), g.gb]))
        data["bus", "connected_to", "gen"].edge_index = torch.tensor(np.vstack([g.gb, np.arange(g.ng)]))
        data.scenario_id = torch.tensor([0])
        self.normalizer.transform(data)
        data = AddPFHeteroMask()(data)
        # Standard PF fixes both V and angle at the reference bus. Use the given
        # case setpoint, never a solved voltage label, to change the upstream mask.
        data.mask_dict["bus"][g.ref, 3] = False
        for kind in ("bus", "gen"):
            data[kind].x = torch.where(data.mask_dict[kind], 0., data[kind].x)
        relation = ("bus", "connects", "bus")
        data[relation].edge_attr = torch.where(data.mask_dict["branch"], 0., data[relation].edge_attr)
        return data

    def batch(self, pd, qd, pg, V=None):
        """Build graph inputs independently of any unknown solved state."""
        g = self.grid
        pd, qd, pg = [np.atleast_2d(x) for x in (pd, qd, pg)]
        size = len(pd)
        if size not in self._batches:
            batch = Batch.from_data_list([self.template] * size).to(self.device)
            for store in batch.stores:
                for key, val in store.items():
                    if torch.is_tensor(val) and val.is_floating_point():
                        store[key] = val.to(self.dtype)
            self._batches[size] = batch
        batch = self._batches[size].clone()
        bx = batch["bus"].x.reshape(size, g.nb, 15)
        gx = batch["gen"].x.reshape(size, g.ng, 6)
        bx[:, :, 0] = torch.as_tensor(pd / self.scale, device=self.device, dtype=self.dtype)
        bx[:, :, 1] = torch.as_tensor(qd / self.scale, device=self.device, dtype=self.dtype)
        gen_mask = batch.mask_dict["gen"][:, 0].reshape(size, g.ng)
        supplied_pg = torch.as_tensor(pg / self.scale, device=self.device, dtype=self.dtype)
        gx[:, :, 0] = torch.where(gen_mask, 0., supplied_pg)
        if V is not None:
            values = np.atleast_2d(V)
            target = batch["bus"].y.reshape(size, g.nb, 5)
            target[:, :, 3] = torch.as_tensor(np.abs(values), device=self.device, dtype=self.dtype)
            target[:, :, 4] = torch.as_tensor(np.angle(values), device=self.device, dtype=self.dtype)
        return batch

    def forward(self, batch):
        output = self.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.mask_dict)
        flows, _ = self.model.branch_flow_layer(output["bus"], batch.edge_index_dict[("bus", "connects", "bus")],
                                                batch.edge_attr_dict[("bus", "connects", "bus")])
        flows = flows.reshape(-1, 2 * self.grid.nl)[:, :self.grid.nl] * self.scale
        return output, flows

    def pf_loss(self, output, batch):
        common = dict(pred_dict=output, target_dict=batch.y_dict, edge_index=batch.edge_index_dict,
                      edge_attr=batch.edge_attr_dict, mask_dict=batch.mask_dict, model=self.model)
        state = self.state_loss(**common)["loss"]
        physics = self.physics_loss(output, batch.y_dict, model=self.model)["loss"]
        return 0.9 * state + 0.1 * physics, state, physics

    def predict(self, point):
        self.model.eval()
        with torch.no_grad():
            out, flows = self.forward(self.batch(point.pd, point.qd, point.pg))
        state = out["bus"].detach().cpu().numpy()
        return state[:, 0] * np.exp(1j * state[:, 1]), flows[0].detach().cpu().numpy()

    def factors(self, point, transfers=None, chunk=16):
        """Block reverse-mode AD in physical MW injection coordinates."""
        self.model.eval()
        D = self.grid.transfer_matrix(transfers)
        batch = self.batch(point.pd, point.qd, point.pg)
        u = torch.zeros(self.grid.nb, device=self.device, dtype=self.dtype, requires_grad=True)
        delta_bus = torch.zeros_like(batch["bus"].x)
        delta_bus[:, 0] = -u * self.load_weights / self.scale
        delta_gen = torch.zeros_like(batch["gen"].x)
        delta_gen[:, 0] = (self.gen_weights @ u) / self.scale
        batch["bus"].x = batch["bus"].x + delta_bus
        batch["gen"].x = batch["gen"].x + delta_gen
        _, flows = self.forward(batch)
        values = flows[0]
        rows = []
        for start in range(0, self.grid.nl, chunk):
            count = min(chunk, self.grid.nl - start)
            selected = values[start:start + count]
            jac = torch.autograd.grad(selected, u, grad_outputs=torch.eye(count, device=self.device, dtype=self.dtype),
                                      is_grads_batched=True, retain_graph=start + count < self.grid.nl)[0]
            rows.append(jac.detach())
        derivatives = torch.cat(rows).cpu().numpy()
        if transfers is None:
            # The reference coordinate has identically zero input mapping.
            # Selecting columns gives the same basis as [I; -1] and avoids a
            # redundant dense matrix product for the complete sensitivity.
            return derivatives[:, self.grid.nonref].astype(float)
        return derivatives @ D

    def directional_factors(self, point, directions):
        """Reverse-over-reverse JVP, used for inexpensive validation directions."""
        return self.directional_batch([point], directions)[0]

    def directional_batch(self, points, directions, capture=False):
        """Independent graph JVPs; return (operating point, branch, transfer)."""
        self.model.eval()
        g = self.grid
        D = g.transfer_matrix(directions)
        size = len(points)
        original = self.batch(np.asarray([p.pd for p in points]),
                              np.asarray([p.qd for p in points]),
                              np.asarray([p.pg for p in points]))
        gpu_directions = torch.as_tensor(D, device=self.device, dtype=self.dtype)
        if capture and self.device == "cuda":
            key = (size, D.tobytes())
            if key not in self._direction_graphs:
                static = original.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self._directional_tensor(static, gpu_directions)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = self._directional_tensor(static, gpu_directions)
                self._direction_graphs[key] = (graph, static, result, gpu_directions)
            graph, static, result, _ = self._direction_graphs[key]
            for kind in ("bus", "gen"):
                static[kind].x.copy_(original[kind].x)
            graph.replay()
            return result.detach().cpu().numpy()
        return self._directional_tensor(original, gpu_directions).detach().cpu().numpy()

    def _directional_tensor(self, original, D):
        size = original.num_graphs
        g = self.grid
        def mapping(u):
            batch = original.clone()
            db = torch.zeros_like(batch["bus"].x)
            dg = torch.zeros_like(batch["gen"].x)
            db[:, 0] = (-u * self.load_weights / self.scale).flatten()
            dg[:, 0] = (u @ self.gen_weights.T / self.scale).flatten()
            batch["bus"].x = batch["bus"].x + db
            batch["gen"].x = batch["gen"].x + dg
            return self.forward(batch)[1]
        u = torch.zeros((size, g.nb), device=self.device, dtype=self.dtype)
        result = []
        for direction in D.T:
            tangent = direction.expand(size, -1)
            _, jvp = torch.autograd.functional.jvp(mapping, u, tangent)
            result.append(jvp.detach())
        return torch.stack(result, dim=-1)

"""Sparse Newton power flow and analytic AC/DC transfer sensitivities.

Network loading and admittance construction use pinned GridFM DataKit. Analytic
derivatives use pandapower's PYPOWER routines. All injections are positive into
the network. Branch outputs are sending-end MW in original branch order.
"""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.sparse import bmat, csr_matrix, hstack
from scipy.sparse.linalg import splu
from pandapower.pypower.dSbus_dV import dSbus_dV
from pandapower.pypower.dSbr_dV import dSbr_dV
from pandapower.pypower.makeBdc import makeBdc
from pandapower.pypower.newtonpf import newtonpf
from pandapower.pypower.idx_bus import bus_cols
from pandapower.pypower.idx_bus_dc import dc_bus_cols
from pandapower.pypower.idx_brch_dc import branch_dc_cols
from pandapower.pypower.idx_svc import svc_cols
from pandapower.pypower.idx_tcsc import tcsc_cols
from pandapower.pypower.idx_ssc import ssc_cols
from pandapower.pypower.idx_vsc import vsc_cols
from pandapower.pypower.idx_source_dc import source_dc_cols

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class OperatingPoint:
    pd: np.ndarray
    qd: np.ndarray
    pg: np.ndarray
    V: np.ndarray | None = None


class Grid:
    def __init__(self, name: str, path: str | Path | None = None):
        from matpowercaseframes import CaseFrames
        from gridfm_datakit.network import Network, makeYbus, branch_vectors

        path = Path(path) if path else ROOT / "vendor/gridfm-datakit/gridfm_datakit/grids" / f"pglib_opf_{name}_corrected.m"
        frames = CaseFrames(str(path))
        mpc = {k: np.asarray(getattr(frames, k)) if hasattr(getattr(frames, k), "values") else getattr(frames, k)
               for k in frames._attributes}
        self.network = Network(mpc)
        self.name, self.path = name, path
        self.bus = self.network.buses.copy()
        self.gen = self.network.gens.copy()
        self.branch = self.network.branches.copy()
        self.cost = self.network.gencosts.copy()
        self.base = float(self.network.baseMVA)
        self.nb, self.ng, self.nl = len(self.bus), len(self.gen), len(self.branch)
        self.ids = self.network.original_bus_indices.copy()
        self.ref = np.flatnonzero(self.bus[:, 1] == 3)
        self.pv = np.flatnonzero(self.bus[:, 1] == 2)
        self.pq = np.flatnonzero(self.bus[:, 1] == 1)
        if len(self.ref) != 1:
            raise ValueError("Exactly one reference bus is required")
        if not np.all(self.branch[:, 10] == 1) or not np.all(self.gen[:, 7] == 1):
            raise ValueError("This fixed-topology experiment requires all elements in service")
        if not np.array_equal(self.bus[:, 0], np.arange(self.nb)):
            raise ValueError("Bus rows must be ordered by their internal indices")
        self.nonref = np.r_[self.pv, self.pq]
        self.gb = self.gen[:, 0].astype(int)
        self.f, self.t = self.branch[:, 0].astype(int), self.branch[:, 1].astype(int)
        self.Ybus, self.Yf, self.Yt = makeYbus(self.base, self.bus, self.branch)
        self.Ytt, self.Yff, self.Yft, self.Ytf = branch_vectors(self.branch, self.nl)
        self.vm_set = self.bus[:, 7].copy()
        self.va_set = np.deg2rad(self.bus[:, 8])
        self.injection_gen_weights = np.zeros((self.ng, self.nb))
        for bus in self.pv:
            generators = np.flatnonzero((self.gb == bus) & (self.gen[:, 8] > 0))
            if len(generators):
                weights = self.gen[generators, 8]
                self.injection_gen_weights[generators, bus] = weights / weights.sum()
        self.injection_load_weights = 1 - self.injection_gen_weights.sum(axis=0)
        self.injection_load_weights[self.ref] = 0
        self._dc = None
        # Direct pandapower NR kernel with the exact DataKit admittance model.
        # Empty device tables disable DC/FACTS extensions in this fixed AC grid.
        pp_bus = np.zeros((self.nb, bus_cols))
        pp_bus[:, :self.bus.shape[1]] = self.bus
        self._ppci = {"baseMVA": self.base, "bus": pp_bus,
                      "gen": self.gen.copy(), "branch": self.branch, "internal": {}}
        for name, columns in (("bus_dc", dc_bus_cols), ("branch_dc", branch_dc_cols),
                              ("svc", svc_cols), ("tcsc", tcsc_cols), ("ssc", ssc_cols),
                              ("vsc", vsc_cols), ("source_dc", source_dc_cols)):
            self._ppci[name] = np.empty((0, columns))

    def default_point(self):
        return OperatingPoint(self.bus[:, 2].copy(), self.bus[:, 3].copy(), self.gen[:, 1].copy())

    def transfer_matrix(self, transfers=None):
        if transfers is None:
            D = np.zeros((self.nb, self.nb - 1))
            D[self.nonref, np.arange(self.nb - 1)] = 1
            D[self.ref[0], :] = -1
            return D
        D = np.asarray(transfers, dtype=float)
        if D.ndim == 1:
            D = D[:, None]
        if D.shape[0] != self.nb or not np.isfinite(D).all():
            raise ValueError("Transfers must have nb rows and finite entries")
        if not np.allclose(D.sum(axis=0), 0, atol=1e-12):
            raise ValueError("Each specified transfer must sum to zero")
        return D

    def perturb(self, point, direction, mw):
        direction = np.asarray(direction)
        pd = point.pd - mw * direction * self.injection_load_weights
        pg = point.pg + mw * self.injection_gen_weights @ direction
        return OperatingPoint(pd, point.qd.copy(), pg, point.V)

    def jacobian(self, V):
        dVm, dVa = dSbus_dV(self.Ybus, V)
        n, q = self.nonref, self.pq
        return bmat([[dVa[n][:, n].real, dVm[n][:, q].real],
                     [dVa[q][:, n].imag, dVm[q][:, q].imag]], format="csc")

    def solve(self, point: OperatingPoint, tol=1e-10, max_iter=30):
        pg_bus = np.bincount(self.gb, weights=point.pg, minlength=self.nb)
        qg_bus = np.bincount(self.gb, weights=self.gen[:, 2], minlength=self.nb)
        specified = (pg_bus - point.pd + 1j * (qg_bus - point.qd)) / self.base
        if point.V is None:
            V = self.vm_set * np.exp(1j * self.va_set)
        else:
            V = np.asarray(point.V).copy()
        vm, va = np.abs(V), np.angle(V)
        vm[np.r_[self.ref, self.pv]] = self.vm_set[np.r_[self.ref, self.pv]]
        va[self.ref] = self.va_set[self.ref]
        V = vm * np.exp(1j * va)
        self._ppci["bus"][:, 2:4] = np.column_stack([point.pd, point.qd])
        self._ppci["gen"][:, 1] = point.pg
        options = {"tolerance_mva": tol, "max_iteration": max_iter, "numba": False,
                   "algorithm": "nr", "voltage_depend_loads": False,
                   "distributed_slack": False, "v_debug": False,
                   "use_umfpack": False, "permc_spec": "COLAMD", "tdpf": False}
        start = time.perf_counter()
        V, converged, iteration, *_ = newtonpf(
            self.Ybus, specified, V, self.ref, self.pv, self.pq, self._ppci, options)
        mismatch = V * np.conj(self.Ybus @ V) - specified
        residual = np.r_[mismatch[self.nonref].real, mismatch[self.pq].imag]
        if not converged or not np.isfinite(V).all() or np.linalg.norm(residual, np.inf) >= tol:
            raise RuntimeError(f"AC power flow failed for {self.name}")
        return V, {"iterations": int(iteration), "residual_pu": float(np.max(np.abs(residual))),
                   "pf_seconds": time.perf_counter() - start,
                   "solver": "pandapower.pypower.newtonpf"}

    def flows(self, V):
        Sf = self.base * V[self.f] * np.conj(self.Yf @ V)
        St = self.base * V[self.t] * np.conj(self.Yt @ V)
        return Sf, St

    def ac_factors(self, V, transfers=None):
        D = self.transfer_matrix(transfers)
        dVa, dVm, *_ = dSbr_dV(self.branch, self.Yf, self.Yt, V)
        K = hstack([dVa[:, self.nonref].real, dVm[:, self.pq].real], format="csr")
        rhs = np.vstack([D[self.nonref] / self.base, np.zeros((len(self.pq), D.shape[1]))])
        return self.base * (K @ splu(self.jacobian(V)).solve(rhs))

    def dc_factors(self, transfers=None):
        D = self.transfer_matrix(transfers)
        if self._dc is None:
            # MATPOWER convention: branch susceptance 1/(x*tap).
            Bbus, Bf, *_ = makeBdc(self.bus, self.branch)
            self._dc = (splu(Bbus[self.nonref][:, self.nonref].tocsc()), Bf[:, self.nonref])
        lu, Bf = self._dc
        return np.asarray(Bf @ lu.solve(D[self.nonref]))

    def limits(self, point, V):
        S = self.base * V * np.conj(self.Ybus @ V)
        qg = S.imag + point.qd
        qmax = np.bincount(self.gb, weights=self.gen[:, 3], minlength=self.nb)
        qmin = np.bincount(self.gb, weights=self.gen[:, 4], minlength=self.nb)
        controlled = np.r_[self.ref, self.pv]
        scheduled = point.pg.copy()
        reference_generators = np.flatnonzero(self.gb == self.ref[0])
        weights = self.gen[reference_generators, 8]
        weights = weights / weights.sum()
        scheduled[reference_generators] = (S.real[self.ref[0]] + point.pd[self.ref[0]]) * weights
        Sf, St = self.flows(V)
        rate = self.branch[:, 5]
        return {"voltage_limit": bool(np.any((abs(V) < self.bus[:, 12]) | (abs(V) > self.bus[:, 11]))),
                "q_limit": bool(np.any((qg[controlled] < qmin[controlled]) | (qg[controlled] > qmax[controlled]))),
                "p_limit": bool(np.any((scheduled < self.gen[:, 9]) | (scheduled > self.gen[:, 8]))),
                "branch_limit": bool(np.any((rate > 0) & (np.maximum(abs(Sf), abs(St)) > rate)))}


def compute_ac_ptdf(case, operating_point=None, transfers=None, method="ac_jacobian", checkpoint=None):
    """Return factors, branch IDs and the fixed-control boundary convention.

`transfers` is an nb-by-q balanced injection matrix, using internal bus rows.
With None, columns are non-reference buses in PV-then-PQ order to reference.
"""
    grid = case if isinstance(case, Grid) else Grid(case)
    point = operating_point or grid.default_point()
    if method == "ac_jacobian":
        V, _ = grid.solve(point)
        factors = grid.ac_factors(V, transfers)
    elif method == "dc":
        factors = grid.dc_factors(transfers)
    elif method == "genco":
        from .neural import GencoSolver
        factors = GencoSolver(grid, checkpoint=checkpoint).factors(point, transfers)
    else:
        raise ValueError(f"Unknown method {method!r}")
    return {"factors": factors, "branch_ids": np.arange(grid.nl),
            "bus_ids": grid.ids, "column_bus_ids": grid.ids[grid.nonref] if transfers is None else None,
            "boundary": {"reference_bus": int(grid.ids[grid.ref[0]]), "units": "MW/MW",
                         "fixed_qd": True, "fixed_bus_types": True, "fixed_voltage_setpoints": True,
                         "bus_types": grid.bus[:, 1].astype(int).tolist(),
                         "voltage_setpoints_pu": grid.vm_set[np.r_[grid.ref, grid.pv]].tolist(),
                         "controlled_bus_ids": grid.ids[np.r_[grid.ref, grid.pv]].tolist(),
                         "reference_angle_rad": float(grid.va_set[grid.ref[0]]),
                         "fixed_qd_mvar": point.qd.tolist(),
                         "scheduled_pg_mw": point.pg.tolist(),
                         "generator_bus_ids": grid.ids[grid.gb].tolist(),
                         "injection_gen_weights": grid.injection_gen_weights.tolist(),
                         "tap_ratios": grid.branch[:, 8].tolist(),
                         "phase_shifts_deg": grid.branch[:, 9].tolist(),
                         "branch_id_convention": "zero-based original branch row",
                         "branch_from_bus_ids": grid.ids[grid.f].tolist(),
                         "branch_to_bus_ids": grid.ids[grid.t].tolist(),
                         "incremental_losses": "reference bus", "branch_end": "from"}}

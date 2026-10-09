import numpy as np
import pytest
from genco_ptdf.classical import Grid, OperatingPoint, compute_ac_ptdf
from genco_ptdf.data import make_transactions


@pytest.mark.parametrize("name", ["case14_ieee", "case118_ieee"])
def test_analytic_matches_independent_ac_perturbations(name):
    grid = Grid(name)
    point = grid.default_point()
    point.V, info = grid.solve(point)
    assert info["residual_pu"] < 1e-9
    D = make_transactions(grid, 4)
    # Also exercise a reference-bus endpoint.
    slack_direction = np.zeros(grid.nb)
    slack_direction[grid.nonref[0]], slack_direction[grid.ref[0]] = 1, -1
    D = np.column_stack([D, slack_direction])
    analytic = grid.ac_factors(point.V, D)
    for eps in (0.1, 0.01):
        values = []
        for direction in D.T:
            vp, _ = grid.solve(grid.perturb(point, direction, eps))
            vm, _ = grid.solve(grid.perturb(point, direction, -eps))
            values.append((grid.flows(vp)[0].real - grid.flows(vm)[0].real) / (2 * eps))
        finite = np.column_stack(values)
        assert np.linalg.norm(finite - analytic) / np.linalg.norm(analytic) < 1e-6
    # Fixed voltage controls have no response to the perturbation.
    assert np.allclose(abs(vp[np.r_[grid.ref, grid.pv]]), grid.vm_set[np.r_[grid.ref, grid.pv]])
    assert np.allclose(np.angle(vp[grid.ref]), grid.va_set[grid.ref])
    assert np.allclose(analytic, grid.ac_factors(point.V) @ np.linalg.lstsq(grid.transfer_matrix(), D, rcond=None)[0])


def test_units_signs_and_unbalanced_transfers():
    grid = Grid("case14_ieee")
    D = make_transactions(grid, 3)
    point = grid.default_point()
    out = compute_ac_ptdf(grid, point, D)
    assert out["factors"].shape == (grid.nl, 3)
    assert out["boundary"]["units"] == "MW/MW"
    assert np.allclose(out["factors"], -compute_ac_ptdf(grid, point, -D)["factors"])
    with pytest.raises(ValueError):
        grid.transfer_matrix(np.ones(grid.nb))
    assert np.allclose(grid.dc_factors(D), -grid.dc_factors(-D))


def test_transformer_phase_shift():
    grid = Grid("case14_ieee")
    # Test nonzero phase shift by rebuilding the exact admittance model.
    from gridfm_datakit.network import makeYbus
    grid.branch[0, 9] = 7.0
    grid.Ybus, grid.Yf, grid.Yt = makeYbus(grid.base, grid.bus, grid.branch)
    point = grid.default_point()
    point.V, _ = grid.solve(point)
    D = make_transactions(grid, 1)
    vp, _ = grid.solve(grid.perturb(point, D[:, 0], 0.01))
    vm, _ = grid.solve(grid.perturb(point, D[:, 0], -0.01))
    finite = (grid.flows(vp)[0].real-grid.flows(vm)[0].real)/0.02
    assert np.allclose(grid.ac_factors(point.V, D)[:, 0], finite, atol=1e-7)
    weights = grid.injection_gen_weights
    assert np.allclose(weights.sum(axis=0)[weights.sum(axis=0) > 0], 1)


@pytest.mark.parametrize("name", ["case14_ieee", "case118_ieee"])
def test_independent_pandapower_solution(name):
    import pandapower as pp
    from pandapower.converter.pypower.from_ppc import from_ppc
    grid = Grid(name)
    point = grid.default_point()
    point.pd *= 1.08
    point.qd *= 1.03
    point.pg *= 1.08
    voltage, _ = grid.solve(point)
    bus, gen = grid.bus.copy(), grid.gen.copy()
    bus[:, 2:4] = np.column_stack([point.pd, point.qd])
    gen[:, 1], gen[:, 5] = point.pg, grid.vm_set[grid.gb]
    net = from_ppc(dict(baseMVA=grid.base, bus=bus, gen=gen, branch=grid.branch.copy()),
                   f_hz=60, validate_conversion=False)
    pp.runpp(net, algorithm="nr", calculate_voltage_angles=True,
             enforce_q_lims=False, voltage_depend_loads=False, trafo_model="pi",
             tolerance_mva=1e-8, numba=False, init="flat")
    independent = net.res_bus.vm_pu.to_numpy() * np.exp(1j * np.deg2rad(net.res_bus.va_degree.to_numpy()))
    assert np.max(np.abs(voltage - independent)) < 1e-8


def test_two_generators_at_one_bus(tmp_path):
    original = Grid("case14_ieee")
    gen = original.gen.copy()
    index = np.flatnonzero(gen[:, 0] == original.pv[0])[0]
    duplicate = gen[index].copy()
    gen[index, 1] *= 0.6
    duplicate[1] *= 0.4
    gen[index, 8] *= 0.6
    duplicate[8] *= 0.4
    gen = np.vstack([gen, duplicate])
    tables = dict(bus=original.bus, gen=gen, branch=original.branch,
                  gencost=np.vstack([original.cost, original.cost[index]]))
    text = "function mpc = multi\nmpc.version = '2';\nmpc.baseMVA = 100;\n"
    for name, table in tables.items():
        text += f"mpc.{name} = [\n" + "\n".join(" ".join(map(str, row)) + ";" for row in table) + "\n];\n"
    path = tmp_path / "multi.m"
    path.write_text(text)
    grid = Grid("multi", path=path)
    point = grid.default_point()
    direction = np.zeros(grid.nb)
    direction[grid.pv[0]], direction[grid.pq[-1]] = 1, -1
    changed = grid.perturb(point, direction, 10.)
    assert np.allclose((changed.pg - point.pg)[[index, -1]], [6., 4.])
    voltage, _ = grid.solve(point)
    assert np.allclose(voltage, original.solve(original.default_point())[0], atol=1e-10)

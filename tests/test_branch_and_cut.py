from dataclasses import dataclass

import pytest

pytest.importorskip("gurobipy")

from uav_ugv_precedence import Instance, Params, branch_and_cut, make_instance  # noqa: E402

START, FINISH = (0.0, 0.0, 0.0), (4000.0, 4000.0, 0.0)


@pytest.mark.parametrize("n,m,seed", [(1, 1, 0), (2, 1, 1), (2, 2, 2), (3, 1, 3)])
def test_returns_a_feasible_plan_of_problem_3(n, m, seed):
    inst = make_instance(n, m, 0.5, seed)
    res = branch_and_cut.solve(inst, time_limit=60)
    assert res.violations == []
    assert res.makespan <= res.info["warm_start_makespan"] + 1e-6
    # the repriced plan is the solver's objective, up to Gurobi's tolerances
    assert res.makespan == pytest.approx(res.info["objective"], rel=1e-3)


def test_respects_margins():
    inst = make_instance(2, 2, 0.5, seed=4)
    params = Params(delta_a=100.0, delta_g=100.0, delta_o=300.0)
    res = branch_and_cut.solve(inst, params, time_limit=60)
    assert res.violations == []
    t = res.visit_times
    assert all(t[q] - t[p] >= 300.0 - 1e-3 for p, q in inst.edges)


def test_plan_is_within_the_optimality_gap_of_a_known_optimum():
    # A point on the diagonal can be flown while the UGV drives straight to
    # the finish, so the optimum is exactly the UGV's drive time. With
    # MipGap = 5% the solver may stop at any plan within 5% of it.
    inst = Instance([(2000.0, 2000.0, 100.0)], [], [START], [FINISH])
    res = branch_and_cut.solve(inst, time_limit=60)
    drive = Params().theta_g(START, FINISH)
    assert drive - 1e-3 <= res.makespan <= drive / 0.95


@pytest.mark.parametrize("delta_o", [0.0, 300.0, 700.0])
def test_warm_start_meets_every_precedence_margin(delta_o):
    params = Params(delta_o=delta_o)
    for n in (2, 3, 4, 6):
        for m in (1, 2, 3):
            for seed in range(6):
                inst = make_instance(n, m, 1.0, seed)
                ws = branch_and_cut._warm_start(inst, params)
                visit = {k: ws["sched"][mu][1][i] for k, (mu, i) in ws["where"].items()}
                assert all(visit[q] - visit[p] >= delta_o - 1e-9 for p, q in inst.edges)


def test_solves_when_the_wait_hides_under_a_recharge():
    # the second point must wait for the first longer than the recharge lasts
    inst = Instance([(0, 0, 100), (100, 0, 100)], [(0, 1)], [START], [START])
    res = branch_and_cut.solve(inst, Params(delta_o=700.0), time_limit=60)
    assert res.violations == []


def test_refuses_a_model_it_cannot_encode():
    @dataclass(frozen=True)
    class Swap(Params):
        def tau_c(self, airborne):
            return 120.0 if airborne > 0 else 0.0

    with pytest.raises(NotImplementedError):
        branch_and_cut.solve(make_instance(2, 1, 0.5, 0), Swap())

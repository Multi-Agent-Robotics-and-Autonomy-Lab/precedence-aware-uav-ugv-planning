import copy

import numpy as np
import pytest

from uav_ugv_precedence import Instance, Params, Plan, make_instance, problem

PAR = Params()
START, FINISH = (0.0, 0.0, 0.0), (4000.0, 4000.0, 0.0)


def two_point_plan(one_tour=True):
    """A hand-built plan for two points and one team, either one tour visiting
    both or one tour per point."""
    q0, q1 = (1000.0, 1000.0, 100.0), (1300.0, 1400.0, 100.0)
    inst = Instance([q0, q1], [(0, 1)], [START], [FINISH])
    g0, g1 = PAR.pi(q0), PAR.pi(q1)
    th_a, th_g = PAR.theta_a, PAR.theta_g
    X = np.zeros((1, 2, 4, 3))
    Ta = np.zeros((1, 2, 3))
    Tg, Tga = np.zeros((1, 2)), np.zeros((1, 1))
    if one_tour:
        X[0, 0] = [g0, q0, q1, g1]
        X[0, 1] = [g1] * 4
        Ta[0, 0] = [th_a(g0, q0), th_a(q0, q1), th_a(q1, g1)]
        Tg[0, 0] = th_g(g0, g1)
        theta_f = th_g(g1, FINISH)
    else:
        X[0, 0] = [g0, q0, q0, g0]
        X[0, 1] = [g1, q1, q1, g1]
        Ta[0, 0] = [th_a(g0, q0), 0.0, th_a(q0, g0)]
        Ta[0, 1] = [th_a(g1, q1), 0.0, th_a(q1, g1)]
        Tga[0, 0] = max(th_g(g0, g1), PAR.tau_c(Ta[0, 0].sum()))
        theta_f = th_g(g1, FINISH)
    plan = Plan(X, Ta, Tg, Tga, np.array([th_g(START, g0)]),
                np.array([theta_f]))
    return inst, plan


# ============================ INSTANCES ======================================
def test_make_instance_is_reproducible_and_seeded():
    a, b = make_instance(30, 3, 0.5, seed=4), make_instance(30, 3, 0.5, seed=4)
    assert a == b
    assert make_instance(30, 3, 0.5, seed=5).points != a.points


def test_make_instance_follows_table_I():
    inst = make_instance(50, 4, 0.5, seed=1)
    pts = np.array(inst.points)
    assert inst.n == 50 and inst.m == 4
    assert np.all((pts[:, :2] >= 0) & (pts[:, :2] <= 4000))
    assert np.all(pts[:, 2] == 100.0)
    assert inst.starts == (START,) * 4 and inst.finishes == (FINISH,) * 4


@pytest.mark.parametrize("n,d", [(25, 0.3), (40, 0.5), (75, 0.5), (40, 3.0)])
def test_precedence_graph_has_round_dn_distinct_acyclic_edges(n, d):
    inst = make_instance(n, 1, d, seed=2)
    assert len(inst.edges) == round(d * n)
    assert len(set(inst.edges)) == len(inst.edges)
    assert len(problem.topological_order(n, inst.edges)) == n


def test_precedence_graph_saturates_at_the_complete_dag():
    assert len(make_instance(10, 1, 20.0, seed=0).edges) == 45


def test_points_do_not_depend_on_density():
    assert (make_instance(30, 2, 0.5, seed=3).points
            == make_instance(30, 2, 8.0, seed=3).points)


@pytest.mark.parametrize("kwargs", [
    dict(edges=[(0, 1), (1, 0)]),                          # cycle
    dict(edges=[(0, 5)]),                                  # bad index
    dict(points=[(10, 10, 0), (20, 20, 100)]),             # point on the ground
    dict(points=[(10, 10, 100), (10, 10, 100)]),           # duplicate points
    dict(starts=[(0, 0, 50)]),                             # start in the air
    dict(finishes=[FINISH, FINISH]),                       # one finish too many
    dict(points=[(10, 10, 100), (5000, 20, 100)]),         # outside the map
])
def test_invalid_instances_are_rejected(kwargs):
    args = dict(points=[(10, 10, 100), (20, 20, 100)], edges=[(0, 1)],
                starts=[START], finishes=[FINISH])
    args.update(kwargs)
    with pytest.raises(ValueError):
        Instance(**args)


# ============================ MODEL ==========================================
def test_travel_times_follow_table_I():
    p = (300.0, 400.0, 100.0)
    assert PAR.pi(p) == (300.0, 400.0, 0.0)
    assert PAR.theta_a(PAR.pi(p), p) == pytest.approx(50.0)     # 100 m at 2 m/s
    assert PAR.theta_a((0.0, 0.0, 100.0), p) == pytest.approx(50.0)  # 500 m at 10 m/s
    assert PAR.theta_g(START, PAR.pi(p)) == pytest.approx(200.0)    # 500 m at 2.5 m/s
    assert PAR.tau_c(123.0) == 123.0


def test_params_reject_nonsense():
    with pytest.raises(ValueError):
        Params(s_g=0.0)
    with pytest.raises(ValueError):
        Params(delta_o=-1.0)


def test_eq2_counts_the_recharge_after_a_last_tour_followed_by_trivial_rows():
    inst, plan = two_point_plan(one_tour=True)
    b = max(plan.Theta_a[0, 0].sum(), plan.Theta_g[0, 0])
    expected = plan.theta_s[0] + b + PAR.tau_c(b) + plan.theta_f[0]
    assert problem.team_times(plan, PAR)[0] == pytest.approx(expected)
    assert problem.check_constraints(inst, PAR, plan) == []


def test_eq2_has_no_recharge_after_the_nth_tour():
    inst, plan = two_point_plan(one_tour=False)
    b0, b1 = (plan.Theta_a[0, i].sum() for i in (0, 1))
    expected = (plan.theta_s[0] + b0 + plan.Theta_ga[0, 0] + b1
                + plan.theta_f[0])
    assert problem.team_times(plan, PAR)[0] == pytest.approx(expected)
    assert problem.check_constraints(inst, PAR, plan) == []


def test_visit_times_accumulate_legs_and_gaps():
    inst, plan = two_point_plan(one_tour=False)
    t = problem.visit_times(inst, PAR, plan)
    first = plan.theta_s[0] + plan.Theta_a[0, 0, 0]
    second = (plan.theta_s[0] + plan.Theta_a[0, 0].sum() + plan.Theta_ga[0, 0]
              + plan.Theta_a[0, 1, 0])
    assert t == pytest.approx([first, second])


# ============================ CONSTRAINT CHECKS ==============================
def _broken(change):
    inst, plan = two_point_plan(one_tour=True)
    plan = copy.deepcopy(plan)
    params = change(inst, plan) or PAR
    return problem.check_constraints(inst, params, plan)


@pytest.mark.parametrize("tag,change", [
    ("(3b)", lambda i, p: p.X[0, 0].__setitem__(2, p.X[0, 0, 1])),
    ("(3c)", lambda i, p: Params(delta_a=460.0)),
    ("(3d)", lambda i, p: Params(delta_g=500.0)),
    ("(3e)", lambda i, p: p.X[0, 0, 3].__setitem__(2, 50.0)),
    ("(3f)", lambda i, p: p.X[0, 0].__setitem__(2, (2000.0, 2000.0, 100.0))),
    ("(3g)", lambda i, p: p.Theta_a[0, 0].__setitem__(1, 1.0)),
    ("(3h)", lambda i, p: p.Theta_g[0].__setitem__(0, 1.0)),
    ("(3i)", lambda i, p: p.X[0, 1].__setitem__(slice(None), (0.0, 4000.0, 0.0))),
    ("(3j)", lambda i, p: p.theta_s.__setitem__(0, 1.0)),
    ("(3k)", lambda i, p: Params(delta_o=1000.0)),
])
def test_each_constraint_check_catches_its_violation(tag, change):
    errs = _broken(change)
    assert any(e.startswith(tag) for e in errs), errs


def test_wrong_shapes_are_reported():
    inst, plan = two_point_plan()
    plan.Theta_g = plan.Theta_g[:, :1]
    assert "Theta_g has shape" in problem.check_constraints(inst, PAR, plan)[0]


def test_assumption_1():
    inst = make_instance(10, 1, 0.5, seed=0)
    assert problem.assumption_1_violations(inst, PAR) == []
    assert len(problem.assumption_1_violations(inst, Params(tau_a_max=90.0))) == 10
    assert len(problem.assumption_1_violations(inst, Params(delta_g=700.0))) == 10

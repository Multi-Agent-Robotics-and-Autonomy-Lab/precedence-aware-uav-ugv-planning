import numpy as np
import pytest

from uav_ugv_precedence import Instance, Params, alg1, make_instance, problem

PAR = Params()
START, FINISH = (0.0, 0.0, 0.0), (4000.0, 4000.0, 0.0)
MARGINS = Params(delta_a=120.0, delta_g=120.0, delta_o=300.0)


def visits_by_team(instance, result):
    return [[v for tour in team for v in tour.visits]
            for team in problem.tours(instance, result.plan)]


@pytest.mark.parametrize("n,m,d,seed", [
    (1, 1, 0.5, 0), (2, 3, 0.5, 1), (5, 2, 1.0, 2), (25, 1, 0.5, 3),
    (25, 4, 0.3, 6), (40, 3, 3.0, 4), (60, 10, 0.5, 5), (30, 2, 12.0, 7),
])
@pytest.mark.parametrize("params", [PAR, MARGINS], ids=["no-margins", "margins"])
def test_returns_a_feasible_plan_visiting_every_point_once(n, m, d, seed, params):
    inst = make_instance(n, m, d, seed)
    res = alg1.solve(inst, params)
    assert res.violations == []
    assert sorted(v for team in visits_by_team(inst, res) for v in team) \
        == list(range(n))
    assert res.makespan == pytest.approx(problem.makespan(res.plan, params))


def test_bookkept_visit_times_match_the_plan():
    inst = make_instance(60, 4, 1.0, seed=8)
    res = alg1.solve(inst, MARGINS)
    assert np.allclose(res.info["visit_times"], res.visit_times, atol=1e-9)


def test_ordering_margin_separates_every_edge():
    inst = make_instance(40, 3, 2.0, seed=9)
    res = alg1.solve(inst, Params(delta_o=400.0))
    t = res.visit_times
    assert min(t[q] - t[p] for p, q in inst.edges) >= 400.0 - 1e-9


def test_single_point_by_hand():
    q = (1000.0, 1000.0, 100.0)
    inst = Instance([q], [], [START], [FINISH])
    res = alg1.solve(inst, PAR)
    g = PAR.pi(q)
    expected = PAR.theta_g(START, g) + 100.0 + PAR.theta_g(g, FINISH)
    assert res.makespan == pytest.approx(expected)
    assert res.visit_times[0] == pytest.approx(PAR.theta_g(START, g) + 50.0)


def test_two_points_in_one_tour_by_hand():
    q0, q1 = (1000.0, 1000.0, 100.0), (1300.0, 1400.0, 100.0)
    inst = Instance([q0, q1], [], [START], [FINISH])
    res = alg1.solve(inst, PAR)
    g0, g1 = PAR.pi(q0), PAR.pi(q1)
    # q0 is reached first; q1 joins its tour; line 50 lands below q1, which
    # is nearer the finish
    assert [t.visits for t in problem.tours(inst, res.plan)[0]] == [[0, 1]]
    assert tuple(res.plan.X[0, 0, -1]) == g1
    b = max(PAR.theta_a(g0, q0) + PAR.theta_a(q0, q1) + PAR.theta_a(q1, g1),
            PAR.theta_g(g0, g1))
    # literal Eq. (2): the trivial second row puts a recharge before theta_f
    expected = (PAR.theta_g(START, g0) + b + PAR.tau_c(b)
                + PAR.theta_g(g1, FINISH))
    assert res.makespan == pytest.approx(expected)


def test_rows_are_padded_as_in_lines_36_and_54():
    inst = make_instance(12, 3, 0.5, seed=2)
    res = alg1.solve(inst, PAR)
    X, n = res.plan.X, inst.n
    for team in problem.tours(inst, res.plan):
        for tour in team:
            # line 36/51: unused columns repeat the last visited point
            assert tuple(X[tour.team, tour.row, n]) == inst.points[tour.visits[-1]]
        if team:
            mu, k = team[0].team, team[-1].row
            # line 54: rows after the last tour are trivial, at its collect point
            assert np.all(X[mu, k + 1:] == X[mu, k, -1])
            assert np.all(res.plan.Theta_ga[mu, k:] == 0)


def test_idle_teams_drive_straight_to_their_finish_and_ties_go_to_team_0():
    inst = Instance([(1000.0, 1000.0, 100.0)], [], [START] * 3, [FINISH] * 3)
    res = alg1.solve(inst, PAR)
    assert [len(t) for t in problem.tours(inst, res.plan)] == [1, 0, 0]
    for mu in (1, 2):
        assert np.all(res.plan.X[mu] == FINISH)
        assert res.team_times[mu] == pytest.approx(PAR.theta_g(START, FINISH))


def test_a_point_can_wait_for_a_predecessor_on_another_team():
    # q1 is next to the start but must follow q0, far away; with two teams the
    # second team takes q1 and waits for q0's visit
    q0, q1 = (3500.0, 500.0, 100.0), (300.0, 300.0, 100.0)
    inst = Instance([q0, q1], [(0, 1)], [START] * 2, [FINISH] * 2)
    res = alg1.solve(inst, Params(delta_o=200.0))
    assert res.violations == []
    assert res.visit_times[1] >= res.visit_times[0] + 200.0 - 1e-9


def test_is_deterministic():
    inst = make_instance(40, 4, 1.0, seed=10)
    a, b = alg1.solve(inst, PAR), alg1.solve(inst, PAR)
    assert np.array_equal(a.plan.X, b.plan.X)
    assert np.array_equal(a.plan.Theta_a, b.plan.Theta_a)


def test_rejects_instances_that_violate_assumption_1():
    inst = make_instance(5, 1, 0.5, seed=0)
    with pytest.raises(ValueError, match="Assumption 1"):
        alg1.solve(inst, Params(tau_a_max=99.0))

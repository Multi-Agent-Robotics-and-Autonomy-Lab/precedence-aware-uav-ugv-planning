import pytest

from uav_ugv_precedence import Instance, Params, decoupled, make_instance, problem

PAR = Params()
START, FINISH = (0.0, 0.0, 0.0), (4000.0, 4000.0, 0.0)


def team_of(instance, result):
    return {v: tour.team for team in problem.tours(instance, result.plan)
            for tour in team for v in tour.visits}


@pytest.mark.parametrize("n,m,d,seed", [
    (1, 1, 0.5, 0), (8, 3, 0.5, 1), (25, 4, 0.5, 2), (40, 2, 3.0, 3),
    (60, 10, 0.3, 4),
])
@pytest.mark.parametrize("params", [PAR, Params(delta_o=250.0, delta_a=60.0)])
def test_returns_a_feasible_plan(n, m, d, seed, params):
    inst = make_instance(n, m, d, seed)
    res = decoupled.solve(inst, params)
    assert res.violations == []
    assert sorted(team_of(inst, res)) == list(range(n))


def test_every_component_stays_on_one_team():
    inst = make_instance(50, 4, 0.8, seed=5)
    res = decoupled.solve(inst, PAR)
    owner = team_of(inst, res)
    for comp in decoupled.components(inst):
        assert len({owner[v] for v in comp}) == 1


def test_largest_workload_goes_first_to_team_0():
    inst = make_instance(30, 3, 0.5, seed=6)
    res = decoupled.solve(inst, PAR)
    comps = decoupled.components(inst)
    work = [decoupled.walk_time(inst, PAR, c) for c in comps]
    biggest = comps[max(range(len(comps)), key=lambda k: (work[k], -comps[k][0]))]
    assert team_of(inst, res)[biggest[0]] == 0


def test_a_singleton_costs_one_take_off_and_landing():
    inst = make_instance(5, 1, 0.0, seed=0)
    assert decoupled.walk_time(inst, PAR, [3]) == pytest.approx(100.0)


def test_walk_is_nearest_neighbour_and_respects_precedence():
    pts = [(100.0, 0.0, 100.0), (200.0, 0.0, 100.0), (300.0, 0.0, 100.0)]
    inst = Instance(pts, [(2, 0)], [START], [FINISH])
    # 0 is nearest to the start but must wait for 2
    assert decoupled.nearest_neighbour_walk(inst, [0, 1, 2], START) == [1, 2, 0]


def test_one_team_follows_the_walk_from_its_start():
    inst = make_instance(15, 1, 0.0, seed=7)
    res = decoupled.solve(inst, PAR)
    order = [v for tour in problem.tours(inst, res.plan)[0] for v in tour.visits]
    assert order == decoupled.nearest_neighbour_walk(inst, range(15), START)

import math
import random

import pytest

from uav_ugv_precedence import Instance, Params, annealing, make_instance, problem
from uav_ugv_precedence.annealing import MOVES, _Context, _copy, warm_start

MARGINS = Params(delta_a=60.0, delta_g=60.0, delta_o=200.0)


@pytest.mark.parametrize("n,m,seed", [(1, 1, 0), (6, 2, 1), (10, 3, 2), (12, 1, 3)])
@pytest.mark.parametrize("params", [Params(), MARGINS], ids=["no-margins", "margins"])
def test_warm_start_is_feasible(n, m, seed, params):
    inst = make_instance(n, m, 1.0, seed)
    ctx = _Context(inst, params)
    state = warm_start(inst, params, random.Random(seed))
    energy, makespan, penalty = ctx.energy(state)
    assert penalty == 0.0 and energy == makespan
    assert problem.check_constraints(inst, params, ctx.plan(state)) == []


@pytest.mark.parametrize("delta_o", [205.0, 700.0])
def test_warm_start_survives_waits_hidden_under_a_recharge(delta_o):
    params = Params(delta_o=delta_o)
    inst = Instance([(0, 0, 100), (100, 0, 100)], [(0, 1)], [(0, 0, 0)], [(0, 0, 0)])
    assert _Context(inst, params).energy(warm_start(inst, params, random.Random(0)))[2] == 0.0
    for n in (3, 6, 10):
        for m in (1, 2, 3):
            for seed in range(8):
                inst = make_instance(n, m, 1.0, seed)
                state = warm_start(inst, params, random.Random(seed))
                assert _Context(inst, params).energy(state)[2] == 0.0


@pytest.mark.parametrize("params", [Params(), MARGINS], ids=["no-margins", "margins"])
def test_energy_prices_states_like_problem_py(params):
    inst = make_instance(8, 3, 1.0, seed=4)
    ctx, rng = _Context(inst, params), random.Random(5)
    state = warm_start(inst, params, rng)
    for _ in range(400):
        move = MOVES[rng.randrange(len(MOVES))][0]
        cand = _copy(state)
        if move(cand, rng, ctx):
            state = cand
        _, makespan, penalty = ctx.energy(state)
        plan = ctx.plan(state)
        assert makespan == pytest.approx(problem.makespan(plan, params), rel=1e-9)
        assert (penalty == 0.0) == (problem.check_constraints(inst, params, plan) == [])


def test_returns_a_feasible_plan_no_worse_than_its_start():
    inst = make_instance(8, 2, 0.5, seed=6)
    res = annealing.solve(inst, MARGINS, steps=3000, seed=1)
    assert res.violations == []
    assert res.makespan <= res.info["initial_makespan"] + 1e-9


def test_is_deterministic_for_a_seed():
    inst = make_instance(6, 2, 0.5, seed=7)
    a = annealing.solve(inst, steps=1500, seed=3)
    b = annealing.solve(inst, steps=1500, seed=3)
    assert a.makespan == b.makespan
    assert (a.plan.X == b.plan.X).all()


def test_cooling_is_logarithmic_and_floored():
    inst = make_instance(3, 1, 0.5, seed=8)
    res = annealing.solve(inst, steps=1000, t_max=500.0, t_min=0.1)
    assert res.info["final_temperature"] == pytest.approx(
        max(0.1, 500.0 * math.log(2) / math.log(1001)))
    res = annealing.solve(inst, steps=10, t_max=0.2, t_min=0.1)
    assert res.info["final_temperature"] == 0.1

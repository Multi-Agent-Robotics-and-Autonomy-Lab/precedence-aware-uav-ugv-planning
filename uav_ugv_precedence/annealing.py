"""Simulated annealing baseline (Section IV-E).

A local search directly over the plan (X, Theta) of Section II: every team's
n x (n+2) matrix X^mu plus the slack of every duration above its fastest travel
time, so (3e) and (3g)-(3j) hold by construction. The remaining constraints are
soft. The energy is the makespan of Eq. (2) plus a fixed charge for every
violation of (3b), (3c), (3d), (3f) or (3k) and a rate per second of violation.

Each step makes one move: rewrite a waypoint slot, move a tour's release and
collect points, exchange slots between two teams, perturb one slack, or zero a
tour's slacks. Moves are accepted by the Metropolis rule at a temperature that
follows a logarithmic schedule, T = t_max ln 2 / ln(1 + k), never below t_min
(over any practical number of steps it stays far above t_min).
The search starts from a feasible plan: a random topological order dealt
round-robin to the teams, cut greedily into tours, with ground waits added
until every precedence edge holds. The best feasible plan visited is returned.

The cost of a step grows as m n^2, so this is meant for small instances.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .problem import (Instance, Params, Plan, Result, assumption_1_violations,
                      make_result)

PENALTY_FIXED = 2000.0    # charge per violated constraint [s]
PENALTY_RATE = 5.0        # charge per second of violation
SLACK_SIGMA = 100.0       # step of a slack perturbation [s]
TOL = 1e-9


def solve(instance: Instance, params: Optional[Params] = None,
          steps: int = 100_000, t_max: float = 500.0, t_min: float = 0.1,
          seed: int = 0) -> Result:
    """Plan the mission with simulated annealing.

    A step costs O(m n^2), so keep `steps` and the instance small.
    """
    params = params or Params()
    bad = assumption_1_violations(instance, params)
    if bad:
        raise ValueError(f"Assumption 1 fails for points {bad}: they have no "
                         f"feasible single-point tour")
    start = time.perf_counter()
    rng = random.Random(seed)
    ctx = _Context(instance, params)
    current = warm_start(instance, params, rng)
    cur = ctx.energy(current)
    initial_makespan = cur[1]
    best, best_e = _copy(current), cur
    total = sum(w for _, w in MOVES)
    accepted = 0
    temperature = t_max
    for step in range(1, steps + 1):
        temperature = max(t_min, t_max * math.log(2) / math.log(1 + step))
        cand = _copy(current)
        if not _pick(rng, total)(cand, rng, ctx):
            continue
        e = ctx.energy(cand)
        delta = e[0] - cur[0]
        if delta <= 0.0 or math.exp(-delta / max(temperature, 1e-12)) > rng.random():
            current, cur = cand, e
            accepted += 1
            if e[2] == 0.0 and e[0] < best_e[0]:
                best, best_e = _copy(current), e
    plan = ctx.plan(best)
    runtime = time.perf_counter() - start
    info = {"steps": steps, "t_max": t_max, "t_min": t_min, "seed": seed,
            "final_temperature": temperature,
            "accept_rate": accepted / max(1, steps),
            "initial_makespan": initial_makespan}
    return make_result("annealing", instance, params, plan, runtime, info)


# ============================ STATE ==========================================
@dataclass
class _Team:
    """X^mu and the slacks above the fastest times. ga[0] is on theta_s and
    ga[i] on the transport into row i; gs is on theta_g, fs on theta_f."""

    X: List[list]
    air: List[List[float]]
    ga: List[float]
    gs: List[float]
    fs: float


def _copy(state):
    return [_Team([list(r) for r in t.X], [list(r) for r in t.air],
                  list(t.ga), list(t.gs), t.fs) for t in state]


class _Context:
    """What the energy and the moves need besides the state."""

    def __init__(self, instance: Instance, params: Params):
        self.instance, self.params = instance, params
        self.n, self.m = instance.n, instance.m
        self.points = instance.points
        self.index = {p: k for k, p in enumerate(instance.points)}

    def energy(self, state):
        """(energy, makespan, penalty) of a state, in one pass over it."""
        par, n, index = self.params, self.n, self.index
        th_a, th_g, tau_c = par.theta_a, par.theta_g, par.tau_c
        t = [None] * n
        makespan, count, amount = 0.0, 0, 0.0
        for mu, team in enumerate(state):
            X = team.X
            release = th_g(self.instance.starts[mu], X[0][0]) + team.ga[0]
            for i in range(n):
                row, slack = X[i], team.air[i]
                arrival = release
                tau_a = 0.0
                for j in range(n + 1):
                    leg = th_a(row[j], row[j + 1]) + slack[j]
                    tau_a += leg
                    arrival += leg
                    if j < n:
                        k = index.get(row[j + 1])
                        if k is not None:
                            if t[k] is None or arrival < t[k]:
                                t[k] = arrival
                        elif row[j + 1] != row[j]:
                            count += 1                                    # (3f)
                tau_g = th_g(row[0], row[n + 1]) + team.gs[i]
                for over in (tau_a + par.delta_a - par.tau_a_max,         # (3c)
                             tau_g + par.delta_g - par.tau_a_max):        # (3d)
                    if over > TOL:
                        count += 1
                        amount += over
                airborne = max(tau_a, tau_g)
                if i < n - 1:
                    release += airborne + max(
                        th_g(row[n + 1], X[i + 1][0]) + team.ga[i + 1],
                        tau_c(airborne))
                else:
                    finish = (release + airborne + team.fs
                              + th_g(row[n + 1], self.instance.finishes[mu]))
            makespan = max(makespan, finish)
        count += sum(v is None for v in t)                                # (3b)
        for p, q in self.instance.edges:                                  # (3k)
            if t[p] is not None and t[q] is not None:
                short = par.delta_o - (t[q] - t[p])
                if short > TOL:
                    count += 1
                    amount += short
        penalty = PENALTY_FIXED * count + PENALTY_RATE * amount
        return makespan + penalty, makespan, penalty

    def plan(self, state) -> Plan:
        par, n = self.params, self.n
        Ta, Tg, Tga, ts, tf = [], [], [], [], []
        for mu, team in enumerate(state):
            X = team.X
            Ta.append([[par.theta_a(X[i][j], X[i][j + 1]) + team.air[i][j]
                        for j in range(n + 1)] for i in range(n)])
            Tg.append([par.theta_g(X[i][0], X[i][n + 1]) + team.gs[i]
                       for i in range(n)])
            Tga.append([par.theta_g(X[i][n + 1], X[i + 1][0]) + team.ga[i + 1]
                        for i in range(n - 1)])
            ts.append(par.theta_g(self.instance.starts[mu], X[0][0]) + team.ga[0])
            tf.append(par.theta_g(X[n - 1][n + 1], self.instance.finishes[mu])
                      + team.fs)
        m = len(state)
        return Plan(np.array([t.X for t in state], dtype=float), np.array(Ta),
                    np.array(Tg), np.array(Tga, dtype=float).reshape(m, n - 1),
                    np.array(ts), np.array(tf))


# ============================ MOVES ==========================================
def _move_slot(state, rng, ctx):
    """A waypoint slot becomes a random monitoring point or a repeat."""
    team = state[rng.randrange(len(state))]
    i, s = rng.randrange(ctx.n), rng.randrange(1, ctx.n + 1)
    if rng.random() < 0.5:
        team.X[i][s] = ctx.points[rng.randrange(ctx.n)]
    else:
        team.X[i][s] = team.X[i][s - 1]
    return True


def _move_ground(state, rng, ctx):
    """A tour's release and collect points jump anywhere on the ground."""
    team = state[rng.randrange(len(state))]
    i = rng.randrange(ctx.n)
    w, h = ctx.instance.size
    for col in (0, ctx.n + 1):
        x, y, _ = team.X[i][col]
        team.X[i][col] = (min(max(x + rng.uniform(-w, w), 0.0), w),
                          min(max(y + rng.uniform(-h, h), 0.0), h), 0.0)
    return True


def _move_cross(state, rng, ctx):
    """Move or swap a waypoint slot between two teams."""
    if len(state) < 2:
        return False
    a, b = rng.sample(range(len(state)), 2)
    ia, sa = rng.randrange(ctx.n), rng.randrange(1, ctx.n + 1)
    ib, sb = rng.randrange(ctx.n), rng.randrange(1, ctx.n + 1)
    A, B = state[a].X, state[b].X
    if rng.random() < 0.5:
        B[ib][sb] = A[ia][sa]
        A[ia][sa] = A[ia][sa - 1]
    else:
        A[ia][sa], B[ib][sb] = B[ib][sb], A[ia][sa]
    return True


def _move_pace(state, rng, ctx):
    """Perturb one slack: a transport, a flight leg, a UGV relocation or the
    final transport."""
    team = state[rng.randrange(len(state))]
    i, r = rng.randrange(ctx.n), rng.random()
    if r < 0.4:
        team.ga[i] = max(0.0, team.ga[i] + rng.gauss(0.0, SLACK_SIGMA))
    elif r < 0.7:
        s = rng.randrange(ctx.n + 1)
        team.air[i][s] = max(0.0, team.air[i][s] + rng.gauss(0.0, SLACK_SIGMA))
    elif r < 0.9:
        team.gs[i] = max(0.0, team.gs[i] + rng.gauss(0.0, SLACK_SIGMA))
    else:
        team.fs = max(0.0, team.fs + rng.gauss(0.0, SLACK_SIGMA))
    return True


def _move_zero(state, rng, ctx):
    """Remove every slack of one tour."""
    team = state[rng.randrange(len(state))]
    i = rng.randrange(ctx.n)
    if team.ga[i] <= TOL and team.gs[i] <= TOL and max(team.air[i]) <= TOL:
        return False
    team.ga[i] = team.gs[i] = 0.0
    team.air[i] = [0.0] * (ctx.n + 1)
    return True


MOVES = [(_move_slot, 10.0), (_move_ground, 6.0), (_move_cross, 5.0),
         (_move_pace, 5.0), (_move_zero, 2.0)]


def _pick(rng, total):
    x, c = rng.random() * total, 0.0
    for move, weight in MOVES:
        c += weight
        if x <= c:
            return move
    return MOVES[-1][0]


# ============================ WARM START =====================================
@dataclass
class _Tour:
    way: List[int]        # points in visit order
    col: int = 0          # the collect point is below way[col]
    ga: float = 0.0       # wait added to the transport into this tour


def warm_start(instance: Instance, params: Params, rng: random.Random):
    """A feasible starting state. A random topological order is dealt
    round-robin to the teams; each team's points are cut greedily into tours
    released below their first point. With delta_o > 0, tours holding two
    points joined by a precedence path are split first. Then, in topological
    order, a point whose predecessor shares its tour is split off into a new
    tour, and any remaining shortfall is waited out on the ground before its
    tour. If a violation survives, every tour is split into single-point tours
    and the waits are recomputed, which always succeeds. Rows after a team's
    last tour are trivial, at its finish."""
    n, m, P = instance.n, instance.m, instance.points
    preds = instance.predecessors()
    order = _random_topological_order(instance, rng)
    tours = [[] for _ in range(m)]
    for k, q in enumerate(order):
        tours[k % m].append(q)
    tours = [_cut(instance, params, seq) for seq in tours]
    if params.delta_o > TOL:
        _split_related(instance, params, tours)
    _repair(instance, params, tours, order, preds)
    if not _feasible_tours(instance, params, tours):
        # Ground waits move a tour rigidly, so interleaved tours can be
        # impossible to separate, and a split half can exceed the endurance.
        # One point per tour is always feasible: Assumption 1 covers energy,
        # and the waits then form an acyclic system.
        tours = [[_Tour([q]) for tour in team for q in tour.way] for team in tours]
        _repair(instance, params, tours, order, preds)

    state = []
    for mu, team in enumerate(tours):
        X = [[instance.finishes[mu]] * (n + 2) for _ in range(n)]
        ga = [0.0] * n
        for i, tour in enumerate(team):
            row = [params.pi(P[tour.way[0]])] + [P[w] for w in tour.way]
            row += [row[-1]] * (n + 1 - len(row))
            X[i] = row + [params.pi(P[tour.way[tour.col]])]
            ga[i] = tour.ga
        state.append(_Team(X, [[0.0] * (n + 1) for _ in range(n)], ga,
                           [0.0] * n, 0.0))
    return state


def _repair(instance, params, tours, order, preds):
    """Up to eight passes in topological order: split a point off from a
    predecessor that shares its tour, then wait on the ground before its tour
    until every predecessor is delta_o earlier."""
    for _ in range(8):
        t = _times(instance, params, tours)
        if all(t[q] - t[p] >= params.delta_o - TOL for p, q in instance.edges):
            return
        changed = False
        for q in order:
            if not preds[q]:
                continue
            where = _locate(tours)
            mu, i, k = where[q]
            if k > 0 and any(where[p][:2] == (mu, i) for p in preds[q]):
                _split_before(instance, params, tours[mu], i, k)
                t = _times(instance, params, tours)
                mu, i, _ = _locate(tours)[q]
                changed = True
            need = max(t[p] + params.delta_o for p in preds[q])
            if need > t[q] + TOL:
                _delay(instance, params, tours[mu], i, need - t[q])
                t = _times(instance, params, tours)
                changed = True
        if not changed:
            return


def _delay(instance, params, team, i, late):
    """Wait on the ground so that tour i starts `late` seconds later. The gap
    into a tour is max(transport + wait, recharge), so the wait first uses up
    any slack under the recharge."""
    tour = team[i]
    if i > 0:
        prev, P = team[i - 1], instance.points
        drive = params.theta_g(params.pi(P[prev.way[prev.col]]),
                               params.pi(P[tour.way[0]]))
        recharge = params.tau_c(max(*_tour_flight(instance, params, prev.way,
                                                  prev.col)))
        tour.ga = max(tour.ga, recharge - drive)
    tour.ga += late


def _feasible_tours(instance, params, tours):
    t = _times(instance, params, tours)
    if any(t[q] - t[p] < params.delta_o - TOL for p, q in instance.edges):
        return False
    return all(_feasible(params, *_tour_flight(instance, params, tour.way, tour.col))
               for team in tours for tour in team)


def _random_topological_order(instance, rng):
    succ, waiting = instance.successors(), [len(p) for p in instance.predecessors()]
    ready = [k for k in range(instance.n) if waiting[k] == 0]
    order = []
    while ready:
        k = rng.randrange(len(ready))
        ready[k], ready[-1] = ready[-1], ready[k]
        v = ready.pop()
        order.append(v)
        for s in succ[v]:
            waiting[s] -= 1
            if waiting[s] == 0:
                ready.append(s)
    return order


def _tour_flight(instance, params, way, col):
    """(tau_a, tau_g) of a tour released below way[0], landing below way[col]."""
    P = instance.points
    release, collect = params.pi(P[way[0]]), params.pi(P[way[col]])
    tau_a = params.theta_a(release, P[way[0]])
    tau_a += sum(params.theta_a(P[a], P[b]) for a, b in zip(way, way[1:]))
    tau_a += params.theta_a(P[way[-1]], collect)
    return tau_a, params.theta_g(release, collect)


def _feasible(params, tau_a, tau_g):
    return (tau_a + params.delta_a <= params.tau_a_max
            and tau_g + params.delta_g <= params.tau_a_max)


def _cut(instance, params, seq):
    """Greedy maximal tours along `seq`: extend while some collect point below
    one of the tour's points keeps (3c) and (3d), landing at the one with the
    smallest airborne time."""
    tours, k = [], 0
    while k < len(seq):
        tour = _Tour([seq[k]])
        k += 1
        while k < len(seq):
            trial = tour.way + [seq[k]]
            best = None
            for c in range(len(trial)):
                tau_a, tau_g = _tour_flight(instance, params, trial, c)
                if _feasible(params, tau_a, tau_g) and (
                        best is None or max(tau_a, tau_g) < best[1]):
                    best = (c, max(tau_a, tau_g))
            if best is None:
                break
            tour.way, tour.col = trial, best[0]
            k += 1
        tours.append(tour)
    return tours


def _best_col(instance, params, way):
    """Feasible collect points first, then the smallest airborne time."""
    def key(c):
        tau_a, tau_g = _tour_flight(instance, params, way, c)
        return (not _feasible(params, tau_a, tau_g), max(tau_a, tau_g))
    return min(range(len(way)), key=key)


def _split_before(instance, params, team, i, k):
    way = team[i].way
    left = _Tour(way[:k], _best_col(instance, params, way[:k]), team[i].ga)
    right = _Tour(way[k:], _best_col(instance, params, way[k:]))
    team[i:i + 1] = [left, right]


def _split_related(instance, params, tours):
    """No tour may hold two points joined by a path in the precedence graph:
    a tour moves rigidly under ground waits, so their separation is fixed."""
    succ = instance.successors()
    reach = []
    for u in range(instance.n):
        seen, stack = set(), list(succ[u])
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                stack.extend(succ[v])
        reach.append(seen)
    changed = True
    while changed:
        changed = False
        for team in tours:
            for i, tour in enumerate(team):
                cut = next((k for k in range(1, len(tour.way))
                            if any(tour.way[k] in reach[u]
                                   for u in tour.way[:k])), None)
                if cut is not None:
                    _split_before(instance, params, team, i, cut)
                    changed = True
                    break
            if changed:
                break


def _locate(tours):
    return {w: (mu, i, k) for mu, team in enumerate(tours)
            for i, tour in enumerate(team) for k, w in enumerate(tour.way)}


def _times(instance, params, tours):
    """Visit times of a warm-start state, by Eq. (2)'s accumulation."""
    P, t = instance.points, [None] * instance.n
    for mu, team in enumerate(tours):
        if not team:
            continue
        clock = (params.theta_g(instance.starts[mu], params.pi(P[team[0].way[0]]))
                 + team[0].ga)
        for i, tour in enumerate(team):
            release = params.pi(P[tour.way[0]])
            collect = params.pi(P[tour.way[tour.col]])
            acc, tail = 0.0, release
            for w in tour.way:
                acc += params.theta_a(tail, P[w])
                t[w] = clock + acc
                tail = P[w]
            tau_a, tau_g = _tour_flight(instance, params, tour.way, tour.col)
            airborne = max(tau_a, tau_g)
            if i < len(team) - 1:
                nxt = team[i + 1]
                clock += airborne + max(
                    params.theta_g(collect, params.pi(P[nxt.way[0]])) + nxt.ga,
                    params.tau_c(airborne))
    return t

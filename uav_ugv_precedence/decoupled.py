"""Decoupled precedence baseline (Section IV-F).

Each weakly connected component of the precedence graph goes to a single
team, so every precedence edge stays inside one team and the teams can be
planned independently. A component's workload is the flight time of a
nearest-neighbour walk through it that respects precedence; components are
assigned by Longest Processing Time (largest workload first, to the team with
the smallest accumulated workload). Each team then visits its points in
nearest-neighbour order from its start, and the tours and their timing come
from Alg. 1's own subroutines.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence

import numpy as np

from .alg1 import Builder
from .problem import (Instance, Params, Point, Result,
                      assumption_1_violations, make_result)


def solve(instance: Instance, params: Optional[Params] = None) -> Result:
    """Plan the mission with the decoupled precedence baseline."""
    params = params or Params()
    bad = assumption_1_violations(instance, params)
    if bad:
        raise ValueError(f"Assumption 1 fails for points {bad}: they have no "
                         f"feasible single-point tour")
    start = time.perf_counter()
    comps = components(instance)
    workload = [walk_time(instance, params, comp) for comp in comps]

    load = [0.0] * instance.m
    teams: List[List[int]] = [[] for _ in range(instance.m)]
    for k in sorted(range(len(comps)), key=lambda k: (-workload[k], comps[k][0])):
        mu = min(range(instance.m), key=lambda nu: load[nu])
        teams[mu].extend(comps[k])
        load[mu] += workload[k]

    builder = Builder(instance, params)
    for mu, points in enumerate(teams):
        for q in nearest_neighbour_walk(instance, points, instance.starts[mu]):
            builder.commit(mu, q, builder.evaluate(mu, q))
    builder.close()
    plan = builder.plan()
    runtime = time.perf_counter() - start
    info = {"visit_times": np.array(builder.t, dtype=float),
            "candidates": builder.candidates,
            "components": comps, "teams": teams}
    return make_result("decoupled", instance, params, plan, runtime, info)


def components(instance: Instance) -> List[List[int]]:
    """Weakly connected components of the precedence graph, each sorted,
    ordered by their smallest point."""
    parent = list(range(instance.n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in instance.edges:
        parent[find(a)] = find(b)
    groups = {}
    for k in range(instance.n):
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values())


def nearest_neighbour_walk(instance: Instance, points: Sequence[int],
                           start: Point) -> List[int]:
    """Visit `points` from `start`, always moving to the nearest point whose
    predecessors among `points` have all been visited. Ties go to the lowest
    index."""
    P, preds, succ = instance.points, instance.predecessors(), instance.successors()
    left = set(points)
    waiting = {q: sum(p in left for p in preds[q]) for q in left}
    order, here = [], start
    while left:
        q = min((k for k in left if waiting[k] == 0),
                key=lambda k: (math.dist(here, P[k]), k))
        order.append(q)
        left.remove(q)
        here = P[q]
        for s in succ[q]:
            if s in waiting:
                waiting[s] -= 1
    return order


def walk_time(instance: Instance, params: Params, comp: Sequence[int]) -> float:
    """A component's workload: the UAV flight time of a nearest-neighbour
    walk through it that starts at its lowest-index source, including the
    take-off below the first point and the landing below the last. Transport
    to and from the team's start and finish, tour cuts and recharging are not
    counted."""
    P, preds = instance.points, instance.predecessors()
    inside = set(comp)
    first = min(k for k in comp if not any(p in inside for p in preds[k]))
    walk = [first] + nearest_neighbour_walk(
        instance, [k for k in comp if k != first], P[first])
    legs = sum(params.theta_a(P[a], P[b]) for a, b in zip(walk, walk[1:]))
    return (params.theta_a(params.pi(P[walk[0]]), P[walk[0]]) + legs
            + params.theta_a(P[walk[-1]], params.pi(P[walk[-1]])))

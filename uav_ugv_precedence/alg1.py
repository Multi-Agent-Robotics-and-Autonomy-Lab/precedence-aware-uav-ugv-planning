"""Alg. 1: Precedence-Aware Robust UAV-UGV Planning, the paper's method.

One monitoring point is placed per iteration, so the assignment, the visit
order, the tours and their timing are built together, and every visit is
planned against the already fixed visit times of its predecessors. Comments
give the line numbers of Alg. 1 in the paper.

`Builder` holds the state of line 1 and the subroutines that evaluate one
(team, point) pairing (lines 5-25), commit it (lines 30-45) and close every
team's plan (lines 48-57). The decoupled baseline reuses it.
"""

from __future__ import annotations

import bisect
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .problem import (Instance, Params, Plan, Point, Result,
                      assumption_1_violations, make_result)


@dataclass
class Option:
    """<t, r, c, sigma_a, sigma_g, C', Gamma'> of one pairing (line 26)."""

    t: float                  # visit time of the point
    r: Optional[Point]        # release point of a new tour; None to append
    c: Optional[Point]        # collect point of the tour being closed
    sigma_a: float            # airborne wait before reaching the point
    sigma_g: float            # ground wait before the new release
    C: List[Point]            # candidate collect points C'
    Gamma: Optional[float]    # release clock of the new tour


class Builder:
    """Alg. 1's state and its per-pairing subroutines.

    The lists carry an unused index 0, so X[mu][i][j], Theta_a[mu][i][j],
    Theta_g[mu][i] and Theta_ga[mu][i] use the paper's 1-based i and j.
    """

    def __init__(self, instance: Instance, params: Params):
        n, m = instance.n, instance.m
        self.instance, self.params = instance, params
        self.preds = instance.predecessors()
        # line 1
        self.X = [[[None] * (n + 3) for _ in range(n + 1)] for _ in range(m)]
        self.Theta_a = [[[0.0] * (n + 2) for _ in range(n + 1)]
                        for _ in range(m)]
        self.Theta_g = [[0.0] * (n + 1) for _ in range(m)]
        self.Theta_ga = [[0.0] * n for _ in range(m)]
        self.theta_s = [0.0] * m
        self.theta_f = [0.0] * m
        self.i = [0] * m          # active tour (row) of each team
        self.j = [2] * m          # next free column of that row
        self.Gamma = [0.0] * m    # time the active tour was released
        self.eta = [0.0] * m      # flight time accumulated in the active tour
        self.C = [[] for _ in range(m)]   # its candidate collect points
        self.T = [0.0] * m        # team cost: visit time of its last point
        self.t = [None] * n       # committed visit times t_q
        self.candidates = [[] for _ in range(m)]   # C_mu of every closed tour

    def evaluate(self, mu: int, q: int) -> Option:
        """Lines 5-25: the option of giving point q to team mu.

        The visit time t of lines 12 and 25 equals max(e, w) exactly and is
        computed that way, so options that all wait until w stay exactly tied
        and line 27's tie-break, not rounding, decides between them.
        """
        par = self.params
        th_a, th_g, tau_c = par.theta_a, par.theta_g, par.tau_c
        p = self.instance.points[q]
        X, i, j, eta = self.X[mu], self.i[mu], self.j[mu], self.eta[mu]
        w = max([0.0] + [self.t[k] + par.delta_o for k in self.preds[q]])  # 5
        C, r, c, Gamma = [], None, None, None                               # 6
        sigma_a = sigma_g = 0.0
        if i >= 1:                                                          # 7
            last, release = X[i][j - 1], X[i][1]
            leg = th_a(last, p)
            e = self.Gamma[mu] + eta + leg                                  # 8
            sigma_a, sigma_g = max(0.0, w - e), 0.0                         # 9
            P = [par.pi(X[i][k]) for k in range(2, j)] + [par.pi(p)]       # 10
            C = [ch for ch in P                                             # 11
                 if eta + leg + sigma_a + th_a(p, ch) + par.delta_a
                 <= par.tau_a_max
                 and th_g(release, ch) + par.delta_g <= par.tau_a_max]
            t = max(e, w)                                     # 12: e + sigma_a
        if not C:                                                           # 13
            r = par.pi(p)                                                   # 14
            if i == 0:                                                      # 15
                drive = th_g(self.instance.starts[mu], r)
                e = drive + th_a(r, p)                                      # 16
                sigma_g, sigma_a = max(0.0, w - e), 0.0                     # 17
                Gamma = drive + sigma_g                                     # 18
            else:                                                           # 19
                c, b = self._collect(                                       # 20-21
                    mu, lambda ch, bh: bh + max(th_g(ch, r), tau_c(bh)))
                gap = max(th_g(c, r), tau_c(b))
                e = self.Gamma[mu] + b + gap + th_a(r, p)                   # 22
                sigma_g, sigma_a = max(0.0, w - e), 0.0                     # 23
                Gamma = self.Gamma[mu] + b + gap + sigma_g                  # 24
            t = max(e, w)                         # 25: Gamma' + theta_a(r, q)
            C = [r]
        return Option(t, r, c, sigma_a, sigma_g, C, Gamma)                  # 26

    def commit(self, mu: int, q: int, opt: Option) -> None:
        """Lines 30-45: write the chosen option into team mu's plan."""
        par, n = self.params, self.instance.n
        p = self.instance.points[q]
        X, Ta = self.X[mu], self.Theta_a[mu]
        i, j = self.i[mu], self.j[mu]
        if opt.r is None:                                                   # 30
            X[i][j] = p                                                     # 31
            Ta[i][j - 1] = par.theta_a(X[i][j - 1], p) + opt.sigma_a        # 32
            self.eta[mu] += Ta[i][j - 1]                                    # 33
        else:                                                               # 34
            if i >= 1:                                                      # 35
                self._close_row(mu, opt.c)                                  # 36-38
                b = max(self.eta[mu] + Ta[i][n + 1], self.Theta_g[mu][i])   # 39
                self.Theta_ga[mu][i] = (max(par.theta_g(opt.c, opt.r),
                                            par.tau_c(b)) + opt.sigma_g)
            else:                                                           # 40
                self.theta_s[mu] = (par.theta_g(self.instance.starts[mu],   # 41
                                                opt.r) + opt.sigma_g)
            i = self.i[mu] = i + 1                                          # 42
            self.j[mu], self.Gamma[mu] = 2, opt.Gamma
            X[i][1], X[i][2] = opt.r, p                                     # 43
            Ta[i][1] = par.theta_a(opt.r, p)                                # 44
            self.eta[mu] = Ta[i][1]
        self.C[mu] = opt.C                                                  # 45
        self.j[mu] += 1
        self.t[q] = self.T[mu] = opt.t

    def close(self) -> None:
        """Lines 48-57: land every active team's last tour, send idle UGVs
        straight to their finish."""
        par, n = self.params, self.instance.n
        for mu in range(self.instance.m):                                   # 48
            pf, X = self.instance.finishes[mu], self.X[mu]
            if self.i[mu] >= 1:                                             # 49
                c, _ = self._collect(                                       # 50
                    mu, lambda ch, bh: bh + par.theta_g(ch, pf))
                self._close_row(mu, c)                                      # 51-53
                self.theta_f[mu] = par.theta_g(c, pf)
                for k in range(self.i[mu] + 1, n + 1):                      # 54
                    X[k][1:] = [c] * (n + 2)
            else:                                                           # 55
                for k in range(1, n + 1):                                   # 56
                    X[k][1:] = [pf] * (n + 2)
                self.theta_s[mu] = par.theta_g(self.instance.starts[mu],    # 57
                                               pf)

    def plan(self) -> Plan:
        """The finished (X, Theta) as 0-based arrays."""
        m, n = self.instance.m, self.instance.n
        return Plan(
            X=np.array([[row[1:] for row in team[1:]] for team in self.X],
                       dtype=float),
            Theta_a=np.array([[row[1:] for row in team[1:]]
                              for team in self.Theta_a]),
            Theta_g=np.array([team[1:] for team in self.Theta_g]),
            Theta_ga=np.array([team[1:] for team in self.Theta_ga],
                              dtype=float).reshape(m, n - 1),
            theta_s=np.array(self.theta_s),
            theta_f=np.array(self.theta_f))

    def _collect(self, mu: int, cost: Callable[[Point, float], float]
                 ) -> Tuple[Point, float]:
        """Lines 20-21 and 50: the candidate collect point c of team mu's
        active tour minimising cost(c, b), where b is the tour's airborne time
        if it lands at c. Returns (c, b); ties go to the first candidate."""
        par = self.params
        X, i, j = self.X[mu], self.i[mu], self.j[mu]
        last, release = X[i][j - 1], X[i][1]
        best = None
        for ch in self.C[mu]:
            bh = max(self.eta[mu] + par.theta_a(last, ch),
                     par.theta_g(release, ch))
            value = cost(ch, bh)
            if best is None or value < best[0]:
                best = (value, ch, bh)
        return best[1], best[2]

    def _close_row(self, mu: int, c: Point) -> None:
        """Lines 36-38 (and 51-53): pad the active row, land it at c."""
        par, n = self.params, self.instance.n
        X, i, j = self.X[mu], self.i[mu], self.j[mu]
        last = X[i][j - 1]
        for k in range(j, n + 2):
            X[i][k] = last
        X[i][n + 2] = c
        self.Theta_a[mu][i][n + 1] = par.theta_a(last, c)
        self.Theta_g[mu][i] = par.theta_g(X[i][1], c)
        self.candidates[mu].append(list(self.C[mu]))


def solve(instance: Instance, params: Optional[Params] = None) -> Result:
    """Plan the mission with Alg. 1.

    Raises ValueError if Assumption 1 fails, i.e. some point cannot be
    visited even by a tour of its own.
    """
    params = params or Params()
    bad = assumption_1_violations(instance, params)
    if bad:
        raise ValueError(f"Assumption 1 fails for points {bad}: they have no "
                         f"feasible single-point tour")
    start = time.perf_counter()
    n, m = instance.n, instance.m
    builder = Builder(instance, params)
    succ = instance.successors()
    waiting = [len(p) for p in builder.preds]
    ready = [q for q in range(n) if waiting[q] == 0]                        # 2
    while ready:                                                            # 3
        best = None
        for mu in range(m):                                                 # 4
            others = max((builder.T[nu] for nu in range(m) if nu != mu),
                         default=0.0)
            for q in ready:
                opt = builder.evaluate(mu, q)                               # 5-26
                score = max(opt.t, others)                                  # 27
                # strict <: ties go to the lowest team, then the lowest point
                if best is None or score < best[0]:
                    best = (score, mu, q, opt)
        _, mu, q, opt = best                                                # 28-29
        builder.commit(mu, q, opt)                                          # 30-45
        ready.remove(q)                                                     # 46-47
        for s in succ[q]:
            waiting[s] -= 1
            if waiting[s] == 0:
                bisect.insort(ready, s)
    builder.close()                                                         # 48-57
    plan = builder.plan()                                                   # 58
    runtime = time.perf_counter() - start
    info = {"visit_times": np.array(builder.t, dtype=float),
            "candidates": builder.candidates}
    return make_result("alg1", instance, params, plan, runtime, info)

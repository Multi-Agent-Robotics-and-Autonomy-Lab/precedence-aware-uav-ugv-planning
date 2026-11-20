"""Branch and cut (Section IV-D): problem (3) as a mixed-integer quadratically
constrained program (MIQCP), solved by Gurobi.

The variables are the plan and its timing as in Section II. Every entry of
X^mu is a point: binary z[mu, i, j, k] puts monitoring point k in waypoint slot
j of tour i, and (3f) copies the previous entry into an empty slot (a bilinear
constraint). Release and collect points are continuous positions on the ground,
as (3e) allows. Each distance is a variable tied to its coordinate differences
by a non-convex quadratic equality, with an octagonal outer approximation added
as valid cuts. Clocks accumulate Eq. (2), and visit times are linked to arrival
times by big-M constraints. Each point is visited exactly once, which loses no
optimal value, because a repeated visit can be replaced by waiting. A
round-robin plan with one point per tour seeds the search, and its makespan
bounds every clock of an optimal plan.

Needs gurobipy. The model grows as m n^3; the size-limited licence that ships
with the pip package covers n <= 3 with one team or n = 2 with two teams.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from .problem import (Instance, Params, Plan, Result, assumption_1_violations,
                      make_result)

# Gurobi settings of Section IV-D
SETTINGS = {"MIPGap": 0.05, "FeasibilityTol": 1e-4, "IntFeasTol": 1e-5,
            "OptimalityTol": 1e-3, "NonConvex": 2}


def solve(instance: Instance, params: Optional[Params] = None,
          time_limit: float = 600.0, verbose: bool = False) -> Result:
    """Solve (3) by branch and cut. Returns the best plan found within
    `time_limit` seconds; `info` holds the solver status, the objective, the
    best bound and the optimality gap."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as err:
        raise ImportError("branch_and_cut needs gurobipy: pip install gurobipy") from err
    params = params or Params()
    bad = assumption_1_violations(instance, params)
    if bad:
        raise ValueError(f"Assumption 1 fails for points {bad}: they have no "
                         f"feasible single-point tour")
    for name in ("theta_a", "theta_g", "tau_c", "pi"):
        if getattr(type(params), name) is not getattr(Params, name):
            raise NotImplementedError(
                f"the MIQCP encodes the Table I model; Params.{name} is overridden")

    start = time.perf_counter()
    n, m, P = instance.n, instance.m, instance.points
    W, H = instance.size
    z_top = max(p[2] for p in P)
    rows, slots, legs = range(n), range(n), range(n + 1)
    tau_max = params.tau_a_max
    ws = _warm_start(instance, params)
    horizon = ws["makespan"] * (1 + 1e-6) + 1.0

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", int(verbose))
    env.start()
    model = gp.Model("problem_3", env=env)
    for key, value in SETTINGS.items():
        model.setParam(key, value)
    model.setParam("TimeLimit", time_limit)

    # ---- variables -------------------------------------------------------
    x = model.addVars(m, n, n + 2, 3, lb=0.0, name="x")
    for mu in range(m):
        for i in rows:
            for j in range(n + 2):
                x[mu, i, j, 0].ub, x[mu, i, j, 1].ub, x[mu, i, j, 2].ub = W, H, z_top
    z = model.addVars(m, n, n, n, vtype=GRB.BINARY, name="z")
    th_a = model.addVars(m, n, n + 1, lb=0.0, ub=tau_max, name="Theta_a")
    th_g = model.addVars(m, n, lb=0.0, ub=tau_max, name="Theta_g")
    th_ga = model.addVars(m, max(n - 1, 0), lb=0.0, ub=horizon, name="Theta_ga")
    th_s = model.addVars(m, lb=0.0, ub=horizon, name="theta_s")
    th_f = model.addVars(m, lb=0.0, ub=horizon, name="theta_f")
    tau_a = model.addVars(m, n, lb=0.0, ub=tau_max, name="tau_a")
    airborne = model.addVars(m, n, lb=0.0, ub=tau_max, name="airborne")
    gap = model.addVars(m, max(n - 1, 0), lb=0.0, ub=horizon, name="gap")
    release = model.addVars(m, n, lb=0.0, ub=horizon, name="release")
    arrive = model.addVars(m, n, n + 2, lb=0.0, ub=horizon, name="arrive")
    t = model.addVars(n, lb=0.0, ub=horizon, name="t")
    team_time = model.addVars(m, lb=0.0, ub=horizon, name="team_time")
    makespan = model.addVar(lb=0.0, ub=horizon, name="makespan")

    dist = {}

    def distance(key, dx, dy):
        """A variable equal to the planar distance between two points."""
        d = model.addVar(lb=0.0, name=f"d_{key}")
        model.addQConstr(d * d == dx * dx + dy * dy)
        for e in (dx, -dx, dy, -dy):
            model.addConstr(d >= e)
        for e in (dx + dy, dx - dy, -dx + dy, -dx - dy):
            model.addConstr(math.sqrt(2.0) * d >= e)
        dist[key] = d
        return d

    def gap_xy(a, b):
        return x[a + (0,)] - x[b + (0,)], x[a + (1,)] - x[b + (1,)]

    # ---- (3b) coverage, one visit per point --------------------------------
    for k in range(n):
        model.addConstr(z.sum("*", "*", "*", k) == 1)
    for mu in range(m):
        for i in rows:
            for j in slots:
                model.addConstr(z.sum(mu, i, j, "*") <= 1)
                for k in range(n):                    # slot j holds point k
                    for c, big in ((0, W), (1, H), (2, z_top)):
                        model.addConstr(x[mu, i, j + 1, c] - P[k][c]
                                        <= big * (1 - z[mu, i, j, k]))
                        model.addConstr(x[mu, i, j + 1, c] - P[k][c]
                                        >= -big * (1 - z[mu, i, j, k]))
                used = z.sum(mu, i, j, "*")           # (3f): else repeat
                for c in range(3):
                    model.addConstr(
                        x[mu, i, j + 1, c]
                        == gp.quicksum(z[mu, i, j, k] * P[k][c] for k in range(n))
                        + (1 - used) * x[mu, i, j, c])
            model.addConstr(x[mu, i, 0, 2] == 0)       # (3e)
            model.addConstr(x[mu, i, n + 1, 2] == 0)

    # ---- (3c)-(3d), (3g)-(3j) and the terms of Eq. (2) ---------------------
    for mu in range(m):
        for i in rows:
            for j in legs:                             # (3g)
                dz = model.addVar(lb=-z_top, ub=z_top, name=f"dz_{mu}_{i}_{j}")
                dz_abs = model.addVar(lb=0.0, ub=z_top, name=f"adz_{mu}_{i}_{j}")
                model.addConstr(dz == x[mu, i, j, 2] - x[mu, i, j + 1, 2])
                model.addGenConstrAbs(dz_abs, dz)
                dist["dz", mu, i, j], dist["adz", mu, i, j] = dz, dz_abs
                d = distance(("leg", mu, i, j), *gap_xy((mu, i, j), (mu, i, j + 1)))
                model.addConstr(th_a[mu, i, j] >= d / params.s_h + dz_abs / params.s_v)
            model.addConstr(tau_a[mu, i] == th_a.sum(mu, i, "*"))
            model.addConstr(tau_a[mu, i] + params.delta_a <= tau_max)        # (3c)
            d = distance(("tour", mu, i), *gap_xy((mu, i, n + 1), (mu, i, 0)))
            model.addConstr(th_g[mu, i] >= d / params.s_g)                   # (3h)
            model.addConstr(th_g[mu, i] + params.delta_g <= tau_max)         # (3d)
            model.addGenConstrMax(airborne[mu, i], [tau_a[mu, i], th_g[mu, i]])
        for i in range(n - 1):                         # (3i), gap of Eq. (2)
            d = distance(("between", mu, i), *gap_xy((mu, i + 1, 0), (mu, i, n + 1)))
            model.addConstr(th_ga[mu, i] >= d / params.s_g)
            recharge = model.addVar(lb=0.0, name=f"tau_c_{mu}_{i}")
            model.addConstr(recharge == params.gamma * airborne[mu, i])
            model.addGenConstrMax(gap[mu, i], [th_ga[mu, i], recharge])
            dist["recharge", mu, i] = recharge
        s, f = instance.starts[mu], instance.finishes[mu]  # (3j)
        d = distance(("start", mu), x[mu, 0, 0, 0] - s[0], x[mu, 0, 0, 1] - s[1])
        model.addConstr(th_s[mu] >= d / params.s_g)
        d = distance(("finish", mu), x[mu, n - 1, n + 1, 0] - f[0],
                     x[mu, n - 1, n + 1, 1] - f[1])
        model.addConstr(th_f[mu] >= d / params.s_g)

        # clocks: Eq. (2) as a running sum
        model.addConstr(release[mu, 0] == th_s[mu])
        for i in range(n - 1):
            model.addConstr(release[mu, i + 1]
                            == release[mu, i] + airborne[mu, i] + gap[mu, i])
        for i in rows:
            model.addConstr(arrive[mu, i, 0] == release[mu, i])
            for j in legs:
                model.addConstr(arrive[mu, i, j + 1] == arrive[mu, i, j] + th_a[mu, i, j])
        model.addConstr(team_time[mu]
                        == release[mu, n - 1] + airborne[mu, n - 1] + th_f[mu])
        model.addConstr(makespan >= team_time[mu])     # (3a)

    # ---- visit times and (3k) ----------------------------------------------
    for k in range(n):
        for mu in range(m):
            for i in rows:
                for j in slots:
                    model.addConstr(t[k] <= arrive[mu, i, j + 1] + horizon * (1 - z[mu, i, j, k]))
                    model.addConstr(t[k] >= arrive[mu, i, j + 1] - horizon * (1 - z[mu, i, j, k]))
    for p, q in instance.edges:
        model.addConstr(t[p] + params.delta_o <= t[q])
    model.setObjective(makespan, GRB.MINIMIZE)

    _apply_warm_start(ws, instance, params, dict(
        x=x, z=z, th_a=th_a, th_g=th_g, th_ga=th_ga, th_s=th_s, th_f=th_f,
        tau_a=tau_a, airborne=airborne, gap=gap, release=release,
        arrive=arrive, t=t, team_time=team_time, makespan=makespan), dist)
    model.optimize()
    if model.SolCount == 0:
        status = model.Status
        model.dispose()
        env.dispose()
        raise RuntimeError(f"no feasible plan found (Gurobi status {status})")

    plan = _extract(instance, params, model, x, z, th_a, th_g, th_ga, th_s, th_f)
    info = {"status": model.Status, "objective": model.ObjVal,
            "bound": model.ObjBound, "mip_gap": model.MIPGap,
            "nodes": model.NodeCount, "warm_start_makespan": ws["makespan"]}
    model.dispose()
    env.dispose()
    runtime = time.perf_counter() - start
    # Gurobi meets constraints to FeasibilityTol = 1e-4, so check to 1e-3
    return make_result("branch_and_cut", instance, params, plan, runtime, info,
                       tol=1e-3)


def _extract(instance, params, model, x, z, th_a, th_g, th_ga, th_s, th_f):
    """The incumbent as a Plan. Waypoint slots snap to the exact monitoring
    point (or repeat) that z selects, ground points to z = 0, and every
    duration is lifted to at least its fastest time, removing solver
    tolerance."""
    n, m, P = instance.n, instance.m, instance.points
    X = np.zeros((m, n, n + 2, 3))
    for mu in range(m):
        for i in range(n):
            X[mu, i, 0] = (x[mu, i, 0, 0].X, x[mu, i, 0, 1].X, 0.0)
            for j in range(n):
                k = next((k for k in range(n) if z[mu, i, j, k].X > 0.5), None)
                X[mu, i, j + 1] = P[k] if k is not None else X[mu, i, j]
            X[mu, i, n + 1] = (x[mu, i, n + 1, 0].X, x[mu, i, n + 1, 1].X, 0.0)
    pt = lambda mu, i, j: tuple(X[mu, i, j])
    Ta = np.array([[[max(th_a[mu, i, j].X, params.theta_a(pt(mu, i, j), pt(mu, i, j + 1)))
                     for j in range(n + 1)] for i in range(n)] for mu in range(m)])
    Tg = np.array([[max(th_g[mu, i].X, params.theta_g(pt(mu, i, 0), pt(mu, i, n + 1)))
                    for i in range(n)] for mu in range(m)])
    Tga = np.array([[max(th_ga[mu, i].X, params.theta_g(pt(mu, i, n + 1), pt(mu, i + 1, 0)))
                     for i in range(n - 1)] for mu in range(m)], dtype=float).reshape(m, n - 1)
    ts = np.array([max(th_s[mu].X, params.theta_g(instance.starts[mu], pt(mu, 0, 0)))
                   for mu in range(m)])
    tf = np.array([max(th_f[mu].X, params.theta_g(pt(mu, n - 1, n + 1), instance.finishes[mu]))
                   for mu in range(m)])
    return Plan(X, Ta, Tg, Tga, ts, tf)


# ============================ WARM START =====================================
def _warm_start(instance, params):
    """One point per tour, dealt round-robin along a topological order,
    released and collected directly below it; rows after a team's last tour
    are trivial at its last release (or its start). Precedence is then met by
    waiting on the ground before the tour of each late point. One pass in
    topological order suffices: delaying a tour only moves points that come
    later in the order. Under Assumption 1 the result is feasible."""
    n, m, P = instance.n, instance.m, instance.points
    order = _fifo_topological_order(instance)
    preds = instance.predecessors()
    teams = [order[mu::m] for mu in range(m)]
    X, where = [], {}
    for mu, pts in enumerate(teams):
        rows, park = [], instance.starts[mu]
        for i in range(n):
            if i < len(pts):
                q = pts[i]
                g = params.pi(P[q])
                rows.append([g] + [P[q]] * n + [g])
                where[q] = (mu, i)
                park = g
            else:
                rows.append([park] * (n + 2))
        X.append(rows)
    th_a, th_g = params.theta_a, params.theta_g
    Ta = [[[th_a(X[mu][i][j], X[mu][i][j + 1]) for j in range(n + 1)] for i in range(n)]
          for mu in range(m)]
    Tg = [[th_g(X[mu][i][0], X[mu][i][n + 1]) for i in range(n)] for mu in range(m)]
    Tga = [[th_g(X[mu][i][n + 1], X[mu][i + 1][0]) for i in range(n - 1)] for mu in range(m)]
    ts = [th_g(instance.starts[mu], X[mu][0][0]) for mu in range(m)]
    tf = [th_g(X[mu][n - 1][n + 1], instance.finishes[mu]) for mu in range(m)]

    def clocks(mu):
        rel, out = [ts[mu]], []
        for i in range(n):
            out.append(rel[i] + Ta[mu][i][0])          # the tour's one visit
            if i < n - 1:
                air = max(sum(Ta[mu][i]), Tg[mu][i])
                rel.append(rel[i] + air + max(Tga[mu][i], params.tau_c(air)))
        return rel, out

    sched = [clocks(mu) for mu in range(m)]
    for q in order:
        if not preds[q]:
            continue
        mu, i = where[q]
        need = max(sched[where[p][0]][1][where[p][1]] + params.delta_o for p in preds[q])
        late = need - sched[mu][1][i]
        if late > 1e-9:
            if i == 0:
                ts[mu] += late
            else:
                # the gap is max(transport, recharge): lift the transport to
                # the recharge first, so the whole wait moves the tour
                air = max(sum(Ta[mu][i - 1]), Tg[mu][i - 1])
                Tga[mu][i - 1] = max(Tga[mu][i - 1], params.tau_c(air)) + late
            sched[mu] = clocks(mu)
    times = []
    for mu in range(m):
        rel = sched[mu][0]
        times.append(rel[n - 1] + max(sum(Ta[mu][n - 1]), Tg[mu][n - 1]) + tf[mu])
    return dict(X=X, Ta=Ta, Tg=Tg, Tga=Tga, ts=ts, tf=tf, sched=sched,
                where=where, team_times=times, makespan=max(times))


def _fifo_topological_order(instance):
    succ, waiting = instance.successors(), [len(p) for p in instance.predecessors()]
    ready = [k for k in range(instance.n) if waiting[k] == 0]
    order = []
    while ready:
        k = ready.pop(0)
        order.append(k)
        for s in succ[k]:
            waiting[s] -= 1
            if waiting[s] == 0:
                ready.append(s)
    return order


def _apply_warm_start(ws, instance, params, v, dist):
    """Set a start value on every variable; Gurobi only accepts a start it can
    check against the quadratic constraints if the derived values are set
    too."""
    n, m = instance.n, instance.m
    X, Ta, Tg, Tga = ws["X"], ws["Ta"], ws["Tg"], ws["Tga"]
    for mu in range(m):
        rel, visit = ws["sched"][mu]
        for i in range(n):
            for j in range(n + 2):
                for c in range(3):
                    v["x"][mu, i, j, c].Start = X[mu][i][j][c]
            for j in range(n):
                for k in range(n):
                    v["z"][mu, i, j, k].Start = 1.0 if (ws["where"].get(k) == (mu, i) and j == 0) else 0.0
            arrival = rel[i]
            v["arrive"][mu, i, 0].Start = arrival
            for j in range(n + 1):
                a, b = X[mu][i][j], X[mu][i][j + 1]
                v["th_a"][mu, i, j].Start = Ta[mu][i][j]
                dist["leg", mu, i, j].Start = math.hypot(a[0] - b[0], a[1] - b[1])
                dist["dz", mu, i, j].Start = a[2] - b[2]
                dist["adz", mu, i, j].Start = abs(a[2] - b[2])
                arrival += Ta[mu][i][j]
                v["arrive"][mu, i, j + 1].Start = arrival
            a, b = X[mu][i][0], X[mu][i][n + 1]
            dist["tour", mu, i].Start = math.hypot(a[0] - b[0], a[1] - b[1])
            air = max(sum(Ta[mu][i]), Tg[mu][i])
            v["tau_a"][mu, i].Start = sum(Ta[mu][i])
            v["th_g"][mu, i].Start = Tg[mu][i]
            v["airborne"][mu, i].Start = air
            v["release"][mu, i].Start = rel[i]
            if i < n - 1:
                a, b = X[mu][i][n + 1], X[mu][i + 1][0]
                dist["between", mu, i].Start = math.hypot(a[0] - b[0], a[1] - b[1])
                v["th_ga"][mu, i].Start = Tga[mu][i]
                dist["recharge", mu, i].Start = params.tau_c(air)
                v["gap"][mu, i].Start = max(Tga[mu][i], params.tau_c(air))
        s, f = instance.starts[mu], instance.finishes[mu]
        a, b = X[mu][0][0], X[mu][n - 1][n + 1]
        dist["start", mu].Start = math.hypot(a[0] - s[0], a[1] - s[1])
        dist["finish", mu].Start = math.hypot(b[0] - f[0], b[1] - f[1])
        v["th_s"][mu].Start = ws["ts"][mu]
        v["th_f"][mu].Start = ws["tf"][mu]
        v["team_time"][mu].Start = ws["team_times"][mu]
    for k, (mu, i) in ws["where"].items():
        v["t"][k].Start = ws["sched"][mu][1][i]
    v["makespan"].Start = ws["makespan"]

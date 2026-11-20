"""Problem definition shared by every planner.

Section II of the paper, in code: the parameters of Table I (`Params`), a
mission (`Instance`), a plan and its timing (`Plan`, the pair (X, Theta)), the
team execution time of Eq. (2), the visit times t_p, the constraint checks
(3b)-(3k) and Assumption 1. Every planner returns the same `Result`.

Indexing. Arrays are 0-based: row i of a team's plan is tour i+1 of the paper,
column 0 is the release point, columns 1..n hold the visited points and column
n+1 is the collect point.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float, float]


# ============================ PARAMETERS =====================================
@dataclass(frozen=True)
class Params:
    """Model and method parameters. The defaults are Table I.

    The travel-time models, the recharge function and the ground projection
    are methods, so a subclass can replace any of them (the paper does not
    commit to specific forms, footnote 2). The defaults describe the
    obstacle-free environment of Section IV with a VTOL UAV.
    """

    s_h: float = 10.0          # UAV cruise speed s_H [m/s]
    s_v: float = 2.0           # UAV climb speed s_V [m/s]
    s_g: float = 2.5           # UGV speed s_g [m/s]
    gamma: float = 1.0         # recharge ratio
    tau_a_max: float = 600.0   # maximum flight time, tau_bar_a [s]
    delta_a: float = 0.0       # UAV energy robustness margin [s]
    delta_g: float = 0.0       # UGV energy robustness margin [s]
    delta_o: float = 0.0       # ordering robustness margin [s]

    def __post_init__(self):
        if min(self.s_h, self.s_v, self.s_g, self.tau_a_max) <= 0:
            raise ValueError("speeds and tau_a_max must be positive")
        if min(self.gamma, self.delta_a, self.delta_g, self.delta_o) < 0:
            raise ValueError("gamma and the margins must be non-negative")

    def theta_a(self, u: Point, v: Point) -> float:
        """Fastest UAV travel time: horizontal part at s_H, vertical at s_V."""
        return (math.hypot(u[0] - v[0], u[1] - v[1]) / self.s_h
                + abs(u[2] - v[2]) / self.s_v)

    def theta_g(self, u: Point, v: Point) -> float:
        """Fastest UGV travel time, l(u, v) / s_g with l Euclidean."""
        return math.dist(u, v) / self.s_g

    def tau_c(self, airborne: float) -> float:
        """Recharge time for a tour that kept the UAV airborne this long."""
        return self.gamma * airborne

    def pi(self, p: Point) -> Point:
        """Feasible ground projection: with no obstacles, straight down."""
        return (p[0], p[1], 0.0)


# ============================ INSTANCE =======================================
@dataclass(frozen=True)
class Instance:
    """A mission: monitoring points P_UAV, the precedence DAG E, and each
    team's start and finish position. `size` is (x_bar, y_bar) of Eq. (1)."""

    points: Sequence[Point]
    edges: Sequence[Tuple[int, int]]
    starts: Sequence[Point]
    finishes: Sequence[Point]
    size: Tuple[float, float] = (4000.0, 4000.0)

    def __post_init__(self):
        pts = tuple(_as_point(p) for p in self.points)
        starts = tuple(_as_point(p) for p in self.starts)
        finishes = tuple(_as_point(p) for p in self.finishes)
        size = (float(self.size[0]), float(self.size[1]))
        n = len(pts)
        if n < 1:
            raise ValueError("an instance needs at least one monitoring point")
        if not starts or len(starts) != len(finishes):
            raise ValueError("give one start and one finish per team")
        for p in pts:
            if p[2] <= 0 or not _inside(p, size):
                raise ValueError(f"monitoring point {p} is not in the air "
                                 f"space of the environment")
        if len(set(pts)) != n:
            raise ValueError("monitoring points must be distinct")
        for p in starts + finishes:
            if p[2] != 0 or not _inside(p, size):
                raise ValueError(f"start/finish {p} is not on the ground")
        edges = tuple(sorted({(int(a), int(b)) for a, b in self.edges}))
        for a, b in edges:
            if not (0 <= a < n and 0 <= b < n) or a == b:
                raise ValueError(f"bad precedence edge {(a, b)}")
        if len(topological_order(n, edges)) != n:
            raise ValueError("the precedence graph must be acyclic")
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "finishes", finishes)
        object.__setattr__(self, "size", size)

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def m(self) -> int:
        return len(self.starts)

    def predecessors(self) -> List[List[int]]:
        out = [[] for _ in range(self.n)]
        for a, b in self.edges:
            out[b].append(a)
        return out

    def successors(self) -> List[List[int]]:
        out = [[] for _ in range(self.n)]
        for a, b in self.edges:
            out[a].append(b)
        return out


def _as_point(p) -> Point:
    x, y, z = (float(c) for c in p)
    return (x, y, z)


def _inside(p: Point, size, tol: float = 0.0) -> bool:
    return -tol <= p[0] <= size[0] + tol and -tol <= p[1] <= size[1] + tol


def topological_order(n: int, edges) -> List[int]:
    """Kahn's algorithm; shorter than n exactly when the graph has a cycle."""
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for a, b in edges:
        succ[a].append(b)
        indeg[b] += 1
    ready = [k for k in range(n) if indeg[k] == 0]
    order = []
    while ready:
        k = ready.pop()
        order.append(k)
        for s in succ[k]:
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
    return order


def make_instance(n: int, m: int, d: float = 0.5, seed: int = 0,
                  size: float = 4000.0, altitude: float = 100.0,
                  start: Point = (0.0, 0.0, 0.0),
                  finish: Optional[Point] = None) -> Instance:
    """A random instance as in Section IV.

    Points are uniform over [0, size]^2 at `altitude`. The precedence graph
    takes a random permutation of the points and adds round(d n) edges between
    distinct random pairs, each directed from the earlier point in the
    permutation to the later one, so it is acyclic. All teams share `start`
    and `finish` (Table I). Points and edges use separate random streams, so
    changing `d` leaves the points unchanged.
    """
    if finish is None:
        finish = (size, size, 0.0)
    rng = random.Random(seed)
    points = [(rng.uniform(0, size), rng.uniform(0, size), altitude)
              for _ in range(n)]
    edges = random_dag(n, d, random.Random(1000 * seed + 7))
    return Instance(points, edges, [start] * m, [finish] * m, (size, size))


def random_dag(n: int, d: float, rng: random.Random) -> List[Tuple[int, int]]:
    """round(d n) distinct edges oriented along a random permutation."""
    k = min(round(d * n), n * (n - 1) // 2) if d > 0 else 0
    order = list(range(n))
    rng.shuffle(order)
    rank = {v: i for i, v in enumerate(order)}
    pairs, tries = set(), 0
    while len(pairs) < k and tries < 100 * max(1, k):
        u, v = rng.sample(range(n), 2)
        pairs.add((u, v) if rank[u] < rank[v] else (v, u))
        tries += 1
    return sorted(pairs)


# ============================ PLAN AND RESULT ================================
@dataclass
class Plan:
    """The plan X and its timing Theta, shaped as in Section II.

    X         (m, n, n+2, 3)  X^mu_{i,j}, points in Q_f
    Theta_a   (m, n, n+1)     UAV flight time from X_{i,j} to X_{i,j+1}
    Theta_g   (m, n)          UGV time from the release to the collect of tour i
    Theta_ga  (m, n-1)        UGV transport from tour i's collect to tour i+1's release
    theta_s   (m,)            initial transport, p_s to X_{1,1}
    theta_f   (m,)            final transport, X_{n,n+2} to p_f
    """

    X: np.ndarray
    Theta_a: np.ndarray
    Theta_g: np.ndarray
    Theta_ga: np.ndarray
    theta_s: np.ndarray
    theta_f: np.ndarray


class Tour(NamedTuple):
    """One non-trivial row of a plan, for printing and plotting."""

    team: int
    row: int
    release: Point
    visits: List[int]          # monitoring points in visit order
    collect: Point


@dataclass
class Result:
    """What every planner's `solve` returns.

    `makespan` is (3a) and `team_times` is Eq. (2), both computed from `plan`;
    `visit_times` holds t_p and `violations` lists every failed constraint of
    (3b)-(3k), so an empty list means the plan is feasible. `runtime` is the
    planner's wall-clock time in seconds; `info` holds planner extras.
    """

    method: str
    plan: Plan
    makespan: float
    team_times: np.ndarray
    visit_times: np.ndarray
    violations: List[str]
    runtime: float
    info: dict = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return not self.violations


def make_result(method: str, instance: Instance, params: Params, plan: Plan,
                runtime: float, info: Optional[dict] = None,
                tol: float = 1e-6) -> Result:
    times = team_times(plan, params)
    return Result(method=method, plan=plan, makespan=float(times.max()),
                  team_times=times,
                  visit_times=visit_times(instance, params, plan),
                  violations=check_constraints(instance, params, plan, tol),
                  runtime=runtime, info=info or {})


# ============================ OBJECTIVE ======================================
def team_times(plan: Plan, params: Params) -> np.ndarray:
    """tau(X^mu, Theta^mu) of Eq. (2) for every team.

    Taken literally, over all n rows. Alg. 1 fills the rows after a team's
    last tour with trivial tours (line 54), so for such a team the gap term
    after its last tour is tau_c of that tour: the UAV is recharged before the
    final transport is counted.
    """
    m, n = plan.Theta_g.shape
    out = np.empty(m)
    for mu in range(m):
        tau_a = plan.Theta_a[mu].sum(axis=1)
        airborne = np.maximum(tau_a, plan.Theta_g[mu])
        gaps = sum(max(plan.Theta_ga[mu, i], params.tau_c(airborne[i]))
                   for i in range(n - 1))
        out[mu] = (plan.theta_s[mu] + plan.theta_f[mu] + airborne.sum()
                   + gaps)
    return out


def makespan(plan: Plan, params: Params) -> float:
    """(3a): the largest team execution time."""
    return float(team_times(plan, params).max())


def visit_times(instance: Instance, params: Params, plan: Plan) -> np.ndarray:
    """t_p: the earliest time any UAV reaches p, from the mission start.

    Tour i is released at theta_s plus, for every earlier tour, its airborne
    time max(tau_a, tau_g) and its gap max(tau_ga, tau_c), as in Eq. (2).
    Unvisited points get NaN.
    """
    index = {p: k for k, p in enumerate(instance.points)}
    m, n = plan.Theta_g.shape
    X, Ta = plan.X.tolist(), plan.Theta_a.tolist()
    Tg, Tga = plan.Theta_g.tolist(), plan.Theta_ga.tolist()
    t = [math.nan] * instance.n
    for mu in range(m):
        release = float(plan.theta_s[mu])
        for i in range(n):
            arrival = release
            for j in range(1, n + 1):
                arrival += Ta[mu][i][j - 1]
                k = index.get(tuple(X[mu][i][j]))
                if k is not None and (math.isnan(t[k]) or arrival < t[k]):
                    t[k] = arrival
            if i < n - 1:
                airborne = max(sum(Ta[mu][i]), Tg[mu][i])
                release += airborne + max(Tga[mu][i], params.tau_c(airborne))
    return np.array(t)


# ============================ CONSTRAINTS ====================================
def check_constraints(instance: Instance, params: Params, plan: Plan,
                      tol: float = 1e-6) -> List[str]:
    """Every violated constraint of (3b)-(3k), as readable messages."""
    m, n = instance.m, instance.n
    expect = {"X": (m, n, n + 2, 3), "Theta_a": (m, n, n + 1),
              "Theta_g": (m, n), "Theta_ga": (m, n - 1),
              "theta_s": (m,), "theta_f": (m,)}
    bad = [f"{k} has shape {getattr(plan, k).shape}, expected {s}"
           for k, s in expect.items() if getattr(plan, k).shape != s]
    if bad:
        return bad

    errs = []
    points = set(instance.points)
    th_a, th_g = params.theta_a, params.theta_g
    X, Ta = plan.X.tolist(), plan.Theta_a.tolist()
    Tg, Tga = plan.Theta_g.tolist(), plan.Theta_ga.tolist()
    for mu in range(m):
        for i in range(n):
            row = [tuple(x) for x in X[mu][i]]
            tau_a = sum(Ta[mu][i])
            tau_g = Tg[mu][i]
            if tau_a + params.delta_a > params.tau_a_max + tol:        # (3c)
                errs.append(f"(3c) team {mu} tour {i + 1}: tau_a + delta_a = "
                            f"{tau_a + params.delta_a:.3f} > {params.tau_a_max}")
            if tau_g + params.delta_g > params.tau_a_max + tol:        # (3d)
                errs.append(f"(3d) team {mu} tour {i + 1}: tau_g + delta_g = "
                            f"{tau_g + params.delta_g:.3f} > {params.tau_a_max}")
            for j in (0, n + 1):                                       # (3e)
                if abs(row[j][2]) > tol or not _inside(row[j], instance.size, tol):
                    errs.append(f"(3e) team {mu} tour {i + 1}: "
                                f"{'release' if j == 0 else 'collect'} "
                                f"{row[j]} is not on the ground")
            for j in range(1, n + 1):                                  # (3f)
                if row[j] not in points and row[j] != row[j - 1]:
                    errs.append(f"(3f) team {mu} tour {i + 1} column {j + 1}: "
                                f"not a monitoring point nor a repeat")
            for j in range(n + 1):                                     # (3g)
                low = th_a(row[j], row[j + 1])
                if Ta[mu][i][j] < low - tol:
                    errs.append(f"(3g) team {mu} tour {i + 1} leg {j + 1}: "
                                f"{Ta[mu][i][j]:.3f} < {low:.3f}")
            low = th_g(row[0], row[n + 1])                             # (3h)
            if tau_g < low - tol:
                errs.append(f"(3h) team {mu} tour {i + 1}: {tau_g:.3f} < {low:.3f}")
            if i < n - 1:                                              # (3i)
                low = th_g(row[n + 1], tuple(X[mu][i + 1][0]))
                if Tga[mu][i] < low - tol:
                    errs.append(f"(3i) team {mu} after tour {i + 1}: "
                                f"{Tga[mu][i]:.3f} < {low:.3f}")
        low = th_g(instance.starts[mu], tuple(X[mu][0][0]))            # (3j)
        if plan.theta_s[mu] < low - tol:
            errs.append(f"(3j) team {mu}: theta_s {plan.theta_s[mu]:.3f} < {low:.3f}")
        low = th_g(tuple(X[mu][n - 1][n + 1]), instance.finishes[mu])
        if plan.theta_f[mu] < low - tol:
            errs.append(f"(3j) team {mu}: theta_f {plan.theta_f[mu]:.3f} < {low:.3f}")

    t = visit_times(instance, params, plan)
    for k in np.flatnonzero(np.isnan(t)):                              # (3b)
        errs.append(f"(3b) point {k} is never visited")
    for p, q in instance.edges:                                        # (3k)
        if not t[p] + params.delta_o <= t[q] + tol:
            errs.append(f"(3k) edge {p}->{q}: t_q - t_p = {t[q] - t[p]:.3f} "
                        f"< delta_o = {params.delta_o}")
    return errs


def assumption_1_violations(instance: Instance, params: Params) -> List[int]:
    """Points with no feasible single-point tour from their ground projection."""
    out = []
    for k, q in enumerate(instance.points):
        r = params.pi(q)
        air = params.theta_a(r, q) + params.theta_a(q, r) + params.delta_a
        ground = params.theta_g(r, r) + params.delta_g
        if air > params.tau_a_max or ground > params.tau_a_max:
            out.append(k)
    return out


def tours(instance: Instance, plan: Plan) -> List[List[Tour]]:
    """The non-trivial rows of every team's plan, in order."""
    index = {p: k for k, p in enumerate(instance.points)}
    m, n = plan.Theta_g.shape
    X = plan.X.tolist()
    out = []
    for mu in range(m):
        team = []
        for i in range(n):
            row = [tuple(x) for x in X[mu][i]]
            if all(p == row[0] for p in row):
                continue
            visits = []
            for p in row[1:n + 1]:
                k = index.get(p)
                if k is not None and (not visits or visits[-1] != k):
                    visits.append(k)
            team.append(Tour(mu, i, row[0], visits, row[n + 1]))
        out.append(team)
    return out

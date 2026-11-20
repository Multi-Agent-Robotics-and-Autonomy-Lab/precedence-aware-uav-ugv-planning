"""Figures: the precedence graph, a plan in space, and a plan in time.

Each `plot_*` function draws on a matplotlib Axes (a new one if omitted) and
returns it. `theme="dark"` uses light ink on a transparent background, for
dark-mode pages.

Visual vocabulary (as in Fig. 1 of the paper): squares are release points,
filled pentagons the chosen collect points, hollow pentagons feasible collect
points that were not taken, solid lines UAV flights, dashed lines UGV routes,
and triangles the teams' start and finish positions.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from .problem import Instance, Params, Result, tours

TEAM_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple",
               "tab:red", "tab:brown", "tab:pink", "tab:cyan"]
METHOD_NAMES = {"alg1": "Alg. 1", "decoupled": "Decoupled baseline",
                "annealing": "Simulated annealing",
                "branch_and_cut": "Branch and cut"}
THEMES = {
    "light": dict(ink="black", muted="0.55", face="white",
                  depth=("0.80", "0.05")),
    "dark": dict(ink="#e6edf3", muted="#8b949e", face="#0d1117",
                 depth=("0.30", "0.97")),
}


def team_color(mu: int) -> str:
    return TEAM_COLORS[mu % len(TEAM_COLORS)]


# ============================ PANELS =========================================
def plot_precedence(instance: Instance, ax=None, theme: str = "light"):
    """The points and the precedence graph. Ordered points are shaded by
    longest-path depth (sources faint, sinks strong); points with no
    precedence edge are hollow."""
    ax, th = _axes(ax), THEMES[theme]
    P = instance.points
    for a, b in instance.edges:
        _arrow(ax, P[a], P[b], th["muted"], lw=0.9, rad=0.14, zorder=1)
    depth = longest_path_depth(instance)
    deepest = max(depth) or 1
    ordered = {k for e in instance.edges for k in e}
    lo, hi = (np.array(to_rgb(c)) for c in th["depth"])
    for k, p in enumerate(P):
        if k in ordered:
            shade = lo + (hi - lo) * depth[k] / deepest
            ax.scatter(p[0], p[1], s=36, color=shade, edgecolors=th["ink"],
                       linewidths=0.4, zorder=3)
        else:
            ax.scatter(p[0], p[1], s=30, facecolors="none",
                       edgecolors=th["muted"], linewidths=0.9, zorder=3)
    _endpoints(ax, instance, th)
    _frame(ax, instance, th, "Precedence graph")
    return ax


def plot_plan(instance: Instance, result: Result, ax=None,
              theme: str = "light", show_candidates: bool = True):
    """The plan in space (top view), one colour per team."""
    ax, th = _axes(ax), THEMES[theme]
    P = instance.points
    candidates = result.info.get("candidates") if show_candidates else None
    for team in tours(instance, result.plan):
        if not team:
            continue
        mu = team[0].team
        c = team_color(mu)
        ugv = [instance.starts[mu]]
        for k, tour in enumerate(team):
            ax.scatter([P[v][0] for v in tour.visits],
                       [P[v][1] for v in tour.visits], s=20, color=c, zorder=4)
            if candidates is not None and k < len(candidates[mu]):
                for cand in candidates[mu][k]:
                    if cand != tour.collect:
                        ax.scatter(cand[0], cand[1], s=110, marker="p",
                                   facecolors="none", edgecolors=c,
                                   linewidths=1.3, zorder=5)
            chain = [tour.release] + [P[v] for v in tour.visits] + [tour.collect]
            for a, b in zip(chain, chain[1:]):
                _arrow(ax, a, b, c, lw=1.4, zorder=3)
            ax.scatter(tour.release[0], tour.release[1], s=170, marker="s",
                       color=c, edgecolors=th["ink"], linewidths=0.8, zorder=6)
            ax.scatter(tour.collect[0], tour.collect[1], s=95, marker="p",
                       color=c, edgecolors=th["ink"], linewidths=0.8, zorder=7)
            ugv += [tour.release, tour.collect]
        ugv.append(instance.finishes[mu])
        for a, b in zip(ugv, ugv[1:]):
            _arrow(ax, a, b, c, lw=1.4, dashed=True, zorder=2)
    _endpoints(ax, instance, th)
    _frame(ax, instance, th, METHOD_NAMES.get(result.method, result.method))
    return ax


def plot_timeline(instance: Instance, result: Result,
                  params: Optional[Params] = None, ax=None,
                  theme: str = "light"):
    """The plan in time: one row per team, wide bars while the UAV is
    airborne, thin bars while it rides the UGV, dots at the visit times and
    an arrow for every precedence edge. Precedence holds exactly when every
    arrow points right."""
    params = params or Params()
    if ax is None:
        ax = plt.subplots(figsize=(9, 0.6 * instance.m + 1.6))[1]
    th = THEMES[theme]
    plan, m = result.plan, instance.m
    team_tours = tours(instance, plan)
    team_of = {v: tr.team for team in team_tours for tr in team
               for v in tr.visits}
    for mu, sched in enumerate(tour_schedule(plan, params)):
        c = team_color(mu)
        rows = {tr.row for tr in team_tours[mu]}
        for i, (rel, land, nxt) in enumerate(sched):
            if i in rows:
                ax.barh(-mu, land - rel, left=rel, height=0.42, color=c,
                        alpha=0.35, zorder=2)
                end = nxt if i + 1 in rows else result.team_times[mu]
                ax.barh(-mu, end - land, left=land, height=0.12, color=c,
                        alpha=0.35, zorder=2)
        if not rows:
            ax.barh(-mu, result.team_times[mu], left=0.0, height=0.12,
                    color=c, alpha=0.35, zorder=2)
    t = result.visit_times
    fade = min(1.0, max(0.15, 40 / max(1, len(instance.edges))))
    for p, q in instance.edges:
        cross = team_of.get(p) != team_of.get(q)
        _arrow(ax, (t[p], -team_of.get(p, 0)), (t[q], -team_of.get(q, 0)),
               th["ink"] if cross else th["muted"], lw=0.9 if cross else 0.6,
               alpha=fade, zorder=3)
    for v, mu in team_of.items():
        ax.scatter(t[v], -mu, s=16, color=team_color(mu), zorder=5)
    ax.set_yticks([-mu for mu in range(m)])
    ax.set_yticklabels([f"team {mu + 1}" for mu in range(m)], color=th["ink"])
    ax.set_xlabel("time (s)", color=th["ink"])
    ax.set_ylim(-m + 0.4, 0.6)
    ax.set_xlim(0, result.makespan * 1.01)
    ax.tick_params(colors=th["ink"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["ink"])
    ax.set_facecolor("none")
    return ax


def figure(instance: Instance, result: Result,
           panels: Sequence[str] = ("precedence", "plan"),
           params: Optional[Params] = None, theme: str = "light",
           size: float = 4.6):
    """A figure with the requested panels side by side ("precedence",
    "plan", "timeline"). Returns the matplotlib Figure."""
    widths = [2.0 if p == "timeline" else 1.0 for p in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(size * sum(widths), size),
                             gridspec_kw={"width_ratios": widths},
                             squeeze=False)
    for ax, name in zip(axes[0], panels):
        if name == "precedence":
            plot_precedence(instance, ax, theme)
        elif name == "plan":
            plot_plan(instance, result, ax, theme)
        elif name == "timeline":
            plot_timeline(instance, result, params, ax, theme)
        else:
            raise ValueError(f"unknown panel {name!r}")
    fig.patch.set_facecolor(THEMES[theme]["face"])
    if theme == "dark":
        fig.patch.set_alpha(0.0)
    fig.tight_layout()
    return fig


# ============================ HELPERS ========================================
def tour_schedule(plan, params: Params):
    """Per team and row: (release, landing, next release) times, Eq. (2)."""
    m, n = plan.Theta_g.shape
    out = []
    for mu in range(m):
        rows, release = [], plan.theta_s[mu]
        for i in range(n):
            airborne = max(plan.Theta_a[mu, i].sum(), plan.Theta_g[mu, i])
            gap = (max(plan.Theta_ga[mu, i], params.tau_c(airborne))
                   if i < n - 1 else 0.0)
            rows.append((release, release + airborne,
                         release + airborne + gap))
            release += airborne + gap
        out.append(rows)
    return out


def longest_path_depth(instance: Instance):
    """Depth of every point in the precedence graph: 0 at the sources."""
    succ, waiting = instance.successors(), [len(p) for p in
                                            instance.predecessors()]
    depth = [0] * instance.n
    stack = [k for k in range(instance.n) if waiting[k] == 0]
    while stack:
        u = stack.pop()
        for v in succ[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            waiting[v] -= 1
            if waiting[v] == 0:
                stack.append(v)
    return depth


def _axes(ax):
    return plt.subplots(figsize=(5, 5))[1] if ax is None else ax


def _arrow(ax, a, b, color, lw=1.0, dashed=False, rad=None, alpha=1.0,
           zorder=2):
    if abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9:
        return
    props = dict(arrowstyle="->", lw=lw, color=color, shrinkA=2, shrinkB=2,
                 linestyle="--" if dashed else "-", alpha=alpha)
    if rad is not None:
        props["connectionstyle"] = f"arc3,rad={rad}"
    ax.annotate("", xy=(b[0], b[1]), xytext=(a[0], a[1]), arrowprops=props,
                zorder=zorder)


def _endpoints(ax, instance: Instance, th):
    for p in set(instance.starts):
        ax.scatter(p[0], p[1], s=140, marker="^", color=th["ink"], zorder=8)
    for p in set(instance.finishes):
        ax.scatter(p[0], p[1], s=140, marker="v", color=th["ink"], zorder=8)


def _frame(ax, instance: Instance, th, title: str):
    pad = 0.06 * max(instance.size)
    ax.set_xlim(-pad, instance.size[0] + pad)
    ax.set_ylim(-pad, instance.size[1] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(th["muted"])
    ax.set_facecolor("none")
    ax.set_title(title, color=th["ink"])

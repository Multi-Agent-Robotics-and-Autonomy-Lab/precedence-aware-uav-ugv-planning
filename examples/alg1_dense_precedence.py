"""Dense precedence: one connected ordering, still shared by all teams.

With three edges per point the precedence graph is a single weakly connected
component, so assigning whole components to teams would leave all but one
idle. Alg. 1 places ordered points on different teams and times each visit
after its predecessors. Saves outputs/alg1_dense_precedence.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from uav_ugv_precedence import Params, alg1, decoupled, make_instance, plotting, problem  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main():
    instance = make_instance(n=40, m=4, d=3.0, seed=2)
    params = Params()
    result = alg1.solve(instance, params)

    team_of = {v: tour.team for team in problem.tours(instance, result.plan)
               for tour in team for v in tour.visits}
    crossing = sum(team_of[p] != team_of[q] for p, q in instance.edges)
    print(f"{len(instance.edges)} precedence edges, "
          f"{len(decoupled.components(instance))} weakly connected component(s)")
    print(f"feasible: {result.feasible}   makespan: {result.makespan:.0f} s")
    print(f"points per team: "
          f"{[sum(1 for v in team_of if team_of[v] == mu) for mu in range(instance.m)]}")
    print(f"edges between points of different teams: {crossing}")

    OUT.mkdir(exist_ok=True)
    fig = plotting.figure(instance, result, panels=("plan", "timeline"),
                          params=params)
    fig.savefig(OUT / "alg1_dense_precedence.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT / 'alg1_dense_precedence.png'}")


if __name__ == "__main__":
    main()

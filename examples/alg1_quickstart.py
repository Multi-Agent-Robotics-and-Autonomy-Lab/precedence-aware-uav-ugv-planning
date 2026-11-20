"""Plan a random mission with Alg. 1 and draw it.

25 monitoring points, 4 UAV-UGV teams, the parameters of Table I.
Saves outputs/alg1_quickstart.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from uav_ugv_precedence import Params, alg1, make_instance, plotting, problem  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main():
    instance = make_instance(n=25, m=4, d=0.5, seed=0)
    params = Params()
    result = alg1.solve(instance, params)

    print(f"{instance.n} points, {len(instance.edges)} precedence edges, "
          f"{instance.m} teams")
    print(f"feasible: {result.feasible}   makespan: {result.makespan:.0f} s   "
          f"planned in {1000 * result.runtime:.1f} ms")
    for team in problem.tours(instance, result.plan):
        for tour in team:
            print(f"  team {tour.team + 1}, tour {tour.row + 1}: "
                  f"visits {tour.visits}")

    OUT.mkdir(exist_ok=True)
    fig = plotting.figure(instance, result, params=params)
    fig.savefig(OUT / "alg1_quickstart.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT / 'alg1_quickstart.png'}")


if __name__ == "__main__":
    main()

"""The decoupled precedence baseline on a random mission.

Each weakly connected component of the precedence graph goes whole to one
team (longest workload first, to the least-loaded team), then each team's
points are cut into tours and timed by Alg. 1's subroutines.
Saves outputs/decoupled_baseline.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from uav_ugv_precedence import Params, decoupled, make_instance, plotting  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main():
    instance = make_instance(n=40, m=4, d=0.5, seed=3)
    params = Params()
    result = decoupled.solve(instance, params)

    comps = result.info["components"]
    print(f"{len(comps)} components, largest has {max(map(len, comps))} points")
    print(f"points per team: {[len(t) for t in result.info['teams']]}")
    print(f"feasible: {result.feasible}   makespan: {result.makespan:.0f} s   "
          f"planned in {1000 * result.runtime:.1f} ms")

    OUT.mkdir(exist_ok=True)
    fig = plotting.figure(instance, result, params=params)
    fig.savefig(OUT / "decoupled_baseline.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT / 'decoupled_baseline.png'}")


if __name__ == "__main__":
    main()

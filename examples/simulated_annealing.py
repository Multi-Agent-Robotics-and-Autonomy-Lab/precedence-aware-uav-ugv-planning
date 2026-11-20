"""Simulated annealing on a tiny mission.

10 points, 2 teams, 15 000 steps from t_max = 500.
Saves outputs/simulated_annealing.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from uav_ugv_precedence import Params, annealing, make_instance, plotting  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main():
    instance = make_instance(n=10, m=2, d=0.5, seed=1)
    params = Params()
    result = annealing.solve(instance, params, steps=15_000, t_max=500.0, seed=0)

    print(f"warm start: {result.info['initial_makespan']:.0f} s   "
          f"best found: {result.makespan:.0f} s   feasible: {result.feasible}")
    print(f"{result.info['steps']} steps, acceptance {result.info['accept_rate']:.2f}, "
          f"final temperature {result.info['final_temperature']:.0f}, "
          f"{result.runtime:.1f} s")

    OUT.mkdir(exist_ok=True)
    fig = plotting.figure(instance, result, params=params)
    fig.savefig(OUT / "simulated_annealing.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT / 'simulated_annealing.png'}")


if __name__ == "__main__":
    main()

"""Robustness margins: plan the same mission with and without them.

delta_a and delta_g keep every tour within the UAV's endurance even if its
flight or the UGV's drive takes that much longer; delta_o keeps every
precedence edge at least that far apart in time. The script checks both on
the returned plan. Saves outputs/alg1_robust_margins.png (plan in time).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from uav_ugv_precedence import Params, alg1, make_instance, plotting  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main():
    instance = make_instance(n=30, m=3, d=1.0, seed=0)
    for params in (Params(), Params(delta_a=60.0, delta_g=60.0, delta_o=300.0)):
        result = alg1.solve(instance, params)
        t = result.visit_times
        separation = round(min(t[q] - t[p] for p, q in instance.edges), 6) + 0.0
        flight = result.plan.Theta_a.sum(axis=2).max()
        print(f"delta_a = delta_g = {params.delta_a:.0f} s, "
              f"delta_o = {params.delta_o:.0f} s")
        print(f"  feasible: {result.feasible}   makespan: {result.makespan:.0f} s")
        print(f"  tightest precedence separation: {separation:.0f} s")
        print(f"  longest tour flight: {flight:.0f} s "
              f"(limit {params.tau_a_max - params.delta_a:.0f} s)")

    OUT.mkdir(exist_ok=True)
    fig = plotting.figure(instance, result, panels=("timeline",),
                          params=params, size=3.4)
    fig.savefig(OUT / "alg1_robust_margins.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT / 'alg1_robust_margins.png'}")


if __name__ == "__main__":
    main()

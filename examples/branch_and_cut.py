"""Branch and cut on a tiny mission (needs gurobipy).

3 points and 1 team: problem (3) solved as an MIQCP to a 5% optimality gap.
Release and collect points may sit anywhere on the ground. Saves
outputs/branch_and_cut.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from uav_ugv_precedence import Params, make_instance, plotting  # noqa: E402


def main():
    try:
        from uav_ugv_precedence import branch_and_cut
        import gurobipy  # noqa: F401
    except ImportError:
        print("this example needs gurobipy: pip install gurobipy")
        return

    instance = make_instance(n=3, m=1, d=0.5, seed=2)
    params = Params()
    result = branch_and_cut.solve(instance, params, time_limit=60)

    print(f"makespan: {result.makespan:.0f} s   feasible: {result.feasible}   "
          f"optimality gap: {100 * result.info['mip_gap']:.1f}%   "
          f"solved in {result.runtime:.1f} s")
    for k in range(instance.n):
        print(f"  point {k} visited at {result.visit_times[k]:.0f} s")

    out = Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(exist_ok=True)
    fig = plotting.figure(instance, result, params=params)
    fig.savefig(out / "branch_and_cut.png", dpi=150, bbox_inches="tight")
    print(f"saved {out / 'branch_and_cut.png'}")


if __name__ == "__main__":
    main()

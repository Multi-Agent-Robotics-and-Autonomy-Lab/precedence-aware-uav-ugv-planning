"""Regenerate the README figures, in light and dark versions.

The mission is the one drawn in Fig. 1 of the paper: n = 25, m = 4, d = 0.3,
seed 6, with the UGV at s_g = 10 m/s instead of Table I's 2.5 m/s. At
s_g >= s_H every tour lands below its last waypoint, so no landing leg is
drawn back onto a point the UAV already visited.

    python docs/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from uav_ugv_precedence import Params, alg1, make_instance, plotting  # noqa: E402

DOCS = Path(__file__).resolve().parent


def main():
    instance = make_instance(n=25, m=4, d=0.3, seed=6)
    params = Params(s_g=10.0)
    result = alg1.solve(instance, params)
    assert result.feasible, result.violations
    for theme in ("light", "dark"):
        fig = plotting.figure(instance, result, params=params, theme=theme)
        fig.savefig(DOCS / f"plan_{theme}.png", dpi=160, bbox_inches="tight",
                    transparent=True)
        plt.close(fig)
        fig = plotting.figure(instance, result, panels=("timeline",),
                              params=params, theme=theme, size=3.2)
        fig.savefig(DOCS / f"timeline_{theme}.png", dpi=160,
                    bbox_inches="tight", transparent=True)
        plt.close(fig)
    print(f"wrote {sorted(p.name for p in DOCS.glob('*.png'))}")


if __name__ == "__main__":
    main()

"""Build a mission by hand and change the vehicle model.

A site whose perimeter must be surveyed before its interior: every perimeter
point precedes every interior point. Two teams start at different corners
and finish at the opposite ones. The UAV battery is swapped in a fixed time
instead of recharged in proportion to the flight, by overriding tau_c, one of
the model functions the paper leaves open. Saves
outputs/alg1_custom_instance.png.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from uav_ugv_precedence import Instance, Params, alg1, plotting, problem  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


@dataclass(frozen=True)
class BatterySwap(Params):
    """Table I, but the UAV gets a fresh battery in `swap_time` seconds."""

    swap_time: float = 120.0

    def tau_c(self, airborne):
        return self.swap_time if airborne > 0 else 0.0


def main():
    altitude = 80.0
    perimeter = [(x, y, altitude) for x, y in
                 [(800, 800), (2000, 600), (3200, 800), (3400, 2000),
                  (3200, 3200), (2000, 3400), (800, 3200), (600, 2000)]]
    interior = [(x, y, altitude) for x, y in
                [(1500, 1500), (2500, 1500), (2500, 2500), (1500, 2500)]]
    points = perimeter + interior
    edges = [(p, len(perimeter) + q)
             for p in range(len(perimeter)) for q in range(len(interior))]
    instance = Instance(points, edges,
                        starts=[(0, 0, 0), (4000, 0, 0)],
                        finishes=[(4000, 4000, 0), (0, 4000, 0)])
    params = BatterySwap(s_g=5.0, delta_o=60.0)

    result = alg1.solve(instance, params)
    print(f"feasible: {result.feasible}   makespan: {result.makespan:.0f} s")
    for team in problem.tours(instance, result.plan):
        for tour in team:
            kinds = ["interior" if v >= len(perimeter) else "perimeter"
                     for v in tour.visits]
            print(f"  team {tour.team + 1}, tour {tour.row + 1}: "
                  f"{list(zip(tour.visits, kinds))}")

    OUT.mkdir(exist_ok=True)
    fig = plotting.figure(instance, result, params=params)
    fig.savefig(OUT / "alg1_custom_instance.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT / 'alg1_custom_instance.png'}")


if __name__ == "__main__":
    main()

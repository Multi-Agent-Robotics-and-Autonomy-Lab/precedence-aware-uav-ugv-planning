import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from uav_ugv_precedence import Params, alg1, decoupled, make_instance, plotting  # noqa: E402


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("method", [alg1, decoupled])
def test_figures_render(tmp_path, theme, method):
    inst = make_instance(20, 3, 1.0, seed=1)
    res = method.solve(inst, Params())
    fig = plotting.figure(inst, res, panels=("precedence", "plan", "timeline"),
                          theme=theme)
    path = tmp_path / f"{theme}.png"
    fig.savefig(path, dpi=40)
    plt.close(fig)
    assert path.stat().st_size > 0


def test_unknown_panel_is_an_error():
    inst = make_instance(5, 1, 0.5, seed=0)
    with pytest.raises(ValueError):
        plotting.figure(inst, alg1.solve(inst), panels=("map",))


def test_depth_is_zero_at_sources_and_grows_along_edges():
    inst = make_instance(30, 1, 2.0, seed=3)
    depth = plotting.longest_path_depth(inst)
    assert all(depth[b] >= depth[a] + 1 for a, b in inst.edges)
    assert all(depth[k] == 0 for k in range(inst.n)
               if not inst.predecessors()[k])

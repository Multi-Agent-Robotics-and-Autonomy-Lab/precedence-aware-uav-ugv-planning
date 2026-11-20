"""Precedence-aware robust planning for energy-constrained UAV-UGV teams.

Alg. 1 of "Robust and Scalable UAV-UGV Planning for Aerial Monitoring under
Energy and Precedence Constraints" (Er and Yazicioglu, 2026) and the baselines
it is compared against. Every planner module exposes
`solve(instance, params) -> Result`.
"""

from .problem import Instance, Params, Plan, Result, make_instance

__version__ = "1.0.0"
__all__ = ["Instance", "Params", "Plan", "Result", "make_instance"]

"""Versioned goal-first STOP teacher for the second DAgger round.

The original teacher and its provenance remain unchanged for prior runs.
"""
import numpy as np

from .recovery import RECOVERY_RULE as ORIGINAL_RULE
from .recovery import RouteExpert as OriginalRouteExpert

RECOVERY_RULE = {**ORIGINAL_RULE, "name": "goal_first_stop_2m_ordered_waypoints_v2",
                 "goal_radius_m": 2.0, "movement_final_follower_radius_m": ORIGINAL_RULE["goal_radius_m"],
                 "stop_priority": "STOP before consulting route progress whenever geodesic task-goal distance <2m"}


class RouteExpert(OriginalRouteExpert):
    def query(self):
        distance = float(self.sim.geodesic_distance(self.sim.get_agent_state().position, self.goal))
        if np.isfinite(distance) and distance < RECOVERY_RULE["goal_radius_m"]:
            return 0, {"route_index": self.index, "target_distance_m": distance,
                       "task_goal_distance_m": distance, "final_waypoint": self.index == len(self.route) - 1,
                       "goal_stop_override": True, "error": None}
        action, detail = super().query()
        detail.update(task_goal_distance_m=distance if np.isfinite(distance) else None, goal_stop_override=False)
        if action == 0:
            raise AssertionError("Movement expert returned STOP outside the declared 2m region")
        return action, detail

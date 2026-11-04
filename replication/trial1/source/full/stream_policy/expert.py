"""StreamVLN's DAgger expert: Habitat's ShortestPathFollower toward the episode's reference waypoints (env rolevln).

As in streamvln_dagger.py @ e48f6ff7, intermediate waypoints use goal radius 1.8 m and the final one
0.25 m, and "reached" means the follower would STOP. Two deviations, both for label validity: points
are snapped onto the agent's navmesh island (sim/recovery.py does the same), and a follower failure
returns None instead of STOP, so it can never be mistaken for reaching a waypoint. The follower is
rebuilt for every query because the executed actions need not be the expert's. Labels only: a query
never moves the agent.
"""
import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_sim.errors import GreedyFollowerError

from .controller import STOP, DaggerRule


class WaypointExpert:
    def __init__(self, sim, episode, rule=DaggerRule()):
        self.sim = sim
        island = sim.pathfinder.get_island(sim.get_agent_state().position)
        self.points, self.snap_distances = [], []
        for point in episode.reference_path:
            snapped = np.asarray(sim.pathfinder.snap_point(point, island_index=island), dtype=np.float32)
            if not np.isfinite(snapped).all():
                snapped = np.asarray(point, dtype=np.float32)
            self.points.append(snapped)
            self.snap_distances.append(float(np.linalg.norm(snapped - np.asarray(point))))
        self.followers = {final: ShortestPathFollower(sim, radius, return_one_hot=False, stop_on_error=False)
                          for final, radius in ((False, rule.waypoint_radius_m), (True, rule.goal_radius_m))}
        self.errors = 0

    def label(self, index):
        """Expert action toward waypoint `index` from the actual pose, or None if the follower fails."""
        follower = self.followers[index == len(self.points) - 1]
        follower._build_follower()
        follower._follower.reset()
        try:
            action = follower.get_next_action(self.points[index])
        except GreedyFollowerError:
            self.errors += 1
            return None
        return None if action is None else int(action)

    def reached(self, index):
        return self.label(index) == STOP

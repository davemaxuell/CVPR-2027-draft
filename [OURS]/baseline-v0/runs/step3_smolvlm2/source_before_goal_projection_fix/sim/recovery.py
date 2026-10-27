"""Privileged, ordered-reference-route expert used only to construct DAgger labels."""
import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_sim.errors import GreedyFollowerError


RECOVERY_RULE = {
    "name": "ordered_reference_waypoints_v1",
    "route": "actual positions from validated group-A expert replay, followed by episode.goals[0].position",
    "progress": "advance only consecutive reached waypoints; never regress or jump to a later nearest point",
    "waypoint_radius_m": 0.5,
    "goal_radius_m": 1.0,
    "max_navmesh_snap_m": 0.5,
    "follower": "Habitat ShortestPathFollower; reset each query because expert actions may not be executed",
    "errors": "invalid label, excluded from SFT; never convert follower failure to STOP",
    "privileged_inputs": "reference route, goal, simulator pose and navigation mesh; labels only",
}


class RouteExpert:
    def __init__(self, sim, episode, reference_positions=None):
        self.sim = sim
        raw = list(reference_positions if reference_positions is not None else episode.reference_path) + [episode.goals[0].position]
        island = sim.pathfinder.get_island(sim.get_agent_state().position)
        self.route, self.snap_distances = [], []
        for point in raw:
            snapped = np.asarray(sim.pathfinder.snap_point(point, island_index=island), dtype=np.float32)
            distance = float(np.linalg.norm(snapped - np.asarray(point)))
            if not np.isfinite(snapped).all() or distance > RECOVERY_RULE["max_navmesh_snap_m"]:
                raise ValueError(f"Reference waypoint cannot be snapped safely: {point}, distance={distance}")
            if not self.route or np.linalg.norm(snapped - self.route[-1]) > 1e-4:
                self.route.append(snapped)
                self.snap_distances.append(distance)
        self.goal = np.asarray(episode.goals[0].position, dtype=np.float32)
        self.index = 0
        self.followers = {
            False: ShortestPathFollower(sim, RECOVERY_RULE["waypoint_radius_m"], return_one_hot=False, stop_on_error=False),
            True: ShortestPathFollower(sim, RECOVERY_RULE["goal_radius_m"], return_one_hot=False, stop_on_error=False),
        }

    def query(self):
        position = self.sim.get_agent_state().position
        last = len(self.route) - 1
        while self.index < last:
            distance = float(self.sim.geodesic_distance(position, self.route[self.index]))
            if distance >= RECOVERY_RULE["waypoint_radius_m"]:
                break
            self.index += 1
        final = self.index == last
        target = self.route[self.index]
        distance = float(self.sim.geodesic_distance(position, target))
        detail = {"route_index": self.index, "target_distance_m": distance if np.isfinite(distance) else None,
                  "final_waypoint": final, "error": None}
        if not np.isfinite(distance):
            detail["error"] = "unreachable_waypoint"
            return None, detail
        if final and distance < RECOVERY_RULE["goal_radius_m"]:
            # Also check the unsnapped task goal, whose distance determines success.
            if float(self.sim.geodesic_distance(position, self.goal)) < RECOVERY_RULE["goal_radius_m"]:
                return 0, detail
        follower = self.followers[final]
        follower._build_follower()
        follower._follower.reset()
        try:
            action = int(follower.get_next_action(target))
        except GreedyFollowerError:
            detail["error"] = "greedy_follower_error"
            return None, detail
        if action not in (1, 2, 3):
            detail["error"] = "unexpected_follower_stop" if action == 0 else "invalid_follower_action"
            return None, detail
        return action, detail

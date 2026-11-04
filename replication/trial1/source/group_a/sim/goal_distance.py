"""CPU-only task-goal distances on the same saved navmeshes as Habitat."""
import hashlib
from pathlib import Path

import habitat_sim
import numpy as np

DATA = Path("/data/team_a/dave-workspace/RoleVLN/data")


class GoalDistances:
    def __init__(self):
        self.scene = None
        self.pathfinder = None
        self.cache = {}
        self.mesh_hashes = {}

    def distances(self, scene, positions, goal):
        if scene != self.scene:
            path = DATA / "scene_datasets/mp3d" / scene / f"{scene}.navmesh"
            self.pathfinder = habitat_sim.PathFinder()
            assert self.pathfinder.load_nav_mesh(str(path)), path
            self.scene = scene
            self.cache = {}
            self.mesh_hashes[scene] = hashlib.sha256(path.read_bytes()).hexdigest()
        goal = np.asarray(goal, dtype=np.float32)
        goal_key = tuple(goal.tolist())
        query = habitat_sim.MultiGoalShortestPath()
        query.requested_ends = goal[None]
        output = []
        for position in positions:
            position = np.asarray(position, dtype=np.float32)
            key = (goal_key, tuple(position.tolist()))
            if key not in self.cache:
                query.requested_start = position
                self.pathfinder.find_path(query)
                self.cache[key] = float(query.geodesic_distance)
            output.append(self.cache[key])
        return np.asarray(output, dtype=np.float32)

"""ETPNav's evaluation env plus the queries the navigation explorer needs (tools/log_etpnav_steps.py).

Runs inside habitat's VectorEnv workers, so this module must stay importable on its own (log_etpnav_steps.py
puts tools/ on sys.path, which the forkserver workers inherit).
"""
import os

import cv2
import numpy as np
import quaternion
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.common.environments import VLNCEDaggerEnv

# World-aligned cube faces rendered at every map node, named after the Matterport skybox images they stand in
# for (_skybox<i>_sami.jpg); orientations are habitat sensor [pitch, yaw, roll] with the agent at heading 0
# (facing -z). The explorer pairs them with the fixed rotation CUBE_ROT (see log_etpnav_steps.py).
CUBE_FACES = {
    'cube0': [np.pi / 2, 0.0, 0.0],      # up
    'cube1': [0.0, 0.0, 0.0],            # -z
    'cube2': [0.0, -np.pi / 2, 0.0],     # +x
    'cube3': [0.0, np.pi, 0.0],          # +z
    'cube4': [0.0, np.pi / 2, 0.0],      # -x
    'cube5': [-np.pi / 2, 0.0, 0.0],     # down
}


@baseline_registry.register_env(name="VLNCEExplorerEnv")
class VLNCEExplorerEnv(VLNCEDaggerEnv):

    def explorer_episode(self):
        """Ground truth of the current episode in habitat coordinates."""
        ep = self._env.current_episode
        ins = ep.instruction
        return {
            'episode_id': str(ep.episode_id),
            'trajectory_id': str(getattr(ep, 'trajectory_id', '')),
            'scene': os.path.splitext(os.path.basename(ep.scene_id))[0],
            'instruction': ins.instruction_text,
            'instruction_id': str(getattr(ins, 'instruction_id', '') or ''),
            'language': getattr(ins, 'language', None),
            'tokens': list(getattr(ins, 'instruction_tokens', []) or []),
            'start': [float(x) for x in ep.start_position],
            'start_rotation': [float(x) for x in ep.start_rotation],
            'goal': [float(x) for x in ep.goals[0].position],
            'reference_path': [[float(x) for x in p] for p in ep.reference_path],
        }

    def explorer_geodesics(self, points, targets):
        """Geodesic distance from each point to each target (inf when unreachable). Points off the navmesh
        (ghost waypoints are estimates) are snapped by the pathfinder."""
        sim = self._env.sim
        return [[float(sim.geodesic_distance(p, t)) for t in targets] for p in points]

    def explorer_cube(self, prefix, size=None):
        """Render the six world-aligned cube faces at the agent's position as <prefix>_<i>.jpg."""
        sim = self._env.sim
        state = sim.get_agent_state()
        identity = quaternion.quaternion(1, 0, 0, 0)
        obs = sim.get_observations_at(state.position, identity, keep_agent_at_new_pose=False)
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        for i, uuid in enumerate(CUBE_FACES):
            img = obs[uuid][:, :, :3][:, :, ::-1]
            cv2.imwrite(f'{prefix}_{i}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return [float(x) for x in state.position]

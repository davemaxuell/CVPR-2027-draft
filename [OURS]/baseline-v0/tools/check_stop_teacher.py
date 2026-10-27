"""Check goal priority without changing route motion or deleting real turns."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.recovery import RouteExpert as Original
from sim.recovery_stop_v2 import RouteExpert
from tools.prepare_stop_data import pose_retained


def main():
    pose = np.array([0., 0., 0.])
    sim = SimpleNamespace(distance=1.5)
    sim.get_agent_state = lambda: SimpleNamespace(position=pose)
    sim.geodesic_distance = lambda a, b: sim.distance
    expert = RouteExpert.__new__(RouteExpert)
    expert.sim, expert.goal, expert.index, expert.route = sim, pose.copy(), 0, [pose.copy(), pose.copy() + 10]
    original = Original.query
    calls = []
    def movement(self):
        calls.append(self.sim.distance)
        return (1 if np.isfinite(self.sim.distance) else None), {"route_index": 0, "final_waypoint": False, "error": None}
    try:
        Original.query = movement
        for distance in (0., 1., 1.999):
            sim.distance = distance
            action, detail = expert.query()
            assert action == 0 and detail["goal_stop_override"] and not detail["final_waypoint"]
        assert not calls
        sim.distance = 2.
        assert expert.query()[0] == 1
        sim.distance = 3.
        assert expert.query()[0] == 1
        sim.distance = np.inf
        assert expert.query()[0] is None
        assert calls[:2] == [2., 3.]
    finally:
        Original.query = original
    assert np.array_equal(pose, [0, 0, 0])
    stationary = {"position": [[0, 0, 0]] * 20, "rotation_wxyz": [[1, 0, 0, 0]] * 20}
    assert len(pose_retained(stationary, [2] * 20)) == 4
    rotations = [[float(np.cos(t)), 0, float(np.sin(t)), 0] for t in np.arange(20) * np.pi / 24]
    turning = {"position": [[0, 0, 0]] * 20, "rotation_wxyz": rotations}
    assert len(pose_retained(turning, [2] * 20)) == 20
    report = {"status": "passed", "STOP_inside_2m_before_route_progress": True,
              "strict_2m_boundary": True, "movement_teacher_unchanged_outside_2m": True,
              "unreachable_is_not_STOP": True, "expert_query_preserves_pose": True,
              "repeated_stuck_frames_capped": True, "turning_views_preserved": True}
    (ROOT / "runs/step3b_smolvlm2_stop/check_teacher.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

"""Native preprocessing/bias parity and real simulator branch/replay checks."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for key in ("MAGNUM_LOG", "HABITAT_SIM_LOG"):
    os.environ.setdefault(key, "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")

import habitat
import numpy as np
import quaternion
import torch
from habitat.core.registry import registry
from habitat.datasets import make_dataset
from PIL import Image

from policy.common import past_frame_indices
from policy.experimental_reference import ExperimentalReference
from policy.online import OnlineInputs
from policy.smolvlm2 import SmolVLM2Inputs
from sim.config import make_ce_config, scene_group
from sim.episodes import select_episodes
from sim.transition_branch import movement_branches, pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/step4_smolvlm2_refined512/check_runtime.json")
    args = ap.parse_args()
    torch.manual_seed(4)
    policy = ExperimentalReference(ROOT / "runs/step4_smolvlm2_refined512/reference")
    native = SmolVLM2Inputs(policy.cfg)
    assert not any(p.requires_grad for p in policy.model.parameters())
    assert not any(m.training for m in policy.model.modules())
    counts = {"episodes": 0, "real_steps_compared": 0, "branch_targets": 0, "blocked_forward_branches": 0,
              "native_input_parity_cases": 0, "decoder_parity_cases": 0, "max_turn_translation_m": 0.0}
    for name in ("r2r", "rxr"):
        cfg = make_ce_config(name, "train", scenes=scene_group("A"), gpu=0)
        full = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        chosen = select_episodes(full.episodes, 4, 4)
        def make_env():
            ds = registry.get_dataset(cfg.habitat.dataset.type)(config=None)
            ds.episodes = chosen
            return habitat.Env(config=cfg, dataset=ds)
        sequence = [1] * 80 + [2, 2, 1, 3, 1, 3, 0]
        def observation_record(env, obs):
            return {"pose": pose(env.sim), "rgb": hashlib.sha256(np.ascontiguousarray(obs["rgb"]).tobytes()).hexdigest(),
                    "metrics": dict(env.get_metrics()), "elapsed": env._elapsed_steps, "ended": env.episode_over,
                    "collided": bool(env.sim.previous_step_collided)}
        traces = {}
        baseline = make_env()
        try:
            for _ in chosen:
                obs = baseline.reset()
                eid = str(baseline.current_episode.episode_id)
                traces[eid] = [observation_record(baseline, obs)]
                for action in sequence:
                    obs = baseline.step(action)
                    traces[eid].append(observation_record(baseline, obs))
        finally:
            baseline.close()
        branched = make_env()
        try:
            for _ in chosen:
                b = branched.reset()
                trace = traces[str(branched.current_episode.episode_id)]
                assert branched.current_episode.scene_id.split("/")[-2] in scene_group("A")
                frames, actions = [], []
                online = OnlineInputs(policy.inputs, policy.device)
                # Long FORWARD prefix forces wall contacts; turns test signed yaw.
                for t, action in enumerate(sequence):
                    assert observation_record(branched, b) == trace[t], (name, t, "prefix drift")
                    frames.append(Image.fromarray(np.array(b["rgb"], copy=True)))
                    if counts["episodes"] == 0:
                        online.append(frames[-1])
                        if t in (0, 3, 8, 40, 80):
                            step = {"instruction": branched.current_episode.instruction.instruction_text,
                                    "executed_actions": list(actions), "t": t,
                                    "images": [frames[i] for i in past_frame_indices(t, policy.cfg) + [t]]}
                            expected = native([step], with_targets=False)
                            saved = policy.inputs([step], with_targets=False)
                            live = online(step["instruction"], actions)
                            assert expected.keys() == saved.keys() == live.keys()
                            for key in expected:
                                assert torch.equal(expected[key].cpu(), saved[key].cpu()), ("saved processor", key)
                                assert torch.equal(expected[key].cpu(), live[key].cpu()), ("online processor", key)
                            counts["native_input_parity_cases"] += 1
                            raw, biased = policy.raw_action_scores(live), policy.action_scores(live)
                            shifted = raw.clone(); shifted[:, 0] -= 0.25
                            assert torch.equal(shifted, biased)
                            assert torch.equal(biased[:, 1:], raw[:, 1:])
                            assert int(biased.argmax(1)) == int(np.argmax(shifted[0].cpu().numpy()))
                            counts["decoder_parity_cases"] += 1
                    branch_by_action = {}
                    if t % 8 == 3 or t >= 80:
                        root = pose(branched.sim)
                        root_rgb = np.array(b["rgb"], copy=True)
                        root_collision = branched.sim.previous_step_collided
                        for branch in movement_branches(branched):
                            branch_by_action[branch["action"]] = branch
                            counts["branch_targets"] += 1
                            if branch["action"] == 1 and branch["collided"] and branch["displacement_m"] < 1e-5:
                                counts["blocked_forward_branches"] += 1
                            if branch["action"] in (2, 3):
                                q0 = np.quaternion(*root["rotation_wxyz"])
                                q1 = np.quaternion(*branch["next_pose"]["rotation_wxyz"])
                                delta = q0.conjugate() * q1
                                if delta.w < 0: delta = -delta
                                yaw = np.degrees(2 * np.arctan2(delta.y, delta.w))
                                assert abs(yaw - (15 if branch["action"] == 2 else -15)) < 0.001, yaw
                                counts["max_turn_translation_m"] = max(counts["max_turn_translation_m"], branch["displacement_m"])
                                assert branch["displacement_m"] < 5e-5, branch["displacement_m"]
                        assert pose(branched.sim) == root
                        assert np.array_equal(root_rgb, branched.sim.render())
                        assert branched.sim.previous_step_collided == root_collision
                    b = branched.step(action)
                    actions.append(action)
                    assert observation_record(branched, b) == trace[t + 1], (name, t, "actual step drift")
                    if action in branch_by_action:
                        branch = branch_by_action[action]
                        assert np.array_equal(branch["rgb"], b["rgb"]), "Branch target differs from actual same-action outcome"
                        assert branch["next_pose"] == pose(branched.sim)
                        assert branch["collided"] == branched.sim.previous_step_collided
                    counts["real_steps_compared"] += 1
                assert branched.episode_over
                counts["episodes"] += 1
                print(json.dumps({"checked_episodes": counts["episodes"], "dataset": name}), flush=True)
        finally:
            branched.close()
    assert counts["blocked_forward_branches"] > 0
    report = {"status": "passed", **counts, "policy_identity_sha256": policy.identity_sha256,
              "trainable_parameters": 0, "evaluation_mode": True, "STOP_bias": -0.25,
              "saved_processor_matches_native_evaluation": True, "online_inputs_bitwise_equal": True,
              "branch_root_pose_and_RGB_restored_exactly": True, "left_right_signs_verified": True,
              "real_trajectory_RGB_poses_metrics_and_STOP_unchanged": True,
              "branch_targets_match_actual_same_command": True,
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                                ("tools/check_step4_runtime.py", "sim/transition_branch.py", "tools/collect_wm_policy.py")}}
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

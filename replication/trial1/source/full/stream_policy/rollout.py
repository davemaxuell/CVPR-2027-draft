"""Closed-loop episodes against a policy server, for evaluation and StreamVLN DAgger (env rolevln).

Habitat and the expert run in this process; the policy runs in its own process (client.PolicyClient)
and receives live, uncompressed RGB. Without a controller the policy chooses every action. With the
StreamVLN controller the expert acts where the rule says so and the server only observes those
steps, so the policy's history is always the actual one. Frame t is saved before action t.
"""
import json
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from .controller import STOP, StreamVLNController
from .expert import WaypointExpert


def pose(env):
    state = env.sim.get_agent_state()
    r = state.rotation
    return state.position.tolist(), [float(r.w), float(r.x), float(r.y), float(r.z)]


def run_episode(env, client, success_distance, dagger=None, frames=None, jpeg_quality=95, log=print):
    """Run the next episode of `env`.

    dagger: dict(rule, actions_per_turn, beta, seed, reference_lengths: episode_id -> expert actions);
    the controller's random stream is seeded per episode from seed and episode id.
    """
    obs = env.reset()
    ep = env.current_episode
    client.reset(ep.instruction.instruction_text)
    controller = expert = None
    if dagger is not None:
        expert = WaypointExpert(env.sim, ep, dagger["rule"])
        controller = StreamVLNController(len(expert.points), dagger["reference_lengths"].get(str(ep.episode_id)),
                                         dagger["actions_per_turn"], dagger["beta"], f"{dagger['seed']}:{ep.episode_id}",
                                         dagger["rule"])
    executed, sources, labels, scores, positions, rotations, collisions = [], [], [], [], [], [], []
    distances = [float(env.get_metrics()["distance_to_goal"])]
    walked, began, policy_seconds, replans, aborted = 0.0, time.time(), 0.0, 0, None
    while not env.episode_over:
        t, rgb = len(executed), obs["rgb"]
        position, rotation = pose(env)
        positions.append(position), rotations.append(rotation)
        if frames is not None:
            Image.fromarray(rgb).save(frames / f"{t:04d}.jpg", quality=jpeg_quality)
        previous, reply = (executed[-1] if executed else None), {}

        def policy():
            action, answer = client.act(t, previous, rgb)
            reply.update(answer)
            return action, answer["remaining"]

        if controller is None:
            action, source = policy()[0], "policy"
        else:
            action, source = controller.decide(t, lambda: expert.label(controller.waypoint),
                                               lambda: expert.reached(controller.waypoint), policy, distances[-1])
            labels.append(controller.last_label)
            if not reply:
                client.observe(t, previous, rgb)
            if action is None:
                aborted = f"expert label unavailable at step {t}"
                break
        replans += bool(reply.get("replanned"))
        policy_seconds += reply.get("seconds", 0.0)
        scores.append(reply.get("scores"))
        sources.append(source)
        before = env.sim.get_agent_state().position
        obs = env.step(action)
        executed.append(int(action))
        collisions.append(bool(env.sim.previous_step_collided))
        walked += float(np.linalg.norm(env.sim.get_agent_state().position - before))
        distances.append(float(env.get_metrics()["distance_to_goal"]))
        if len(executed) % 100 == 0:
            log(json.dumps({"episode_id": str(ep.episode_id), "steps": len(executed), "seconds": time.time() - began}))
    metrics = env.get_metrics()
    final_distance = float(metrics["distance_to_goal"])
    pl = distances[0] / max(distances[0], walked) if distances[0] > 0 else 1.0
    position, rotation = pose(env)
    row = {"episode_id": str(ep.episode_id), "trajectory_id": ep.trajectory_id, "scene": ep.scene_id.split("/")[-2],
           "instruction": {"text": ep.instruction.instruction_text, "language": getattr(ep.instruction, "language", "en")},
           "num_frames": len(executed), "executed_action": executed, "action_source": sources, "policy_scores": scores,
           "position": positions, "rotation_wxyz": rotations, "collided_after_action": collisions,
           "distances_to_goal": distances, "final_position": position, "final_rotation_wxyz": rotation,
           "termination": "aborted" if aborted else "STOP" if executed and executed[-1] == STOP else "step_limit",
           "aborted": aborted, "replans": replans,
           "metrics": {"success": float(metrics["success"]), "spl": float(metrics["spl"]), "distance_to_goal": final_distance,
                       "oracle_success": float(min(distances) < success_distance), "path_length_ratio": pl,
                       "steps": len(executed), "collisions": sum(collisions)},
           "seconds": time.time() - began, "policy_seconds": policy_seconds}
    if controller is not None:
        row.update(expert_action=labels, takeover_steps=controller.takeovers, needed_help=controller.needed_help,
                   finished=controller.finished, waypoints=len(expert.points), waypoint_reached=controller.waypoint - 1,
                   waypoint_snap_distances_m=expert.snap_distances, expert_failures=expert.errors,
                   keep=aborted is None and controller.keep(final_distance, pl))
    return row


def run_workers(command, workers, out, progress, poll=5.0):
    """Run `command + ["--worker-index", i]` for each worker; call progress() until all exit, fail fast.

    SIGTERM to this process stops every worker (and with it each worker's policy server) before exiting.
    """
    children, logs = [], []
    (out / "workers").mkdir(parents=True, exist_ok=True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt("stopped by SIGTERM")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        for rank in range(workers):
            log = open(out / "workers" / f"{rank}.log", "a")
            logs.append(log)
            children.append(subprocess.Popen(command + ["--worker-index", str(rank)], stdout=log, stderr=subprocess.STDOUT))
        while True:
            codes = [child.poll() for child in children]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"worker failed: exit codes {codes}; see {out}/workers/*.log")
            progress()
            if all(code == 0 for code in codes):
                return
            time.sleep(poll)
    finally:
        signal.signal(signal.SIGTERM, previous)
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()


def python_command(script, args, skip=("worker_index",)):
    """Re-invoke `script` with the parsed `args` (store_true flags included), minus `skip`."""
    command = [sys.executable, "-u", str(script)]
    for key, value in vars(args).items():
        if key in skip or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            command.append(flag)
        elif isinstance(value, (list, tuple)):
            command += [flag, *map(str, value)]
        else:
            command += [flag, str(value)]
    return command

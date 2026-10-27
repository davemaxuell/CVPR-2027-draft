"""One-command branches in static Habitat scenes; restore the actual root exactly.

Branches use sim.step, so task steps, metrics, STOP state and policy history never
advance. Native translation/quaternion values avoid matrix decomposition on restore.
This helper deliberately rejects physics-enabled or dynamic-object scenes.
"""
import copy

import magnum as mn
import numpy as np


def pose(sim):
    state = sim.get_agent_state()
    q = state.rotation
    return {"position": state.position.tolist(), "rotation_wxyz": [float(q.w), float(q.x), float(q.y), float(q.z)]}


def movement_branches(env, actions=(1, 2, 3)):
    if any(a not in (1, 2, 3) for a in actions):
        raise ValueError("Only FORWARD/LEFT/RIGHT have dynamics branches")
    sim, agent = env.sim, env.sim.get_agent(0)
    assert not sim.sim_config.sim_cfg.enable_physics, "Branch restore is only defined for static navigation"
    assert not sim.get_rigid_object_manager().get_object_handles(), "Dynamic scene objects are unsupported"
    assert not sim.get_articulated_object_manager().get_object_handles(), "Articulated scene objects are unsupported"
    def transform(node):
        return mn.Vector3(node.translation), mn.Quaternion(node.rotation.vector, node.rotation.scalar)
    body = transform(agent.body.object)
    sensors = {name: transform(sensor.node) for name, sensor in agent._sensors.items()}
    previous_obs = copy.deepcopy(sim._prev_sim_obs)
    last_state = copy.deepcopy(sim._last_state)
    frames, timing = sim._num_total_frames, sim._previous_step_time
    metrics, elapsed, ended = copy.deepcopy(env.get_metrics()), env._elapsed_steps, env.episode_over
    stop_called = env.task.is_stop_called
    root_pose = pose(sim)

    def restore():
        agent.body.object.translation, agent.body.object.rotation = body
        for name, state in sensors.items():
            agent._sensors[name].node.translation, agent._sensors[name].node.rotation = state
        sim._prev_sim_obs = copy.deepcopy(previous_obs)
        sim._last_state = copy.deepcopy(last_state)
        sim._num_total_frames, sim._previous_step_time = frames, timing

    output = []
    try:
        for action in actions:
            restore()
            assert pose(sim) == root_pose
            obs = sim.step(action)
            after = pose(sim)
            output.append({"action": action, "root_pose": root_pose, "next_pose": after,
                           "collided": bool(sim.previous_step_collided),
                           "displacement_m": float(np.linalg.norm(np.asarray(after["position"]) - root_pose["position"])),
                           "rgb": np.array(obs["rgb"], copy=True)})
    finally:
        restore()
    assert pose(sim) == root_pose
    assert env._elapsed_steps == elapsed and env.episode_over == ended
    assert env.task.is_stop_called == stop_called and env.get_metrics() == metrics
    return output

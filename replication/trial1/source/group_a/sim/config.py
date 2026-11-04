"""Habitat configuration for the vanilla baseline, built from manifest/experiment_manifest.yaml.

    from sim.config import make_ce_config, load_expert, scene_group
    cfg = make_ce_config("rxr", "train", scenes=scene_group("A"), gpu=0)
"""
import gzip
import json
import os

import yaml
from omegaconf import OmegaConf

import habitat
from habitat.config import read_write

from .rxr_dataset import RxRDatasetConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "/data/team_a/dave-workspace/RoleVLN/data"
EPISODES = {
    "r2r": DATA + "/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz",
    "rxr": DATA + "/datasets/RxR_VLNCE_v0/{split}/{split}_{role}.json.gz",
}
EXPERT = {
    "r2r": DATA + "/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz",
    "rxr": DATA + "/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz",
}


def load_manifest():
    with open(f"{ROOT}/manifest/experiment_manifest.yaml") as f:
        return yaml.safe_load(f)


def scene_group(group):
    with open(f"{ROOT}/manifest/scene_partition.json") as f:
        return json.load(f)["groups"][group]["scenes"]


def rxr_roles():
    return load_manifest()["datasets"]["rxr_ce"]["roles"]


def make_ce_config(dataset, split, scenes=None, gpu=0, depth=False):
    """VLN-CE task config. `gpu` is Habitat's EGL device; do not remap it with CUDA_VISIBLE_DEVICES.

    The vanilla models use RGB only, so the depth sensor is dropped unless `depth=True`.
    """
    manifest = load_manifest()
    sim = manifest["simulator"]
    cfg = habitat.get_config("benchmark/nav/vln_r2r.yaml")
    with read_write(cfg):
        h = cfg.habitat
        if dataset == "rxr":
            rxr = manifest["datasets"]["rxr_ce"]
            base = OmegaConf.to_container(h.dataset)
            h.dataset = OmegaConf.structured(RxRDatasetConfig(**base, roles=rxr["roles"], languages=rxr["languages"]))
            h.dataset.type = "RxRVLN-v1"
        h.dataset.data_path = EPISODES[dataset]
        h.dataset.scenes_dir = DATA + "/scene_datasets"
        h.dataset.split = split
        h.dataset.content_scenes = list(scenes) if scenes else ["*"]
        h.environment.max_episode_steps = sim["max_episode_steps"][dataset]
        h.environment.iterator_options.shuffle = False
        h.environment.iterator_options.group_by_scene = False
        h.simulator.habitat_sim_v0.gpu_device_id = gpu
        h.simulator.forward_step_size = sim["forward_step_m"]
        h.simulator.turn_angle = sim["turn_angle_deg"]
        sensors = h.simulator.agents.main_agent.sim_sensors
        sensors.rgb_sensor.width, sensors.rgb_sensor.height = sim["rgb"]["width"], sim["rgb"]["height"]
        sensors.rgb_sensor.hfov = sim["rgb"]["hfov"]
        if not depth:
            del sensors["depth_sensor"]
        h.task.measurements.success.success_distance = sim["success_distance_m"]
    return cfg


TURN_IDS = (2, 3)  # habitat task order: stop, move_forward, turn_left, turn_right


def load_expert(dataset, split):
    """Expert actions/locations keyed by str(episode_id), in the configured action space.

    RxR-CE experts were generated with 30-degree turns. With 15-degree turns configured, each
    expert turn becomes two turns; turns do not move the agent, so `locations` are unchanged.
    """
    sim = load_manifest()["simulator"]
    roles = rxr_roles() if dataset == "rxr" else [None]
    expert = {}
    for role in roles:
        with gzip.open(EXPERT[dataset].format(split=split, role=role), "rt") as f:
            expert.update(json.load(f))
    if dataset == "rxr":
        repeat, rest = divmod(sim["rxr_expert_turn_angle_deg"], sim["turn_angle_deg"])
        assert rest == 0, "RxR expert turn angle must be a multiple of the configured turn angle"
        for ref in expert.values():
            ref["actions"] = [b for a in ref["actions"] for b in ([a] * repeat if a in TURN_IDS else [a])]
    return expert

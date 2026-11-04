"""Inspect the real Habitat loader and metric classes without creating a simulator."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
native = HERE.parent / 'continuation_from_990_20261103/source_v1'
model = json.loads((HERE / 'models.json').read_text())['F']
os.environ.update(CUDA_VISIBLE_DEVICES='', ROLEVLN_QUEUE_OUTPUT=model['output_root'],
    ROLEVLN_QUEUE_CODE_ROOT=str(HERE / 'evaluation/source_v1'), ROLEVLN_NATIVE_CODE_ROOT=str(native),
    ROLEVLN_RXR_AUDIT=str(HERE / 'split_audit.json'), ROLEVLN_RXR_PROTOCOL=str(HERE / 'protocol.json'))
sys.path.insert(0, str(HERE / 'evaluation/source_v1'))
import evaluate
import numpy as np
from common import atomic_json, utc, sha256
from official_metrics import CLASSES
from omegaconf import OmegaConf

args = SimpleNamespace(dataset='rxr', split='val_unseen', episodes=0, seed=0)
cfg, episodes = evaluate.episodes_for(args)
assert len(episodes) == 3669
assert set(ep.instruction.language for ep in episodes) == {'en-US', 'en-IN'}
assert cfg.habitat.dataset.content_scenes == ['*']
assert cfg.habitat.simulator.turn_angle == 15
assert cfg.habitat.simulator.forward_step_size == .25
assert cfg.habitat.environment.max_episode_steps == 700
assert list(cfg.habitat.task.actions) == ['stop','move_forward','turn_left','turn_right']
assert cfg.habitat.task.measurements.success.success_distance == 3.0
sensor = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
assert (sensor.width,sensor.height,sensor.hfov) == (640,480,79)
assert list(cfg.habitat.dataset.roles) == ['guide']
assert list(cfg.habitat.dataset.languages) == ['en-US','en-IN']

# Exercise the verbatim upstream NDTW class, its RxR role-map load, string ID
# lookup and metric calculation. No Habitat simulator or policy is instantiated.
fake = SimpleNamespace(position=None)
fake.get_agent_state = lambda:SimpleNamespace(position=fake.position)
measure = CLASSES['NDTW'](sim=fake, config=cfg.habitat.task.measurements.ndtw)
reference = measure.gt_json[episodes[0].episode_id]['locations']
fake.position = np.asarray(reference[0])
measure.reset_metric(episode=episodes[0])
for position in reference[1:]:
    fake.position = np.asarray(position)
    measure.update_metric(episode=episodes[0])
assert math.isclose(measure.get_metric(),1.,abs_tol=1e-8)
import torch
assert not torch.cuda.is_initialized()
atomic_json(HERE / 'evidence/habitat_cpu_probe.json', dict(status='complete',time=utc(),
    episodes=len(episodes), scenes=len({ep.scene_id for ep in episodes}),
    episode_ids_sha256=hashlib.sha256(json.dumps([ep.episode_id for ep in episodes]).encode()).hexdigest(),
    configuration=OmegaConf.to_container(cfg,resolve=True), real_official_ndtw_identity_path=measure.get_metric(),
    cuda_initialized=False, evaluation_sha256=sha256(HERE / 'evaluation/source_v1/evaluate.py'),
    note='CPU loader and metric checks only. Actual GPU RxR evaluation remains queued after active F R2R.'))
print(json.dumps(dict(status='complete',episodes=len(episodes),scenes=len({ep.scene_id for ep in episodes}),
                     official_ndtw_identity=measure.get_metric(),cuda_initialized=False)))

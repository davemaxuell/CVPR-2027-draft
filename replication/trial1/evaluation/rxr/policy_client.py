"""Simulator-side client for the observational queue server. No GPU imports."""
import os
import subprocess
import sys
from pathlib import Path

from common import CODE, OUTPUT, QUEUE_CODE, POLICY_PYTHON

sys.path.insert(0, str(CODE))
from stream_policy.client import PolicyClient


class RecordedClient(PolicyClient):
    def __init__(self, run_dir, weights, log, feature_dir='',force_slot=False):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false')
        env.pop('PYTHONPATH', None)
        command = [POLICY_PYTHON, '-B', '-u', str(QUEUE_CODE / 'policy_server.py'),
                   '--run-dir', str(run_dir), '--weights', str(weights), '--feature-dir', str(feature_dir)]
        if force_slot:
            command.append('--force-slot')
        self.process = subprocess.Popen(command,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, cwd=str(OUTPUT), env=env)
        self.identity = self.call({'op': 'hello'})

    def reset(self, instruction, episode_key=None):
        self.call({'op': 'reset', 'instruction': instruction, 'episode_key': episode_key})

    def set_records(self,records):
        self.call({'op':'set_records','records':records})

    def decoding(self,temperature,seed):
        self.call({'op':'decoding','temperature':temperature,'seed':seed})

    def invalidate_plan(self):
        self.call({'op':'invalidate_plan'})

    def restore_sampling_rng(self,state):
        self.call({'op':'restore_sampling_rng','state':state})

    def openings(self,depth):
        import numpy as np
        value = np.asarray(depth,dtype=np.float32)
        return self.call({'op':'openings','shape':list(value.shape)},value.tobytes())

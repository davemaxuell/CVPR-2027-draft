import sys
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler

from utils.distributed import is_default_gpu
from utils.data import PickSpecificWords


class BaseAgent(object):
    """Base class for a REVERIE navigation agent."""

    def __init__(self, env):
        self.env = env  # Environment the agent interacts with
        self.results = {}  # Stores final navigation trajectories
        self.device = torch.device(f"cuda:0")
        self.finished_counter = 0  # For debugging/batching
        self.token_stats = defaultdict(list)

        # Load irrelevant word IDs from a vocabulary list for filtering purposes
        self.irrelevant_words = []
        self.irrelevant_words_id = set()
        with open(self.env.args.vocabulary_path, "r") as file:
            for line in file:
                self.irrelevant_words_id.add(int(line.strip()))

    def get_results(self, detailed_output=False):
        """Return navigation results, optionally with details."""
        output = []
        for k, v in self.results.items():
            item = {"instr_id": k, "trajectory": v["path"]}
            if detailed_output:
                item["details"] = v["details"]
            output.append(item)
        return output

    def rollout(self, **args):
        """Placeholder to be implemented in child classes.

        Should return a list of dicts with keys:
        - 'instr_id': string
        - 'path': list of (viewpointId, heading, elevation)
        """
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        """Factory method to get a specific agent class by name."""
        return globals()[name + "Agent"]

    def test(self, iters=None, z_dicts={}, z_front_dict={}, **kwargs):
        """Run a test pass through the environment."""
        self.env.reset_epoch(shuffle=(iters is not None))
        self.losses = []
        self.results = {}
        self.loss = 0
        looped = False

        while True:
            for traj in self.rollout(test=True, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs):
                if traj["instr_id"] in self.results:
                    looped = True  # Avoid duplicate processing
                else:
                    self.loss = 0
                    self.results[traj["instr_id"]] = traj
            print("batch: ", self.finished_counter, " is finished.")

            if looped:
                break


class Seq2SeqAgent(BaseAgent):
    """Seq2Seq-based navigation agent with visual-language transformer (VLN-BERT) and critic."""

    # Mapping of high-level action names to 3D navigation deltas
    env_actions = {
        "left": (0, -1, 0),
        "right": (0, 1, 0),
        "up": (0, 0, 1),
        "down": (0, 0, -1),
        "forward": (1, 0, 0),
        "<end>": (0, 0, 0),
        "<start>": (0, 0, 0),
        "<ignore>": (0, 0, 0),
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, tok, rank=0):
        super().__init__(env)
        self.args = args
        self.feature_size = args.feature_size
        self.tok = tok
        self.default_gpu = is_default_gpu(args)
        self.rank = rank

        # Initialize word selector for backdoor updates
        if args.z_instr_update:
            self.word_picker = PickSpecificWords(cat_file=args.cat_file)
            self.instr_specific_dict = defaultdict(lambda: [])

        # Build model components (implemented in child classes)
        self._build_model()

        # Use DistributedDataParallel if in distributed training mode
        if self.args.world_size > 1:
            self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
            self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

        self.models = (self.vln_bert, self.critic)
        self.device = torch.device(f"cuda:{self.rank}")

        # Logging dictionary
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def test(self, use_dropout=False, feedback="argmax", iters=None, z_dicts={}, z_front_dict=None):
        """Run agent evaluation in test mode (no training)."""
        self.feedback = feedback
        self.vln_bert.eval()
        self.critic.eval()
        super().test(iters=iters, z_dicts=z_dicts, z_front_dict=z_front_dict)

    def load(self, path):
        """Load model parameters from checkpoint."""
        states = torch.load(path, map_location=lambda storage, loc: storage)

        def recover_state(name, model):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]["state_dict"].keys())
            state_dict = states[name]["state_dict"]

            # Handle mismatched keys from different DDP wrappers
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith("module.") and list(load_keys)[0].startswith("module."):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                if list(model_keys)[0].startswith("module.") and not list(load_keys)[0].startswith("module."):
                    state_dict = {"module." + k: v for k, v in state_dict.items()}

                # Filter out keys that don't match
                same_state_dict = {}
                extra_keys = []
                for k, v in state_dict.items():
                    if k in model_keys:
                        same_state_dict[k] = v
                    else:
                        extra_keys.append(k)

                state_dict = same_state_dict
                print("Extra keys in state_dict: %s" % (", ".join(extra_keys)))

            # Load model weights
            state.update(state_dict)
            model.load_state_dict(state)

        # Load all model parts
        all_tuple = [("vln_bert", self.vln_bert)]
        for param in all_tuple:
            recover_state(*param)

        return states["vln_bert"]["epoch"] - 1

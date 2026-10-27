import sys
from collections import defaultdict

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import is_default_gpu
from utils.data import PickSpecificWords


class BaseAgent(object):
    """Base class for a REVERIE navigation agent."""

    def __init__(self, env):
        self.env = env  # Environment the agent interacts with
        self.device = torch.device("cuda:0")
        self.results = {}  # Stores final predicted trajectories
        self.finished_counter = 0  # Tracks finished batches
        self.token_stats = defaultdict(list)  # Optional stats tracking

        # Load irrelevant token IDs for filtering
        self.irrelevant_words = []
        self.irrelevant_words_id = set()
        with open(self.env.args.vocabulary_path, "r") as file:
            for line in file:
                self.irrelevant_words_id.add(int(line.strip()))

    def get_results(self, detailed_output=False):
        """Return navigation results, optionally with additional details."""
        output = []
        for instr_id, result in self.results.items():
            obj_id = int(result['pred_objid']) if result['pred_objid'] is not None else None
            entry = {
                'instr_id': instr_id,
                'trajectory': result['path'],
                'predObjId': obj_id
            }
            if detailed_output:
                entry['details'] = result['details']
            output.append(entry)
        return output

    def rollout(self, **kwargs):
        """Must be implemented in child classes. Performs one rollout and returns a list of results."""
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        """Retrieve a specific agent class by name."""
        return globals()[name + "Agent"]

    def test(self, iters=None, z_dicts=None, z_front_dict=None, **kwargs):
        """Run testing/evaluation on the environment and collect predictions."""
        self.env.reset_epoch(shuffle=(iters is not None))
        self.losses = []
        self.results = {}
        self.loss = 0
        looped = False

        if iters is not None:
            # Fixed number of iterations (usually shuffled beforehand)
            for _ in range(iters):
                for traj in self.rollout(z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
        else:
            # Run until the environment loops or all data is processed
            while True:
                for traj in self.rollout(z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                if looped:
                    break
                print(f"Batch {self.finished_counter} is finished.")


class Seq2SeqAgent(BaseAgent):
    """Seq2Seq-based REVERIE agent using a VLN-BERT and critic model."""

    # Mapping of high-level actions to 3D movement deltas
    env_actions = {
        'left': (0, -1, 0),
        'right': (0, 1, 0),
        'up': (0, 0, 1),
        'down': (0, 0, -1),
        'forward': (1, 0, 0),
        '<end>': (0, 0, 0),
        '<start>': (0, 0, 0),
        '<ignore>': (0, 0, 0)
    }
    env_actions = {k: [[v_i] for v_i in v] for k, v in env_actions.items()}

    def __init__(self, args, env, tok, rank=0):
        super().__init__(env)
        self.args = args
        self.tok = tok
        self.rank = rank
        self.default_gpu = is_default_gpu(args)

        # Setup category-specific token selector if enabled
        if args.z_instr_update:
            self.word_picker = PickSpecificWords(cat_file=args.cat_file)
            self.instr_specific_dict = defaultdict(list)

        # Build navigation and critic models
        self._build_model()

        # Wrap with DDP if running in distributed mode
        if self.args.world_size > 1:
            self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
            self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

        self.models = (self.vln_bert, self.critic)
        self.device = torch.device(f"cuda:{self.rank}")
        self.logs = defaultdict(list)

        sys.stdout.flush()

    def _build_model(self):
        """Must be implemented in subclasses to define self.vln_bert and self.critic."""
        raise NotImplementedError("Child class should implement `_build_model()`.")

    def test(self, use_dropout=False, feedback='argmax', iters=None, z_dicts=None, z_front_dict=None):
        """Evaluate the agent on a validation/test split."""
        self.feedback = feedback

        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()

        super().test(iters=iters, z_dicts=z_dicts, z_front_dict=z_front_dict)

    def load(self, path):
        """Load model parameters from a saved checkpoint (weights only)."""
        states = torch.load(path)

        def recover_state(name, model):
            state = model.state_dict()
            loaded_state = states[name]['state_dict']

            model_keys = set(state.keys())
            loaded_keys = set(loaded_state.keys())

            # Handle key mismatches due to DDP wrapping (with/without 'module.')
            if model_keys != loaded_keys:
                print("NOTICE: Mismatched keys between model and checkpoint.")

                if not list(model_keys)[0].startswith('module.') and list(loaded_keys)[0].startswith('module.'):
                    loaded_state = {k.replace('module.', ''): v for k, v in loaded_state.items()}
                elif list(model_keys)[0].startswith('module.') and not list(loaded_keys)[0].startswith('module.'):
                    loaded_state = {'module.' + k: v for k, v in loaded_state.items()}

                # Keep only matching keys
                filtered_state = {k: v for k, v in loaded_state.items() if k in model_keys}
                extra_keys = [k for k in loaded_state if k not in model_keys]
                if extra_keys:
                    print(f"Extra keys in state_dict: {', '.join(extra_keys)}")

                state.update(filtered_state)
            state.update(loaded_state)
            model.load_state_dict(state)

        # Load model weights
        recover_state("vln_bert", self.vln_bert)
        return states['vln_bert']['epoch'] - 1

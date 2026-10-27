"""Keep native policy/memory updates; extend only the frozen candidate predictor."""
from collections import deque

import torch
from torchvision.transforms.functional import to_tensor

from navigation_eval.controller import NavigationController as NativeController
from navigation_four.common import ADDITIONS, sample_seeds
from nwm_training.frozen import FrozenNwmWorldModel


class NavigationController(NativeController):
    def __init__(self, mode='policy', checkpoint=None, world_model=None, margin=None):
        self.nwm = None
        self.nwm_requested = mode == 'predicted' and world_model in ADDITIONS
        super().__init__('actual' if self.nwm_requested else mode, checkpoint, world_model, margin)
        self.world_model = world_model
        if self.nwm_requested:
            self.nwm = FrozenNwmWorldModel(world_model, encoder=self.encoder)
            self.mode = 'predicted'

    @torch.inference_mode()
    def reset(self, instruction, budget, *, initialize_instruction=True):
        super().reset(instruction, budget, initialize_instruction=initialize_instruction)
        self.vae_history = deque(maxlen=4)
        self.root_prefix = None
        self.vae_encoding_error = None
        self.last_sample_seeds = None

    def set_episode_identity(self, group, dataset, episode_id, prefix_source='policy'):
        assert group in ('B', 'C', 'E') and dataset in ('r2r', 'rxr') and prefix_source in ('expert', 'policy')
        assert not self.committed and not self.pending
        self.root_prefix = f'{group}/{dataset}/{prefix_source}/{episode_id}/roots'

    @torch.inference_mode()
    def observe(self, image, actions):
        native = super().observe(image, actions)
        if self.nwm_requested:
            self.vae_encoding_error = None
            try:
                self.vae_history.append(self.nwm.encode_actual(to_tensor(image)[None])[0])
            except (ValueError, TypeError, RuntimeError) as error:
                self.vae_history.append(None)
                self.vae_encoding_error = f'{type(error).__name__}: {error}'
        return native

    @torch.inference_mode()
    def futures(self, preview=None):
        if not self.nwm_requested or self.mode != 'predicted': return super().futures(preview)
        assert len(self.committed) >= 3 and self.root_prefix is not None
        if len(self.vae_history) != 4 or any(value is None for value in self.vae_history):
            raise ValueError('Four valid real VAE observations are required')
        root_id = f'{self.root_prefix}/{len(self.committed):04d}'
        self.last_sample_seeds = sample_seeds(root_id)
        moments = torch.stack(list(self.vae_history))[None]
        self.attempted_predictions += 3
        movement = self.nwm.predict_movements(moments,
            context_seed=self.last_sample_seeds['context_seed'], sampling_seed=self.last_sample_seeds['diffusion_seed'])
        return torch.cat((movement[None], self.context.current_patches[:, None]), dim=1)

    @torch.inference_mode()
    def decide(self, preview=None):
        self.last_sample_seeds = None
        live, event = super().decide(preview)
        if self.nwm_requested:
            event['NWM_sampling'] = {'steps':50, 'candidate_batch_order':['FORWARD', 'LEFT', 'RIGHT'],
                'seeds':self.last_sample_seeds, 'vae_encoding_error':self.vae_encoding_error}
        return live, event

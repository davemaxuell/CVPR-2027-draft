import os

import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from tqdm import tqdm
import wandb

from mad_td.cfgs.mad_td_config import MadTdHyperparams, MadTdModels

from mad_td.cfgs.train_config import TrainHyperparams
from mad_td.training_util.action_selection import get_mpc_action, get_policy_action
from mad_td.training_util.rl_targets import soft_target_update
from mad_td.rl_types import (
    AbstractActor,
    AbstractCritic,
    AbstractEncoder,
    AbstractLatentModel,
    get_batch_seed_shape,
)
from mad_td.data.env import Env, make_env
from mad_td.data.replay_buffer import ReplayBuffer
from mad_td.utils.checkpointing import CheckpointHandler
from mad_td.utils import eval_utils
from mad_td.utils.jax import tree_list_mean
from mad_td.utils.adversarial import compute_all_adversarial_metrics
from mad_td.utils.actor_noise import compute_noise
from mad_td.trainers.mad_td_model import (
    MadTdFactory,
)
from mad_td.update_functions.mad_td_update import vmaped_utd_update
from mad_td.utils.logging import (
    Logger,
    multi_seed_return_dict,
)


class MultiSeedTrainer:
    def __init__(
        self,
        critic: AbstractCritic,
        critic_target: AbstractCritic,
        actor: AbstractActor,
        encoder: AbstractEncoder,
        encoder_target: AbstractEncoder,
        latent_model: AbstractLatentModel,
        replay_buffer: ReplayBuffer,
        env: Env,
        algo_hyperparams: MadTdHyperparams,
        train_hyperparams: TrainHyperparams,
        logger: Logger,
    ):
        self.mad_td = MadTdFactory(
            critic=critic,
            critic_target=critic_target,
            actor=actor,
            encoder=encoder,
            encoder_target=encoder_target,
            latent_model=latent_model,
            replay_buffer=replay_buffer,
            encoder_learning_rate=algo_hyperparams.encoder_learning_rate,
            model_learning_rate=algo_hyperparams.model_learning_rate,
            critic_learning_rate=algo_hyperparams.critic_learning_rate,
            actor_learning_rate=algo_hyperparams.actor_learning_rate,
            gradient_clip=algo_hyperparams.gradient_clip,
            seed=train_hyperparams.seed,
            precision=train_hyperparams.precision,
            param_precision=train_hyperparams.param_precision,
        )
        self.models = self.mad_td.init()

        self.encoder = encoder
        self.latent_model = latent_model
        self.critic = critic
        self.actor = actor

        # setup replay buffer
        self.replay_buffer = replay_buffer

        # setup shapes for vmap
        self.batch_shape = get_batch_seed_shape(
            self.replay_buffer.get_dummy_batch(1, batch_size=10)
        )

        # setup hyperparams
        self.algo_hyperparams = algo_hyperparams
        self.train_hyperparams = train_hyperparams

        # setup utility
        self.key = jax.random.PRNGKey(train_hyperparams.seed)
        self.key, env_step_key = jax.jit(jax.random.split)(self.key)

        # setup env
        self.env = env
        self.env_reset = self.env.get_n_reset()
        self.env_step = self.env.get_n_step(env_step_key)

        self.logger = logger

        # prejit update step
        self.dummy_batch = self.replay_buffer.get_dummy_batch(
            num_samples=train_hyperparams.update_steps,
            batch_size=train_hyperparams.batch_size,
        )
        self._update_fn = vmaped_utd_update

        # handle reloading logic here
        self.steps_done = 0

        self.checkpointer = CheckpointHandler(train_hyperparams.save_path)

    def check_pretrain(self, path, alt_path):
        if alt_path:
            path = os.path.join(alt_path, path)

        if os.path.exists(os.path.join(path, "steps_done.txt")):
            self.load(path)
            with open(os.path.join(path, "steps_done.txt"), "r") as f:
                self.steps_done = int(f.read())
            print(f"Resuming from previous checkpoint at {self.steps_done}")
        else:
            print(
                f"No checkpoint found at {os.path.join(path, 'steps_done.txt')}, starting from scratch"
            )

    def save(self, path):
        self.replay_buffer.save(path)
        self.checkpointer.checkpoint_params(self.models, self.steps_done)
        with open(os.path.join(path, "steps_done.txt"), "w") as f:
            f.write(str(self.steps_done))

    def load(self, path):
        self.replay_buffer.load(path)
        reload_checkpointer = CheckpointHandler(path)
        self.models = reload_checkpointer.restore_params(self.models, path)

    def train(self):
        num_seeds = self.train_hyperparams.num_seeds
        key, reset_key = jax.random.split(self.key)
        done = False
        rewards = 0
        reset_rng = jax.random.split(reset_key, num_seeds)
        state = self.env_reset(reset_rng)

        return_dicts = None
        action_info_dicts = None
        states_used_in_model = None

        total_steps = self.train_hyperparams.total_steps

        if self.algo_hyperparams.use_mpc:
            act_fn = get_mpc_action
        else:
            act_fn = get_policy_action
        vmaped_action = jax.jit(
            jax.vmap(
                lambda s, e, a, c, m, noise_sigma, random, key: act_fn(
                    s,
                    e,
                    a,
                    c,
                    m,
                    noise_sigma,
                    random,
                    self.algo_hyperparams,
                    key,
                ),
                in_axes=(
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    0,
                ),
            ),
            static_argnames=("random"),
        )

        with tqdm(total=total_steps) as pbar:
            pbar.update(self.steps_done)
            while self.steps_done <= total_steps:
                for _ in range(self.train_hyperparams.env_steps):
                    # key logic
                    key, step_key = jax.jit(jax.random.split)(key)
                    step_keys = jax.jit(jax.random.split, static_argnums=(1,))(
                        step_key, num_seeds
                    )
                    if np.any(done):
                        state = self.env_reset(step_keys)
                        done = False
                        self.logger.log(
                            multi_seed_return_dict(
                                {"reward": rewards},
                                self.train_hyperparams.num_seeds,
                                1.0,
                            ),
                            self.steps_done,
                        )
                        self.logger.log(
                            {"mean_reward": float(np.mean(rewards))},
                            self.steps_done,
                        )
                        rewards = 0
                    key, action_key = jax.jit(jax.random.split)(key)
                    action_keys = jax.jit(jax.random.split, static_argnums=(1,))(
                        action_key, num_seeds
                    )
                    action, action_info = vmaped_action(
                        self.replay_buffer.normalize(state.obs),
                        self.models.encoder,
                        self.models.actor,
                        self.models.critic,
                        self.models.latent_model,
                        self.algo_hyperparams.exploration_noise_sigma,
                        self.steps_done < self.train_hyperparams.init_steps,
                        action_keys,
                    )
                    if action_info_dicts is None:
                        action_info_dicts = action_info
                    else:
                        action_info_dicts = jax.jit(jax.tree_map, static_argnums=[0])(
                            jnp.add, action_info_dicts, action_info
                        )
                    action_info_dicts
                    next_state, reward, done = self.env_step(
                        step_keys, state.state, action
                    )
                    rewards += reward
                    self.replay_buffer.insert(
                        state.obs, action, reward, done, next_state.obs, None
                    )
                    state = next_state
                self.steps_done += self.train_hyperparams.env_steps
                if self.steps_done > self.train_hyperparams.init_steps:
                    # split all required keys
                    key, model_key = jax.jit(jax.random.split)(key)
                    # sample batches
                    batch = self.replay_buffer.sample(
                        self.train_hyperparams.batch_size, model_key, self.train_hyperparams.update_steps  # type: ignore
                    )
                    # update models
                    self.models, return_dict = jax.jit(
                        self._update_fn, static_argnums=(2, 3, 4, 5)
                    )(
                        self.models,
                        batch,
                        self.algo_hyperparams,
                        self.batch_shape,
                        self.train_hyperparams.num_seeds,
                        self.train_hyperparams.update_steps,
                        key,
                        self.steps_done,
                    )
                    if return_dicts is None:
                        return_dicts = return_dict
                    else:
                        return_dicts = jax.jit(jax.tree_map, static_argnums=[0])(
                            jnp.add, return_dicts, return_dict
                        )

                    # logging
                    if self.steps_done % self.train_hyperparams.log_freq == 0:
                        self.logger.log(
                            multi_seed_return_dict(
                                action_info,
                                self.train_hyperparams.num_seeds,
                                self.train_hyperparams.log_freq,
                            ),
                            self.steps_done,
                        )
                        action_info_dicts = multi_seed_return_dict(
                            action_info_dicts,
                            self.train_hyperparams.num_seeds,
                            self.train_hyperparams.log_freq,
                        )
                        self.logger.log(action_info_dicts, self.steps_done)
                        del action_info_dicts
                        action_info_dicts = None
                        return_dicts = multi_seed_return_dict(
                            return_dicts,
                            self.train_hyperparams.num_seeds,
                            self.train_hyperparams.log_freq,
                        )
                        self.logger.log(return_dicts, self.steps_done)
                        del return_dicts
                        return_dicts = None

                    if self.steps_done % 500 == 0:
                        self.logger.flush()
                    if self.steps_done % self.train_hyperparams.save_freq == 0:
                        self.save(self.train_hyperparams.save_path)
                if self.steps_done % self.train_hyperparams.tqdm_interval == 0:
                    pbar.update(
                        self.train_hyperparams.tqdm_interval
                        * self.train_hyperparams.env_steps
                    )

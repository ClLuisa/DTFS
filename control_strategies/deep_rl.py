import json
import os
import random
from collections import deque, namedtuple

import numpy as np

from connect_protocols.simulation_env import SimulationEnv

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure
except ImportError:
    DQN = None
    BaseCallback = object
    configure = None


# --- Seed -------------------------------------------------------------------
# Set the seed for a training run in one of two ways (the first one found wins):
#   1. DeepQAgentTorch(..., seed=3)
#   2. environment variable DTFS_SEED, e.g. in PowerShell:  $env:DTFS_SEED = "3"
#      (or at the top of connect_simulation.py:  os.environ["DTFS_SEED"] = "3")
# Without a seed, training stays random as before.
# Use a separate experiment name per seed (e.g. "..._seed3"), otherwise the
# checkpoint of the previous run is loaded and training continues from it.
SEED_ENV_VAR = "DTFS_SEED"

# --- Training settings that can be changed per run ------------------------
# DTFS_BUFFER_SIZE: replay buffer size. Default 1_000_000 keeps ALL experience of
#   a multi-year run. Runs shorter than 10_000 steps (e.g. the 2-month agents with
#   8_500 steps) behave exactly as before, because the old buffer (10_000) never filled.
# DTFS_GAMMA: discount factor (default 0.99 = previous setting).
# DTFS_CHECKPOINT_EVERY: save an extra model every N steps (0 = off), e.g. 8760*6//2
#   to compare a half-year model with the final one (detects forgetting).
BUFFER_SIZE = int(os.environ.get("DTFS_BUFFER_SIZE", "1000000"))
GAMMA = float(os.environ.get("DTFS_GAMMA", "0.99"))
CHECKPOINT_EVERY = int(os.environ.get("DTFS_CHECKPOINT_EVERY", "0"))


def resolve_seed(seed=None):
    if seed is not None:
        return int(seed)
    value = os.environ.get(SEED_ENV_VAR, "").strip()
    return int(value) if value else None


def set_global_seed(seed: int):
    """Seed Python, NumPy and PyTorch (CPU and GPU)."""
    random.seed(seed)  # also used by SimulationEnv.resolve_action (action hold)
    np.random.seed(seed)  # used by PersistentDQN.predict for epsilon-greedy
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


CONTROL_ACTION_VALUES = [0, 2, 1, -1]
N_OBSERVATIONS = 3
N_ACTIONS = len(CONTROL_ACTION_VALUES)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class PersistentDQN(DQN):
    """DQN that applies the environment's action-persistence curriculum."""

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "persistence_env"]

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        environment = getattr(self, "persistence_env", None)
        if not deterministic and environment is not None:
            action, state = super().predict(observation, state, episode_start, True)
        else:
            action, state = super().predict(observation, state, episode_start, deterministic)
        if not deterministic and environment is not None:
            epsilon = float(self.exploration_rate)
            if np.random.random() < epsilon:
                proposed = environment.action_space.sample()
                source = "random"
            else:
                proposed = int(np.asarray(action).flat[0])
                source = "learned"
            resolved = environment.resolve_action(
                proposed, self.num_timesteps, source=source, epsilon=epsilon
            )
            action = np.asarray([resolved], dtype=np.int64)
            self.last_decision_info = dict(environment.last_decision_info)
        return action, state


class TrainingMetricsCallback(BaseCallback):
    """Persist per-step rollout metrics for later comparison."""

    def __init__(self, metrics_path: str):
        super().__init__(verbose=0)
        self.metrics_path = metrics_path
        self.file = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        self.file = open(self.metrics_path, "w", encoding="utf-8")

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", [None])).reshape(-1)
        actions = np.asarray(self.locals.get("actions", [None])).reshape(-1)
        loss = self.model.logger.name_to_value.get("train/loss")
        record = {
            "timesteps": int(self.num_timesteps),
            "reward": None if rewards[0] is None else float(rewards[0]),
            "action_index": None if actions[0] is None else int(actions[0]),
            "exploration_rate": float(self.model.exploration_rate),
            "loss": None if loss is None else float(loss),
            "decision": dict(getattr(self.model, "last_decision_info", {})),
        }
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()
        return True

    def _on_training_end(self):
        if self.file is not None:
            self.file.close()


def state_to_observation(state: dict) -> np.ndarray:
    return SimulationEnv.observation(state)


class ReplayMemory:
    def __init__(self, capacity: int = 10_000):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def iter_batches(self, batch_size: int):
        for start in range(0, len(self.memory), batch_size):
            batch = list(self.memory)[start:start + batch_size]
            if batch:
                yield batch

    def __len__(self):
        return len(self.memory)


class DeepQAgentTorch:
    """Stable-Baselines3 DQN adapter retaining the existing agent API."""

    def __init__(self, checkpoint_path: str, env=None, require_checkpoint: bool = False,
                 seed: int | None = None):
        if DQN is None:
            raise ImportError(
                "Deep RL requires gymnasium and stable-baselines3. "
                "Install them with: pip install gymnasium stable-baselines3[extra]"
            )
        self.checkpoint_path = checkpoint_path
        self.require_checkpoint = require_checkpoint
        self.env = env or SimulationEnv()
        self.seed = resolve_seed(seed)
        if self.seed is not None:
            set_global_seed(self.seed)
            self.env.action_space.seed(self.seed)
            self.env.observation_space.seed(self.seed)
        self.model = self._load_or_create()
        self.model.persistence_env = self.env
        self.model.set_logger(configure(folder=None, format_strings=[]))
        self.last_state = None
        self.last_action = None
        self._previous_temperature = None
        self._previous_action = 0

    def _model_path(self):
        return self.checkpoint_path.removesuffix(".pt")

    def _load_or_create(self):
        model_path = self._model_path()
        checkpoint_file = model_path + ".zip"
        checkpoint_exists = os.path.isfile(checkpoint_file) and os.path.getsize(checkpoint_file) > 0
        if self.require_checkpoint and not checkpoint_exists:
            raise FileNotFoundError(
                f"Required DQN checkpoint is missing or empty: {checkpoint_file}. "
                "Complete training successfully before running evaluation."
            )
        if checkpoint_exists:
            if not self.require_checkpoint:
                print(f"[DeepQAgentTorch] Existing checkpoint found, training CONTINUES from it: {checkpoint_file}. "
                      "Use a new experiment name (or delete the folder) for a fresh run.")
            try:
                model = PersistentDQN.load(model_path, env=self.env, device="auto")
                if self.seed is not None:
                    model.set_random_seed(self.seed)
                return model
            except (RuntimeError, ValueError, KeyError) as error:
                if self.require_checkpoint:
                    raise RuntimeError(
                        f"Could not load DQN checkpoint: {checkpoint_file}"
                    ) from error
        return PersistentDQN(
            "MlpPolicy", self.env,
            learning_rate=3e-4, buffer_size=BUFFER_SIZE, learning_starts=64,
            batch_size=64, train_freq=1, gradient_steps=1,
            gamma=GAMMA, tau=1.0, target_update_interval=250,
            exploration_initial_eps=0.85, exploration_final_eps=0.05,
            exploration_fraction=0.25, policy_kwargs={"net_arch": [64, 64]},
            verbose=0, device="auto", seed=self.seed,
        )

    def learn(self, total_timesteps: int):
        try:
            run_dir = os.path.dirname(self.checkpoint_path)
            metrics_path = os.path.join(run_dir, "training_metrics.jsonl")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as f:
                json.dump({"seed": self.seed, "total_timesteps": total_timesteps,
                           "buffer_size": int(self.model.buffer_size), "gamma": float(self.model.gamma)}, f, indent=2)
            callbacks = [TrainingMetricsCallback(metrics_path)]
            if CHECKPOINT_EVERY > 0:
                callbacks.append(CheckpointCallback(save_freq=CHECKPOINT_EVERY,
                                                    save_path=os.path.join(run_dir, "checkpoints"),
                                                    name_prefix="online_model"))
            self.model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=CallbackList(callbacks),
            )
        finally:
            try:
                self._save_model()
            finally:
                self.env.set_learning_done()

    def _save_model(self):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        model_path = self._model_path()
        temporary_path = model_path + ".tmp.zip"
        self.model.save(temporary_path)
        os.replace(temporary_path, model_path + ".zip")

    def select_action(self, state: dict, misc=None) -> int:
        action, _ = self.model.predict(
            state_to_observation(state),
            deterministic=True,
        )
        action_value = CONTROL_ACTION_VALUES[int(action)]
        self.last_decision_info = {"source": "learned", "epsilon": 0.0, "hold_steps_remaining": 0}
        return action_value

    def offline_update(self, batch):
        for transition in batch:
            self.model.replay_buffer.add(
                transition.state, transition.next_state,
                np.asarray([transition.action]), np.asarray([transition.reward], dtype=np.float32),
                np.asarray([False]), [{}],
            )
        if self.model.replay_buffer.size() >= self.model.batch_size:
            self.model.train(gradient_steps=1, batch_size=self.model.batch_size)

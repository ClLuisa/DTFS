import json
import os
from collections import deque, namedtuple

import numpy as np

from connect_protocols.simulation_env import SimulationEnv

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
except ImportError:
    DQN = None
    BaseCallback = object
    configure = None


CONTROL_ACTION_VALUES = [0, 2, 1, -1]
N_OBSERVATIONS = 3
N_ACTIONS = len(CONTROL_ACTION_VALUES)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class PersistentDQN(DQN):
    """DQN that applies the environment's action-persistence curriculum."""

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
    return np.asarray([
        state.get("dry_bulb_temperature", 0.0) / 40.0,
        state.get("total_horizontal_radiation", 0.0) / 1000.0,
        state.get("operative_temperature", 0.0) / 40.0,
    ], dtype=np.float32)


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

    def __init__(self, checkpoint_path: str, env=None):
        if DQN is None:
            raise ImportError(
                "Deep RL requires gymnasium and stable-baselines3. "
                "Install them with: pip install gymnasium stable-baselines3[extra]"
            )
        self.checkpoint_path = checkpoint_path
        self.env = env or SimulationEnv()
        self.model = self._load_or_create()
        self.model.persistence_env = self.env
        self.model.set_logger(configure(folder=None, format_strings=[]))
        self.last_state = None
        self.last_action = None

    def _model_path(self):
        return self.checkpoint_path.removesuffix(".pt")

    def _load_or_create(self):
        model_path = self._model_path()
        if os.path.exists(model_path + ".zip"):
            return PersistentDQN.load(model_path, env=self.env, device="auto")
        return PersistentDQN(
            "MlpPolicy", self.env,
            learning_rate=3e-4, buffer_size=10_000, learning_starts=64,
            batch_size=64, train_freq=1, gradient_steps=1,
            target_update_interval=250,
            exploration_initial_eps=0.85, exploration_final_eps=0.05,
            exploration_fraction=0.25, policy_kwargs={"net_arch": [64, 64]},
            verbose=0, device="auto",
        )

    def learn(self, total_timesteps: int):
        try:
            metrics_path = os.path.join(
                os.path.dirname(self.checkpoint_path), "training_metrics.jsonl"
            )
            self.model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=TrainingMetricsCallback(metrics_path),
            )
        finally:
            self._save_model()
            self.env.set_learning_done()

    def _save_model(self):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        self.model.save(self._model_path())

    def select_action(self, state: dict) -> int:
        action, _ = self.model.predict(state_to_observation(state), deterministic=True)
        self.last_decision_info = {"source": "learned", "epsilon": 0.0, "hold_steps_remaining": 0}
        return CONTROL_ACTION_VALUES[int(action)]

    def offline_update(self, batch):
        for transition in batch:
            self.model.replay_buffer.add(
                transition.state, transition.next_state,
                np.asarray([transition.action]), np.asarray([transition.reward], dtype=np.float32),
                np.asarray([False]), [{}],
            )
        if self.model.replay_buffer.size() >= self.model.batch_size:
            self.model.train(gradient_steps=1, batch_size=self.model.batch_size)

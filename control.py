import os, json
from threading import Thread
from config import RESULTS_PATH

from control_strategies.hard_coded import hard_coded_control
from control_strategies.random import random_control
from control_strategies.deep_rl import DeepQAgentTorch
from connect_protocols.simulation_env import SimulationEnv


class CheckpointMixin:
    """Anything with self.training_type / self.training_source / self.results_path gets a checkpoint path."""
    def _checkpoint_path(self) -> str:
        return os.path.join(
            self.results_path, "models", self.training_type, self.training_source,
            f"{self.training_type}_model.pt"
        )


class StateStorageMixin:
    """Anything that logs (state, control) pairs to a JSONL file."""
    def _state_storage_path(self) -> str:
        raise NotImplementedError

    def _init_state_storage(self):
        self.state_storage_path = self._state_storage_path()
        os.makedirs(os.path.dirname(self.state_storage_path), exist_ok=True)

    def reset_state_storage(self):
        with open(self.state_storage_path, "w") as f:
            pass

    def save_state_to_state_storage(self, data: dict):
        with open(self.state_storage_path, "a") as f:
            f.write(json.dumps(data) + "\n")

class Control(StateStorageMixin, CheckpointMixin):
    def __init__(self, data_type: str, control_type: str, training_type: str | None = None,
                 training_source: str | None = None, run_name: str = "default", results_path=RESULTS_PATH, env=None):

        self.env = env or SimulationEnv()
        self.data_type = data_type
        self.control_type = control_type
        self.training_type = training_type
        self.training_source = training_source
        self.run_name = run_name
        self.results_path = results_path

        self._init_state_storage()

        if control_type == "deep_rl":
            self.agent = DeepQAgentTorch(self._checkpoint_path(), env=self.env)

    def _state_storage_path(self):
        if self.control_type == "deep_rl":
            return os.path.join(self.results_path, "control", self.data_type, self.run_name,
                                 f"control_{self.training_type}_{self.training_source}.jsonl")
        return os.path.join(self.results_path, "control", self.data_type, self.run_name,
                             f"control_{self.control_type}.jsonl")

    def return_control(self, state: dict, misc=None):
        if self.control_type == "random":
            return random_control(state, misc)
        elif self.control_type == "hard_coded":
            return hard_coded_control(state)
        elif self.control_type == "all_closed":
            return 0
        elif self.control_type == "all_open":
            return 2
        elif self.control_type == "inside_open":
            return 1
        elif self.control_type == "outside_open":
            return -1
        elif self.control_type == "deep_rl":
            return self.agent.select_action(state, misc)


class OnlineTrainer(StateStorageMixin, CheckpointMixin):
    def __init__(self, training_source: str, results_path=RESULTS_PATH, env=None,
                 total_timesteps: int | None = None):
        if total_timesteps is not None and total_timesteps < 0:
            raise ValueError("total_timesteps must be non-negative or None")

        self.training_type = "online"
        self.training_source = training_source
        self.results_path = results_path
        self.total_timesteps = total_timesteps
        self.training_steps = 0

        self._init_state_storage()

        self.env = env or SimulationEnv()
        self.agent = DeepQAgentTorch(
            self._checkpoint_path(),
            env=self.env,
        )
        self.env.action_provider = self.agent.select_action
        self.last_state = None
        self.last_action = None
        self.learning_thread = None
        if total_timesteps:
            self.learning_thread = Thread(
                target=self.agent.learn, args=(total_timesteps,), daemon=True
            )
            self.learning_thread.start()

    def _state_storage_path(self):
        return os.path.join(self.results_path, "models", self.training_type, self.training_source, "online.jsonl")

    def return_control(self, state: dict, misc=None):
        control_signal = self.env.submit_state(state, misc)
        self.last_state = state
        self.training_steps = self.agent.model.num_timesteps
        self.last_decision_info = dict(
            getattr(self.agent, "last_decision_info", {})
            or getattr(self.env, "last_decision_info", {})
        )

        return control_signal


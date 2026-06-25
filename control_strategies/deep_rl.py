import os
import math
import random
import json
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import STATE_STORAGE_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTROL_ACTION_VALUES = [0, 2, 1, -1]
WANTED_TEMPERATURE = 21.0
DEADBAND = 0.5
MODEL_FILENAME = "deep_rl_pytorch_model.pt"
MODEL_PATH = os.path.join(STATE_STORAGE_PATH, MODEL_FILENAME)
TRAINING_LOG_PATH = os.path.join(STATE_STORAGE_PATH, "deep_rl_pytorch_training_log.jsonl")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
GAMMA = 0.98
EPS_START = 0.85
EPS_END = 0.05
EPS_DECAY = 2000
TAU = 0.005
LR = 3e-4

N_OBSERVATIONS = 4
N_ACTIONS = len(CONTROL_ACTION_VALUES)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# ---------------------------------------------------------------------------
# State normalisation
# ---------------------------------------------------------------------------
def normalize_state(state: dict) -> torch.Tensor:
    arr = np.array([
        state.get("dry_bulb_temperature", 0.0) / 40.0,
        state.get("total_horizontal_radiation", 0.0) / 1000.0,
        state.get("operative_temperature", 0.0) / 40.0,
        state.get("hour_of_day", 0.0),
    ], dtype=np.float32)
    return torch.tensor(arr, device=device).unsqueeze(0)

# ---------------------------------------------------------------------------
# Replay memory
# ---------------------------------------------------------------------------
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int = 10_000):
        self.memory: deque = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class DeepQAgentTorch:
    def __init__(self):
        self.policy_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
        self.target_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10_000)

        self.steps_done = 0

        self.last_state: torch.Tensor | None = None
        self.last_action: torch.Tensor | None = None
        self.last_raw_state: dict | None = None

        self.prev_raw_state: dict | None = None
        self.prev_action: int | None = None

        # Open training log in append mode
        self._log_file = open(TRAINING_LOG_PATH, "a", encoding="utf-8")

        self._load_model()

    def _log(self, record: dict):
        """Write a JSON record to the training log file."""
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_model(self):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, MODEL_PATH)

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            return
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.steps_done = checkpoint.get("steps_done", 0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Epsilon
    # ------------------------------------------------------------------
    def _epsilon(self) -> float:
        return EPS_END + (EPS_START - EPS_END) * math.exp(-self.steps_done / EPS_DECAY)

    def choose_action(self, state: dict) -> int:
        state_tensor = normalize_state(state)
        self.steps_done += 1

        epsilon = self._epsilon()
        if random.random() < epsilon:
            action_index = random.randrange(N_ACTIONS)
            exploratory = True
        else:
            with torch.no_grad():
                action_index = int(self.policy_net(state_tensor).max(1).indices.item())
            exploratory = False

        self._log({
            "type": "action",
            "step": self.steps_done,
            "epsilon": round(epsilon, 4),
            "action_index": action_index,
            "action_value": CONTROL_ACTION_VALUES[action_index],
            "exploratory": exploratory,
            "operative_temperature": state.get("operative_temperature"),
            "dry_bulb_temperature": state.get("dry_bulb_temperature"),
            "total_horizontal_radiation": state.get("total_horizontal_radiation"),
        })

        self.prev_raw_state = self.last_raw_state
        self.prev_action = (
            int(self.last_action.item()) if self.last_action is not None else None
        )
        self.last_state = state_tensor
        self.last_action = torch.tensor([[action_index]], device=device, dtype=torch.long)
        self.last_raw_state = dict(state)

        return action_index

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------
    def update(self, next_state: dict, reward: float):
        if self.last_state is None or self.last_action is None:
            return

        next_state_tensor = normalize_state(next_state)
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)

        self.memory.push(self.last_state, self.last_action, next_state_tensor, reward_tensor)

        loss = self._optimize()
        self._soft_update_target()
        self._save_model()

        self._log({
            "type": "update",
            "step": self.steps_done,
            "reward": round(reward, 6),
            "loss": round(loss, 6) if loss is not None else None,
            "memory_size": len(self.memory),
        })

    # ------------------------------------------------------------------
    # Mini-batch SGD
    # ------------------------------------------------------------------
    def _optimize(self) -> float | None:
        if len(self.memory) < BATCH_SIZE:
            return None

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(
            state_action_values,
            expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Soft target-network update
    # ------------------------------------------------------------------
    def _soft_update_target(self):
        target_sd = self.target_net.state_dict()
        policy_sd = self.policy_net.state_dict()
        for key in policy_sd:
            target_sd[key] = policy_sd[key] * TAU + target_sd[key] * (1 - TAU)
        self.target_net.load_state_dict(target_sd)

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------
    @staticmethod
    def get_reward(previous_state: dict, next_state: dict, action_index: int) -> float:
        prev_err = abs(previous_state["operative_temperature"] - WANTED_TEMPERATURE)
        next_err = abs(next_state["operative_temperature"] - WANTED_TEMPERATURE)

        prev_err_clipped = max(0.0, prev_err - DEADBAND)
        next_err_clipped = max(0.0, next_err - DEADBAND)

        comfort_improvement = prev_err_clipped - next_err_clipped

        return float(comfort_improvement)


# ---------------------------------------------------------------------------
# Module-level singleton + public API
# ---------------------------------------------------------------------------
_agent = DeepQAgentTorch()


def deep_rl_control(state: dict) -> int:
    if _agent.prev_raw_state is not None and _agent.prev_action is not None:
        reward = _agent.get_reward(_agent.prev_raw_state, state, _agent.prev_action)
        _agent.update(state, reward)

    action_index = _agent.choose_action(state)
    return CONTROL_ACTION_VALUES[action_index]
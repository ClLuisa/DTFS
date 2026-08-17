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

from config import RESULTS_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_OBSERVATIONS = 3
CONTROL_ACTION_VALUES = [0, 2, 1, -1]
N_ACTIONS = len(CONTROL_ACTION_VALUES)

MODEL_FILENAME = "deep_rl_pytorch_model.pt"
MODEL_PATH = os.path.join(RESULTS_PATH, MODEL_FILENAME)

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
        state.get("operative_temperature", 0.0) / 40.0
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

    def iter_batches(self, batch_size: int):
        for start in range(0, len(self.memory), batch_size):
            chunk = list(self.memory)[start:start + batch_size]
            if not chunk:
                continue

            state_batch = torch.cat([t.state for t in chunk])
            action_batch = torch.cat([t.action for t in chunk])
            next_state_batch = torch.cat([t.next_state for t in chunk])
            reward_batch = torch.cat([t.reward for t in chunk])

            yield state_batch, action_batch, next_state_batch, reward_batch

    def __len__(self):
        return len(self.memory)

def load_transitions_from_jsonl(
    filepath: str, 
    reward_fn=None
) -> ReplayMemory:

    memory = ReplayMemory()
    
    # Read all entries
    entries = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return memory
    
    # Create transitions from consecutive entries
    for i in range(len(entries) - 1):
        current_entry = entries[i]
        next_entry = entries[i + 1]
        
        current_state = current_entry.get("state", {})
        next_state = next_entry.get("state", {})
        action = current_entry.get("control", 0)
        try:
            action_index = CONTROL_ACTION_VALUES.index(action)
        except ValueError:
            action_index = 0
        
        # Normalize states to tensors
        state_tensor = normalize_state(current_state)
        next_state_tensor = normalize_state(next_state)
        
        # Convert action to tensor
        action_tensor = torch.tensor([[action_index]], device=device, dtype=torch.long)
        
        # Compute reward
        if reward_fn is not None:
            reward = reward_fn(current_state, action, next_state, current_entry.get("misc", {}))
        else:
            reward = 0.0
        
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
        
        # Add transition to memory
        memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
    
    return memory

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

class DeepQAgentTorch:
    def __init__(self, checkpoint_path: str, mode: str = "eval"):
        self.checkpoint_path = checkpoint_path
        self.policy_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
        self.target_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.mode = mode
        if self.mode == "eval":
            self.policy_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=3e-4, amsgrad=True)
        self.memory = ReplayMemory(10_000)

        self.steps_done = 0

        self.last_state: torch.Tensor | None = None
        self.last_action: torch.Tensor | None = None

        self._load_model()

    def set_mode(self, mode: str):
        self.mode = mode
        self.policy_net.train() if mode == "train" else self.policy_net.eval()

    def _save_model(self):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, self.checkpoint_path)

    def _load_model(self):
        if not os.path.exists(self.checkpoint_path):
            return
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.steps_done = checkpoint.get("steps_done", 0)
        except Exception:
            pass

    def select_action(self, state: dict) -> int:
        """Eval-mode action: argmax, no exploration, no state mutation."""
        state_tensor = normalize_state(state)
        with torch.no_grad():
            return CONTROL_ACTION_VALUES[int(self.policy_net(state_tensor).max(1).indices.item())]

    # ---- online training ----
    def act_and_remember(self, state: dict) -> int:
        """Exploratory action for online rollout; tracks last_state/last_action for the next update().

        Returns the actual control signal sent to the environment, but stores the internal
        network action index for the DQN update.
        """
        state_tensor = normalize_state(state)
        self.steps_done += 1
        epsilon = self._epsilon()
        if random.random() < epsilon:
            action_index = random.randrange(N_ACTIONS)
        else:
            with torch.no_grad():
                action_index = int(self.policy_net(state_tensor).max(1).indices.item())

        self.last_state = state_tensor
        self.last_action = torch.tensor([[action_index]], device=device, dtype=torch.long)
        return CONTROL_ACTION_VALUES[action_index]

    def online_update(self, next_state: dict, reward: float):
        if self.last_state is None:
            return
        next_state_tensor = normalize_state(next_state)
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
        self.memory.push(self.last_state, self.last_action, next_state_tensor, reward_tensor)
        self._optimize()
        self._soft_update_target()
        self._save_model()

    # ---- offline / batch training ----
    def offline_update(self, batch):
        """One gradient step on a pre-sampled batch from the stored dataset (no env interaction)."""
        state_batch, action_batch, next_state_batch, reward_batch = batch
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1).values
        expected = reward_batch + 0.98 * next_state_values
        loss = F.smooth_l1_loss(state_action_values, expected.unsqueeze(1))

        # NOTE: plain DQN loss here will overestimate Q-values for actions never
        # taken from a given state in the logged data (extrapolation error).
        # Worth adding a CQL-style penalty term before trusting this offline —
        # happy to sketch that separately if useful.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update_target()

    def _epsilon(self) -> float:

        EPS_START = 0.85
        EPS_END = 0.05
        EPS_DECAY = 1500

        if self.mode == "eval":
            return 0.0
        return EPS_END + (EPS_START - EPS_END) * math.exp(-self.steps_done / EPS_DECAY)

    def _optimize(self) -> float | None:

        BATCH_SIZE = 64
        GAMMA = 0.98

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

    def _soft_update_target(self):
        TAU = 0.005
        target_sd = self.target_net.state_dict()
        policy_sd = self.policy_net.state_dict()
        for key in policy_sd:
            target_sd[key] = policy_sd[key] * TAU + target_sd[key] * (1 - TAU)
        self.target_net.load_state_dict(target_sd)

    @staticmethod
    def get_reward(previous_state: dict, next_state: dict, action_index: int) -> float:

        WANTED_TEMPERATURE = 21.0
        DEADBAND = 0.5

        prev_err = abs(previous_state["operative_temperature"] - WANTED_TEMPERATURE)
        next_err = abs(next_state["operative_temperature"] - WANTED_TEMPERATURE)

        prev_err_clipped = max(0.0, prev_err - DEADBAND)
        next_err_clipped = max(0.0, next_err - DEADBAND)

        comfort_improvement = prev_err_clipped - next_err_clipped

        return float(comfort_improvement)
import os
import json
import numpy as np

from config import STATE_STORAGE_PATH

CONTROL_ACTION_VALUES = [0, 2, 1, -1]
WANTED_TEMPERATURE = 21.0
MODEL_FILENAME = "deep_rl_model.npz"
MODEL_PATH = os.path.join(STATE_STORAGE_PATH, MODEL_FILENAME)


def normalize_state(state: dict) -> np.ndarray:
    return np.array([
        state.get("dry_bulb_temperature", 0.0) / 40.0,
        state.get("total_horizontal_radiation", 0.0) / 1000.0,
        state.get("operative_temperature", 0.0) / 40.0,
        state.get("hour_of_day", 0.0),
    ], dtype=np.float32)


class DeepQAgent:
    def __init__(self, state_size=4, action_size=4, hidden_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.epsilon = 0.05 #0.85
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.98
        self.learning_rate = 0.007
        self.steps = 0

        self.W1 = np.random.randn(self.hidden_size, self.state_size) * 0.1
        self.b1 = np.zeros((self.hidden_size,), dtype=np.float32)
        self.W2 = np.random.randn(self.action_size, self.hidden_size) * 0.1
        self.b2 = np.zeros((self.action_size,), dtype=np.float32)

        self.prev_state = None
        self.prev_action = None
        self.prev_raw_state = None
        self.last_state = None
        self.last_action = None
        self.last_raw_state = None

        self._load_model()

    def _save_model(self):
        np.savez(
            MODEL_PATH,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            epsilon=self.epsilon,
        )

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            return

        try:
            data = np.load(MODEL_PATH)
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]
            self.epsilon = float(data.get("epsilon", self.epsilon))
        except Exception:
            pass

    def _forward(self, state_vector: np.ndarray):
        hidden = np.tanh(self.W1.dot(state_vector) + self.b1)
        q_values = self.W2.dot(hidden) + self.b2
        return q_values, hidden

    def choose_action(self, state: dict) -> int:
        x = normalize_state(state)
        q_values, _ = self._forward(x)

        self.steps += 1
        if np.random.rand() < self.epsilon:
            action_index = int(np.random.randint(self.action_size))
        else:
            action_index = int(np.argmax(q_values))

        self.prev_state = self.last_state
        self.prev_action = self.last_action
        self.prev_raw_state = self.last_raw_state
        self.last_state = x
        self.last_action = action_index
        self.last_raw_state = dict(state)
        return action_index

    def update(self, next_state: dict, reward: float, done: bool = False):
        if self.prev_state is None or self.prev_action is None:
            return

        next_x = normalize_state(next_state)
        current_q, hidden = self._forward(self.prev_state)
        next_q, _ = self._forward(next_x)

        target_q = current_q.copy()
        if done:
            target_q[self.last_action] = reward
        else:
            target_q[self.last_action] = reward + self.gamma * float(np.max(next_q))

        error = current_q - target_q
        grad_output = error
        dW2 = np.outer(grad_output, hidden)
        db2 = grad_output
        dh = self.W2.T.dot(grad_output) * (1.0 - hidden ** 2)
        dW1 = np.outer(dh, self.last_state)
        db1 = dh

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._save_model()

    def get_reward(self, previous_state: dict, next_state: dict, action_index: int) -> float:
        previous_error = abs(previous_state["operative_temperature"] - WANTED_TEMPERATURE)
        next_error = abs(next_state["operative_temperature"] - WANTED_TEMPERATURE)
        comfort_improvement = previous_error - next_error

        action_cost = 0.01 * abs(CONTROL_ACTION_VALUES[action_index])
        reward = comfort_improvement - action_cost

        if next_state["total_horizontal_radiation"] > 200 and CONTROL_ACTION_VALUES[action_index] == 1:
            reward += 0.02
        if next_state["total_horizontal_radiation"] > 200 and CONTROL_ACTION_VALUES[action_index] == -1:
            reward -= 0.02

        return float(reward)


_agent = DeepQAgent()


def deep_rl_control(state: dict) -> int:
    if _agent.prev_raw_state is not None and _agent.prev_action is not None:
        reward = _agent.get_reward(_agent.prev_raw_state, state, _agent.prev_action)
        _agent.update(state, reward, done=False)

    action_index = _agent.choose_action(state)
    return CONTROL_ACTION_VALUES[action_index]


def pretrain_from_jsonl(file_name: str, epochs: int = 1):
    path = os.path.join(STATE_STORAGE_PATH, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"State storage file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        states = [json.loads(line) for line in file if line.strip()]

    for _ in range(epochs):
        for previous, current in zip(states, states[1:]):
            prev_features = {
                "operative_temperature": previous["operative_temperature"],
                "total_horizontal_radiation": previous["total_horizontal_radiation"],
            }
            reward = _agent.get_reward(prev_features, current, int(np.random.randint(len(CONTROL_ACTION_VALUES))))
            _agent.last_state = normalize_state(previous)
            _agent.last_action = int(np.random.randint(len(CONTROL_ACTION_VALUES)))
            _agent.update(current, reward)

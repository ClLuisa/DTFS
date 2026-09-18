"""Thread-safe Gymnasium bridge between SB3 and the TRNSYS callback."""

import random
from threading import Condition

import gymnasium as gym
import numpy as np
from gymnasium import spaces


CONTROL_ACTION_VALUES = (0, 2, 1, -1)
OBSERVATION_SIZE = 3


class SimulationEnv(gym.Env):
    """Expose TRNSYS states as a synchronous Gymnasium environment.

    SB3 runs in a worker thread. TRNSYS calls :meth:`submit_state` once per
    timestep; the method returns the action selected by SB3 for that state.
    """

    metadata = {"render_modes": []}

    def __init__(self, action_provider=None, reward: str = "default",
                 comfort_band: tuple[float, float] = (20, 22, 24),
                 radiation_temperature_scale: float = 500.0,
                 action_hold_mode: bool = False,
                 min_hold_steps: int = 1, max_hold_steps: int = 15,
                 hold_probability: float = 0.8, late_max_hold_steps: int = 2,
                 late_hold_probability: float = 0.1,
                 training_timesteps: int | None = None):
        super().__init__()

        if min_hold_steps < 1 or max_hold_steps < min_hold_steps:
            raise ValueError("hold steps must satisfy 1 <= min <= max")
        if late_max_hold_steps < 1 or not 0 <= hold_probability <= 1 or not 0 <= late_hold_probability <= 1:
            raise ValueError("invalid action hold configuration")

        self.reward_mode = reward
        self.comfort_band = comfort_band
        if radiation_temperature_scale <= 0:
            raise ValueError("radiation_temperature_scale must be positive")
        self.radiation_temperature_scale = radiation_temperature_scale
        self.action_hold_mode = action_hold_mode
        self.min_hold_steps = min_hold_steps
        self.max_hold_steps = max_hold_steps
        self.hold_probability = hold_probability
        self.late_max_hold_steps = late_max_hold_steps
        self.late_hold_probability = late_hold_probability
        self.training_timesteps = training_timesteps
        self._held_action = None
        self._hold_remaining = 0
        self._hold_elapsed = 0
        self.action_space = spaces.Discrete(len(CONTROL_ACTION_VALUES))
        self.observation_space = spaces.Box(
            low=np.full(OBSERVATION_SIZE, -100.0, dtype=np.float32),
            high=np.full(OBSERVATION_SIZE, 100.0, dtype=np.float32),
            dtype=np.float32,
        )
        self._condition = Condition()
        self._state = None
        self._misc = {}
        self._next_state = None
        self._next_misc = {}
        self._pending_action = None
        self._closed = False
        self._learning_done = False
        self.action_provider = action_provider
        self.last_decision_info = {}

    @staticmethod
    def observation(state: dict, misc=None, previous_temperature=None,
                    previous_action=0, hold_duration=0) -> np.ndarray:
        return np.asarray([
            state.get("dry_bulb_temperature", 0.0) / 40.0,
            state.get("total_horizontal_radiation", 0.0) / 1000.0,
            state.get("operative_temperature", 0.0) / 40.0,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        with self._condition:
            options = options or {}
            if "state" in options:
                self._state = options["state"]
                self._misc = options.get("misc", {})
            else:
                while self._state is None and not self._closed:
                    self._condition.wait()
            if self._closed:
                raise RuntimeError("SimulationEnv was closed")
            self._hold_elapsed = 0
            return self.observation(self._state, self._misc), dict(self._misc)

    def step(self, action):
        with self._condition:
            self._pending_action = int(action)
            self._condition.notify_all()
            while self._next_state is None and not self._closed:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("SimulationEnv was closed")

            previous_state = self._state
            self._state = self._next_state
            self._misc = self._next_misc
            self._next_state = None
            reward = self._reward(previous_state, self._state)
            return self.observation(
                self._state,
                self._misc,
                previous_temperature=previous_state["operative_temperature"],
                previous_action=CONTROL_ACTION_VALUES[int(action)],
                hold_duration=self._hold_elapsed,
            ), reward, False, False, dict(self._misc)

    def submit_state(self, state: dict, misc=None) -> int:
        """Submit a TRNSYS state and wait for SB3's action for it."""
        with self._condition:
            if self._state is None:
                self._state = state
                self._misc = misc or {}
            else:
                self._next_state = state
                self._next_misc = misc or {}
            self._condition.notify_all()
            while self._pending_action is None and not self._closed and not self._learning_done:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("SimulationEnv was closed")
            if self._learning_done and self._pending_action is None:
                if self.action_provider is None:
                    return CONTROL_ACTION_VALUES[0]
                return self.action_provider(state)
            action = self._pending_action
            self._pending_action = None
            self._condition.notify_all()
            return CONTROL_ACTION_VALUES[action]

    def set_learning_done(self):
        with self._condition:
            self._learning_done = True
            self._condition.notify_all()

    def resolve_action(self, action, training_step=0, source="learned", epsilon=0.0):
        """Keep an SB3-selected action for a sampled number of steps."""
        if not self.action_hold_mode:
            self.last_decision_info = {
                "source": source,
                "epsilon": float(epsilon),
                "hold_steps_remaining": 0,
            }
            return int(action)
        if self._hold_remaining:
            self._hold_remaining -= 1
            self._hold_elapsed += 1
            self.last_decision_info = {
                "source": "held",
                "epsilon": float(epsilon),
                "hold_steps_remaining": self._hold_remaining,
            }
            return self._held_action

        progress = 0.0
        if self.training_timesteps:
            progress = min(1.0, training_step / self.training_timesteps)
        probability = self.hold_probability + progress * (
            self.late_hold_probability - self.hold_probability
        )
        maximum = round(self.max_hold_steps + progress * (
            self.late_max_hold_steps - self.max_hold_steps
        ))
        selected_action = int(action)
        if random.random() < probability:
            duration = random.randint(self.min_hold_steps, max(self.min_hold_steps, maximum))
            self._held_action = selected_action
            self._hold_remaining = duration - 1
            self._hold_elapsed = 1
        else:
            self._held_action = None
            self._hold_elapsed = 0
        self.last_decision_info = {
            "source": source,
            "epsilon": float(epsilon),
            "hold_steps_remaining": self._hold_remaining,
        }
        return selected_action

    def _reward(self, previous_state: dict, next_state: dict) -> float:
        previous_op_temperature = previous_state["operative_temperature"]
        next_op_temperature = next_state["operative_temperature"]

        previous_amb_temperature = previous_state["dry_bulb_temperature"]
        next_amb_temperature = next_state["dry_bulb_temperature"]
        previous_radiation = previous_state.get("total_horizontal_radiation", 0.0)
        next_radiation = next_state.get("total_horizontal_radiation", previous_radiation)

        previous_radiation = previous_state["total_horizontal_radiation"]
        next_radiation = next_state["total_horizontal_radiation"]

        last_inside_comfort_band = self.comfort_band[0] <= previous_op_temperature <= self.comfort_band[1]
        next_inside_comfort_band = self.comfort_band[0] <= next_op_temperature <= self.comfort_band[1]

        if last_inside_comfort_band and not next_inside_comfort_band:
            reward = -1.0
        elif not last_inside_comfort_band and next_inside_comfort_band:
            reward = 1.0
        elif last_inside_comfort_band and next_inside_comfort_band:
            if "sensitive" in self.reward_mode:

                previous_error = abs(previous_op_temperature - self.comfort_band[1])
                next_error = abs(next_op_temperature - self.comfort_band[1])
                reward = (float(previous_error - next_error) + 2) / 4

            else:
                reward = 0.05
        else:
            previous_error = min(
                abs(previous_op_temperature - self.comfort_band[0]),
                abs(previous_op_temperature - self.comfort_band[2]),
            )
            next_error = min(
                abs(next_op_temperature - self.comfort_band[0]),
                abs(next_op_temperature - self.comfort_band[2]),
            )
            reward = float(previous_error - next_error)

        if "default" in self.reward_mode:
            pass

        if "ambient_adjusted" in self.reward_mode:

            ambient_temperature_change = next_amb_temperature - previous_amb_temperature
            radiation_change = next_radiation - previous_radiation
            operative_temperature_change = next_op_temperature - previous_op_temperature

            alignment = self._ambient_alignment_factor(
                ambient_temperature_change,
                operative_temperature_change,
                radiation_change=radiation_change,
            )

            max_discount = 0.8
            attribution_weight = 1.0 - max_discount * alignment

            reward *= attribution_weight

        if self.reward_mode == "ambient_adjusted_sensitive":

            ambient_temperature_change = next_amb_temperature - previous_amb_temperature
            radiation_change = next_radiation - previous_radiation
            operative_temperature_change = next_op_temperature - previous_op_temperature

            alignment = self._ambient_alignment_factor(
                ambient_temperature_change,
                operative_temperature_change,
                radiation_change=radiation_change,
            )

            max_discount = 0.8
            attribution_weight = 1.0 - max_discount * alignment

            reward *= attribution_weight

        return reward

    def _ambient_alignment_factor(
        self,
        ambient_change: float,
        operative_change: float,
        radiation_change: float = 0.0,
        min_change: float = 0.02,
    ) -> float:
        """
        Returns a value in [0, 1] indicating how much of the operative temperature
        change plausibly just mirrors the ambient temperature change (i.e., is NOT
        attributable to the flap action). 0 = no shared movement / opposite
        directions, 1 = moved together at the same rate.
        """
        environmental_change = (
            ambient_change
            + radiation_change / self.radiation_temperature_scale
        )

        # Too small to say anything meaningful about direction/rate.
        if abs(environmental_change) < min_change or abs(operative_change) < min_change:
            return 0.0

        same_direction = (environmental_change > 0) == (operative_change > 0)
        if not same_direction:
            return 0.0

        magnitude_ratio = min(abs(operative_change), abs(environmental_change)) / max(
            abs(operative_change), abs(environmental_change)
        )
        return magnitude_ratio

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()

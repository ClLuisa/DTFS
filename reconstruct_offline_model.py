import argparse
import json
import os

import numpy as np

from connect_protocols.simulation_env import SimulationEnv
from control_strategies.deep_rl import CONTROL_ACTION_VALUES, DeepQAgentTorch, state_to_observation


ACTION_INDEX = {value: index for index, value in enumerate(CONTROL_ACTION_VALUES)}
STEPS_PER_DAY = 24 * 60 // 10

# Edit these values before running this file directly.
SOURCE_METRICS = "drl_ambient_adjusted_measured"
SOURCE_DIR = fr"results\models\online\{SOURCE_METRICS}"
DAYS = 7
OUTPUT_PATH = fr"results\models\offline\{SOURCE_METRICS}_{DAYS}_days\offline_model"


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_transitions(state_rows, metric_rows, days=None):
    if len(state_rows) < 2:
        raise ValueError("online.jsonl must contain at least two state records")
    if not metric_rows:
        raise ValueError("training_metrics.jsonl is empty")

    available = min(len(state_rows) - 1, max(int(row["timesteps"]) for row in metric_rows))
    if days is not None:
        if days <= 0:
            raise ValueError("days must be greater than zero")
        available = min(available, int(days * STEPS_PER_DAY))
    transitions = []
    action_mismatches = 0

    for index in range(available):
        current = state_rows[index]
        following = state_rows[index + 1]
        action_value = int(current["control"])
        if action_value not in ACTION_INDEX:
            raise ValueError(f"Unsupported control value at row {index}: {action_value}")

        metric_action = metric_rows[index].get("action_index") if index < len(metric_rows) else None
        if metric_action is not None and int(metric_action) != ACTION_INDEX[action_value]:
            action_mismatches += 1

        transitions.append(
            (
                state_to_observation(current["state"]),
                state_to_observation(following["state"]),
                ACTION_INDEX[action_value],
            )
        )

    return transitions, action_mismatches


def reconstruct(source_dir, output_path, days=None):
    state_path = os.path.join(source_dir, "online.jsonl")
    metrics_path = os.path.join(source_dir, "training_metrics.jsonl")
    state_rows = read_jsonl(state_path)
    metric_rows = read_jsonl(metrics_path)
    transitions, action_mismatches = build_transitions(state_rows, metric_rows, days=days)

    environment = SimulationEnv(reward="ambient_adjusted")
    agent = DeepQAgentTorch(output_path, env=environment)
    model = agent.model
    model.num_timesteps = 0

    for index, (current_observation, next_observation, action_index) in enumerate(transitions):
        current_state = state_rows[index]["state"]
        next_state = state_rows[index + 1]["state"]
        reward = environment._reward(current_state, next_state)
        model.replay_buffer.add(
            np.asarray([current_observation], dtype=np.float32),
            np.asarray([next_observation], dtype=np.float32),
            np.asarray([action_index]),
            np.asarray([reward], dtype=np.float32),
            np.asarray([False]),
            [{}],
        )
        model.num_timesteps += 1
        if model.replay_buffer.size() >= model.batch_size:
            model.train(gradient_steps=1, batch_size=model.batch_size)

    agent._save_model()
    return len(transitions), action_mismatches, output_path.removesuffix(".pt") + ".zip"


def main():
    parser = argparse.ArgumentParser(description="Reconstruct an approximate offline DQN checkpoint from JSONL logs.")
    parser.add_argument("source_dir", nargs="?", default=SOURCE_DIR, help="Folder containing online.jsonl and training_metrics.jsonl")
    parser.add_argument("--output", default=None, help="Checkpoint .pt stem; .zip is written beside it")
    parser.add_argument(
        "--days",
        type=float,
        default=DAYS,
        help="Number of days to reconstruct from the start; each day contains 144 ten-minute steps",
    )
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    output_path = args.output or OUTPUT_PATH or os.path.join(source_dir, "online_model.pt")
    transitions, mismatches, checkpoint = reconstruct(source_dir, output_path, days=args.days)
    print(f"Replayed transitions: {transitions}")
    if args.days is not None:
        print(f"Requested days: {args.days}")
    print(f"Metric/control action mismatches: {mismatches}")
    print(f"Checkpoint written: {checkpoint}")


if __name__ == "__main__":
    main()
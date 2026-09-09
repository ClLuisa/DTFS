import requests
import time
import json
from datetime import datetime
import sys

sys.path.append(r"C:\Users\lkclu\Documents\GitHub\DTFS")

URL = "http://dtaf-core.taild0cac0.ts.net:1880/sensors"
INTERVAL_SECONDS = 10

sys.path.append(r"C:\Users\lkclu\Documents\GitHub\DTFS")

from control import Control, OnlineTrainer
from connect_protocols.simulation_env import SimulationEnv

RUN_MODE = "train"  # "train" or "eval"
ACTIVE_EXPERIMENT = "drl_ambient_adjusted"

EXPERIMENTS = {
    "random_control": {"controller": "random", "reward": "default"},
    "hard_coded_control": {"controller": "hard_coded", "reward": "default"},
    "drl_ambient_adjusted": {
        "controller": "deep_rl",
        "reward": "ambient_adjusted",
    },
}

experiment = EXPERIMENTS[ACTIVE_EXPERIMENT]
simulation_env = SimulationEnv(
    reward=experiment["reward"],
    training_timesteps=8_500,
)

if experiment["controller"] == "deep_rl" and RUN_MODE == "train":
    control = OnlineTrainer(
        training_source=ACTIVE_EXPERIMENT,
        env=simulation_env,
        total_timesteps=8_500,
    )
elif experiment["controller"] == "deep_rl" and RUN_MODE == "eval":
    control = Control(
        data_type="measurements",
        control_type="deep_rl",
        training_type="online",
        training_source=ACTIVE_EXPERIMENT,
        run_name=ACTIVE_EXPERIMENT,
        env=simulation_env,
    )
elif experiment["controller"] in {"random", "hard_coded"}:
    control = Control(
        data_type="measurements",
        control_type=experiment["controller"],
        run_name=ACTIVE_EXPERIMENT,
        env=simulation_env,
    )
else:
    raise ValueError("Unsupported RUN_MODE or controller")


while True:
    try:
        response = requests.get(URL, timeout=5)
        response.raise_for_status()
        data = response.json()

        dt = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        start_of_year = datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)

        print(f"\nTimestamp request: {datetime.now().isoformat()}")
        print(json.dumps(data, indent=2))
        print("---")

        current_state = {
            "dry_bulb_temperature": data["OUTDOOR_AIR_TEMP_C"],
            "total_horizontal_radiation": data["SOLAR_RADIATION_VERTICAL_W_M2"],
            "operative_temperature": data["INDOOR_AIR_TEMP_ANALOG_C"],
        }

        misc = {
            "hour_of_day": dt.hour + dt.minute/60 + dt.second/3600,
            "current_time": (dt - start_of_year).total_seconds() / 3600
        }

        next_control_signal = control.return_control(current_state, misc)
        control.save_state_to_state_storage({
            "state": current_state,
            "control": next_control_signal,
            "misc": misc,
            "decision": getattr(control, "last_decision_info", {}),
        })

    except Exception as e:
        print(f"Error reading sensors: {e}")

    time.sleep(INTERVAL_SECONDS)
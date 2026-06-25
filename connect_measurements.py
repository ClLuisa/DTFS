import requests
import time
import json
from datetime import datetime
import sys

sys.path.append(r"C:\Users\lkclu\Documents\GitHub\DTFS")

URL = "http://dtaf-core.taild0cac0.ts.net:1880/sensors"
INTERVAL_SECONDS = 10 # 1 minute

sys.path.append(r"C:\Users\lkclu\Documents\GitHub\DTFS")

from strategy import return_control
from state_storage import reset_state_storage, save_state_to_state_storage


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
            "hour_of_day": dt.hour + dt.minute/60 + dt.second/3600,
            "control": 0, # replace by correct number
            "current_time": (dt - start_of_year).total_seconds() / 3600
        }

        print('current_state: ', current_state)

        save_state_to_state_storage(current_state)

        next_control_signal = return_control(current_state, control_type="deep_rl")
        print('next_control_signal: ', next_control_signal)

    except Exception as e:
        print(f"Error reading sensors: {e}")

    time.sleep(INTERVAL_SECONDS)
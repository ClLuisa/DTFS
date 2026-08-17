import requests
import time
import json
from datetime import datetime
import sys

sys.path.append(r"C:\Users\lkclu\Documents\GitHub\DTFS")

URL = "http://dtaf-core.taild0cac0.ts.net:1880/sensors"
INTERVAL_SECONDS = 10 # 1 minute

sys.path.append(r"C:\Users\lkclu\Documents\GitHub\DTFS")

from control import Control


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

        control = Control(data_type="measurements", control_type="random", variant="v1") #replace variant with better name
        next_control_signal = control.return_control(current_state)
        control.save_state_to_state_storage({"state": current_state, "control": next_control_signal, "misc": misc})

    except Exception as e:
        print(f"Error reading sensors: {e}")

    time.sleep(INTERVAL_SECONDS)
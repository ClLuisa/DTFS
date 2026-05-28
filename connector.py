import random

from config import CONTROL_TYPE
from deep_rl import deep_rl_control

CONTROL_ACTIONS = {
    "all_closed": 0, # for winter night
    "all_open": 2, # for summer night
    "inside_open": 1, # for winter day (with radiation)
    "outside_open": -1 # for summer day
}

def return_control(state: dict):

    if CONTROL_TYPE == "random":
        return random_control(state)

    elif CONTROL_TYPE == "hard_coded":
        return hard_coded_control(state)

    elif CONTROL_TYPE == "all_closed":
        return 0

    elif CONTROL_TYPE == "all_open":
        return 2
    
    elif CONTROL_TYPE == "inside_open":
        return 1
    
    elif CONTROL_TYPE == "outside_open":
        return -1

    elif CONTROL_TYPE == "deep_rl_run_2":
        return deep_rl_control(state)


def hard_coded_control(state: dict):
        
    dry_bulb_temperature = state["dry_bulb_temperature"]
    total_horizontal_radiation = state["total_horizontal_radiation"]
    operative_temperature = state["operative_temperature"]
    control = state["control"]

    wanted_temperature = 21

    heating_needed = operative_temperature < wanted_temperature
    is_night = total_horizontal_radiation < 10

    if is_night:

        if heating_needed:
            return CONTROL_ACTIONS["all_closed"]   # conserve heat
        else:
            if dry_bulb_temperature < operative_temperature:
                return CONTROL_ACTIONS["all_open"]     # purge heat (cooling)
            else:
                return CONTROL_ACTIONS["all_closed"]

    # DAY CASES
    else:

        if heating_needed:
            # use solar gains if available
            if total_horizontal_radiation > 200:
                return CONTROL_ACTIONS["inside_open"]  # admit solar gains
            else:
                return CONTROL_ACTIONS["all_closed"]    # avoid losses

        else:
            # cooling mode
            if dry_bulb_temperature > operative_temperature:
                return CONTROL_ACTIONS["outside_open"]  # ventilate
            else:
                return CONTROL_ACTIONS["all_open"]      # purge / free cooling

    return control

def random_control(state: dict):
    current_time = state["current_time"]
    control = state["control"]

    if current_time % 2 == 0:
        control = random.choice(list(CONTROL_ACTIONS.values()))

    return control
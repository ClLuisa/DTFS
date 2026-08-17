from config import CONTROL_ACTIONS

_last_action = 0

def hard_coded_control(state: dict):
    global _last_action

    ambient_temperature = state["dry_bulb_temperature"]
    total_horizontal_radiation = state["total_horizontal_radiation"]
    operative_temperature = state["operative_temperature"]

    wanted_temperature = 21
    deadband = 0.5

    if operative_temperature < wanted_temperature - deadband:
        heating_needed = True
    elif operative_temperature > wanted_temperature + deadband:
        heating_needed = False
    else:
        return _last_action

    radiation = total_horizontal_radiation > 10

    if not radiation:
        if heating_needed:
            if ambient_temperature < operative_temperature:
                control = CONTROL_ACTIONS["all_closed"]
            else:
                control = CONTROL_ACTIONS["all_open"]
        else:
            if ambient_temperature < operative_temperature:
                control = CONTROL_ACTIONS["all_open"]
            else:
                control = CONTROL_ACTIONS["all_closed"]
    else: #radiation
        if heating_needed:
            if ambient_temperature < operative_temperature:
                control = CONTROL_ACTIONS["inside_open"]
            else:
                control = CONTROL_ACTIONS["all_open"]
        else: #cooling_needed
            if ambient_temperature < operative_temperature:
                control = CONTROL_ACTIONS["all_open"]
            else:
                control = CONTROL_ACTIONS["outside_open"]
                

    _last_action = control
    return control

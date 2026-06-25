from config import CONTROL_ACTIONS

def hard_coded_control(state: dict):
        
    dry_bulb_temperature = state["dry_bulb_temperature"]
    total_horizontal_radiation = state["total_horizontal_radiation"]
    operative_temperature = state["operative_temperature"]
    control = state["control"]

    wanted_temperature = 21
    deadband = 0.5

    # Only trigger heating/cooling outside the deadband
    if operative_temperature < wanted_temperature - deadband:
        heating_needed = True
    elif operative_temperature > wanted_temperature + deadband:
        heating_needed = False
    else:
        return control  # within deadband, keep current control

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


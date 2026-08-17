import random

from config import CONTROL_ACTIONS

_last_action = 0

def random_control(state: dict, misc: dict):
    global _last_action
    current_time = misc["current_time"]
    control = _last_action

    if current_time % 2 == 0:
        control = random.choice(list(CONTROL_ACTIONS.values()))
        _last_action = control

    return control
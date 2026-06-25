import random

from config import CONTROL_ACTIONS

def random_control(state: dict):
    current_time = state["current_time"]
    control = state["control"]

    if current_time % 2 == 0:
        control = random.choice(list(CONTROL_ACTIONS.values()))

    return control
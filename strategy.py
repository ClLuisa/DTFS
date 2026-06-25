import random

from config import CONTROL_TYPE
from control_strategies.hard_coded import hard_coded_control
from control_strategies.random import random_control
from control_strategies.deep_rl import deep_rl_control

def return_control(state: dict, control_type: str = CONTROL_TYPE):

    if control_type == "random":
        return random_control(state)

    elif control_type == "hard_coded":
        return hard_coded_control(state)

    elif control_type == "all_closed":
        return 0

    elif control_type == "all_open":
        return 2
    
    elif control_type == "inside_open":
        return 1
    
    elif control_type == "outside_open":
        return -1
    
    elif control_type == "deep_rl":
        return deep_rl_control(state)




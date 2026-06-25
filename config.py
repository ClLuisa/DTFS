STATE_STORAGE_PATH = r"C:\Users\lkclu\Documents\GitHub\DTFS\data"
CONTROL_TYPE = "deep_rl" # "hard_coded" # "random" "deep_rl" "all_closed" "all_open" "inside_open" "outside_open" "deep_rl_pytorch"
CONTROL_ACTIONS = {
    "all_closed": 0, # for winter night
    "all_open": 2, # for summer night
    "inside_open": 1, # for winter day (with radiation)
    "outside_open": -1 # for summer day
}
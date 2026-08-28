from datetime import datetime
import sys

sys.path.append(r"C:\Users\lkclu\Documents\GitHub\DTFS")

from control import OfflineTrainer

DATA_FILE_PATH = r"C:\Users\lkclu\Documents\GitHub\DTFS\results\control\simulation\stg_aug_sep\control_random.jsonl"

trainer = OfflineTrainer("stg_aug_sep_random")
trainer.train(DATA_FILE_PATH)
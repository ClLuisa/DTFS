import os, json
from config import STATE_STORAGE_PATH, CONTROL_TYPE

def reset_state_storage():
    file_path = os.path.join(STATE_STORAGE_PATH, f"{CONTROL_TYPE}.jsonl")

    with open(file_path, "w") as f:
        pass

def save_state_to_state_storage(data: dict):
    file_path = os.path.join(STATE_STORAGE_PATH, f"{CONTROL_TYPE}.jsonl")

    with open(file_path, "a") as f:
        f.write(json.dumps(data) + "\n")

def read_last_state_from_state_storage():
    file_path = os.path.join(STATE_STORAGE_PATH, f"{CONTROL_TYPE}.jsonl")

    try:
        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)

            if f.tell() == 0:
                return None

            pointer = f.tell() - 1

            while pointer >= 0:
                f.seek(pointer)
                if f.read(1) != b"\n":
                    break
                pointer -= 1

            while pointer >= 0:
                f.seek(pointer)
                if f.read(1) == b"\n":
                    pointer += 1
                    break
                pointer -= 1

            f.seek(pointer)
            last_line = f.readline().decode()

        return json.loads(last_line)

    except (FileNotFoundError, json.JSONDecodeError):
        return None
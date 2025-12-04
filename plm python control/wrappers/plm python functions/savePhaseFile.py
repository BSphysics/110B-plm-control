import json
import os
CONFIG_FILE = "beam_config.json"

def save_phase_file(key, file_path):
    """
    Save a phase correction file path to the config JSON.
    
    key: str, e.g., "beam_A_phase_file" or "beam_B_phase_file"
    file_path: str, full path to the file
    """
    cfg = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            print("Warning: could not read existing config file:", e)

    cfg[key] = file_path

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"Saved {key} -> {file_path}")
    except Exception as e:
        print("Error saving config file:", e)

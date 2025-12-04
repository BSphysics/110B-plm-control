import os
import json
import numpy as np

def load_last_phase_corrections(config_file, import_func):
    if not os.path.exists(config_file):
        print(f"No config file found at {config_file}, skipping phase load.")
        return

    try:
        with open(config_file, "r") as f:
            cfg = json.load(f)

        if "beam_A_phase_file" in cfg:
            path = cfg["beam_A_phase_file"]
            if os.path.exists(path):
                import_func("A", path)
                print("Loaded Beam A correction:", path)

        if "beam_B_phase_file" in cfg:
            path = cfg["beam_B_phase_file"]
            if os.path.exists(path):
                import_func("B", path)
                print("Loaded Beam B correction:", path)

    except Exception as e:
        print("Error loading persisted phase files:", e)

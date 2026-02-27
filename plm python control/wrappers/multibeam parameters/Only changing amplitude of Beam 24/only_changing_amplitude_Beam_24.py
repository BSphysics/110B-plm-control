# -*- coding: utf-8 -*-
"""
Generate independent amplitude sweeps for Beam 24
based on master file, in 10% steps.
"""

import pandas as pd
import os
import numpy as np

# =========================
# SETTINGS
# =========================

workingDir = os.getcwd()
master_filename = 'multiBeam_BeamA_100_BeamB_100.xlsx'
master_path = os.path.join(workingDir, master_filename)

if not os.path.exists(master_path):
    raise FileNotFoundError(f"Master file not found: {master_filename}")

print(f"Loading master file: {master_filename}")

# =========================
# LOAD MASTER FILE
# =========================

master_df = pd.read_excel(master_path, header=None)

if master_df.shape[1] < 4:
    raise ValueError("Excel file does not appear to have 4 columns.")

# =========================
# HELPER TO GET ROWS FOR A SPECIFIC BEAM
# =========================

def get_beam_mask(df, beam_label, beam_number):
    """
    Return mask for rows corresponding to a given beam and relative amplitude.
    Each beam has 1 row per amplitude in this layout.
    """
    col0_clean = df[0].astype(str).str.strip()
    col2_clean = df[2].astype(str).str.strip()
    mask_all = (col0_clean == beam_label) & (col2_clean == "Relative amplitude")
    
    all_indices = np.where(mask_all)[0]
    if len(all_indices) == 0:
        raise ValueError(f"No '{beam_label}' relative amplitude rows found.")
    
    # Beam numbering is 0-based: beam 24 → index 23
    start_idx = beam_number * 1
    row_idx = all_indices[start_idx:start_idx+1]
    return row_idx

# =========================
# SELECT BEAM 24
# =========================

beam_number = 23  # Beam 24 (zero-indexed)

mask_A_24 = get_beam_mask(master_df, "Beam A", beam_number)
mask_B_24 = get_beam_mask(master_df, "Beam B", beam_number)

original_A = master_df.loc[mask_A_24, 3].copy()
original_B = master_df.loc[mask_B_24, 3].copy()

# =========================
# SWEEP BEAM A (B stays 100%)
# =========================

for percent in range(100, -10, -10):
    scale = percent / 100.0
    percent_str = f"{percent:03d}"

    df = master_df.copy()
    df.loc[mask_A_24, 3] = original_A * scale

    new_filename = f"multiBeam_BeamA_{percent_str}_BeamB_100_beam24_only.xlsx"
    new_path = os.path.join(workingDir, new_filename)
    df.to_excel(new_path, header=False, index=False)

    print(f"Saved Beam A {percent_str}% version: {new_filename}")

# =========================
# SWEEP BEAM B (A stays 100%)
# =========================

for percent in range(100, -10, -10):
    scale = percent / 100.0
    percent_str = f"{percent:03d}"

    df = master_df.copy()
    df.loc[mask_B_24, 3] = original_B * scale

    new_filename = f"multiBeam_BeamA_100_BeamB_{percent_str}_beam24_only.xlsx"
    new_path = os.path.join(workingDir, new_filename)
    df.to_excel(new_path, header=False, index=False)

    print(f"Saved Beam B {percent_str}% version: {new_filename}")

print("\nBeam 24 independent sweeps complete.")
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
master_filename = 'multiBeam_9_spot_MASTER.xlsx'
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
# ZERO GLOBAL AMPLITUDE FOR ALL EXCEPT SELECTED BEAMS
# =========================

# Beams to KEEP (1-based numbering as in Excel)
beams_to_keep = [1, 4, 7, 22, 25, 28, 43, 46, 49]

# Clean columns for safety
col0_clean = master_df[0].astype(str).str.strip()
col2_clean = master_df[2].astype(str).str.strip()

# Mask for all Global amplitude rows
global_amp_mask = (
    (col0_clean == "Beam") &
    (col2_clean == "Global amplitude")
)

# Mask for beams we want to zero
zero_mask = global_amp_mask & (~master_df[1].isin(beams_to_keep))

# Set to zero
master_df.loc[zero_mask, 3] = 0

print("Global amplitude set to 0 for all beams except:", beams_to_keep)

output_filename = "multibeam_9_spot.xlsx"
output_path = os.path.join(workingDir, output_filename)

master_df.to_excel(output_path, header=False, index=False)

print(f"Saved modified file to: {output_filename}")

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
# SELECT BEAM 25
# =========================

beam_number = 24  # Beam 25 (zero-indexed)

mask_A_25 = get_beam_mask(master_df, "Beam A", beam_number)
mask_B_25 = get_beam_mask(master_df, "Beam B", beam_number)

original_A_25 = master_df.loc[mask_A_25, 3].copy()
original_B_25 = master_df.loc[mask_B_25, 3].copy()

# =========================
# SWEEP BEAM A FOR BEAM 25 (B stays 100%)
# =========================

for percent in range(100, -20, -20):   # ← changed step to -20
    scale = percent / 100.0
    percent_str = f"{percent:03d}"

    df = master_df.copy()
    df.loc[mask_A_25, 3] = original_A_25 * scale

    new_filename = f"multiBeam_9_spot_25_BeamA_{percent_str}_BeamB_100.xlsx"
    new_path = os.path.join(workingDir, new_filename)
    df.to_excel(new_path, header=False, index=False)

    print(f"Saved Beam 25 - Beam A {percent_str}% version: {new_filename}")

# =========================
# SWEEP BEAM B FOR BEAM 25 (A stays 100%)
# =========================

for percent in range(100, -20, -20):   # ← changed step to -20
    scale = percent / 100.0
    percent_str = f"{percent:03d}"

    df = master_df.copy()
    df.loc[mask_B_25, 3] = original_B_25 * scale

    new_filename = f"multiBeam_9_spot_25_BeamA_100_BeamB_{percent_str}.xlsx"
    new_path = os.path.join(workingDir, new_filename)
    df.to_excel(new_path, header=False, index=False)

    print(f"Saved Beam 25 - Beam B {percent_str}% version: {new_filename}")
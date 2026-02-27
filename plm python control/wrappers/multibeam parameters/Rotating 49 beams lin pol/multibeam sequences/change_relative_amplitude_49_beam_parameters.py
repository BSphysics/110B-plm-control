# -*- coding: utf-8 -*-
"""
Generate Beam A amplitude sweep from 100% down to 0%
in 10% steps, based on master file.
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
# LOAD MASTER FILE ONCE
# =========================

master_df = pd.read_excel(master_path, header=None)

if master_df.shape[1] < 4:
    raise ValueError("Excel file does not appear to have 4 columns.")

# Clean for matching only
col0_clean = master_df[0].astype(str).str.strip()
col2_clean = master_df[2].astype(str).str.strip()

mask = (
    (col0_clean == "Beam A") &
    (col2_clean == "Relative amplitude")
)

rows_found = mask.sum()

if rows_found == 0:
    raise ValueError("No 'Beam A' relative amplitude rows found.")

print(f"Found {rows_found} Beam A amplitude rows.")

print("\nOriginal amplitudes:")
print(master_df.loc[mask])

# Ensure numeric once
master_df.loc[mask, 3] = pd.to_numeric(master_df.loc[mask, 3], errors='coerce')

original_values = master_df.loc[mask, 3].copy()

# =========================
# GENERATE SWEEP
# =========================

for percent in range(100, -10, -10):  # 100 → 0 in steps of 10

    scale_factor = percent / 100.0
    percent_str = f"{percent:03d}"

    print(f"\nGenerating {percent_str}% version...")

    # Work on a fresh copy each time
    df = master_df.copy()

    df.loc[mask, 3] = original_values * scale_factor

    new_filename = f"multiBeam_BeamA_{percent_str}_BeamB_100.xlsx"
    new_path = os.path.join(workingDir, new_filename)

    df.to_excel(new_path, header=False, index=False)

    print(f"Saved: {new_filename}")

print("\nDone. Beam A sweep complete.")


# ============================================================
# BEAM B SWEEP (100% → 0%)
# ============================================================

# Clean for Beam B matching
col0_clean = master_df[0].astype(str).str.strip()
col2_clean = master_df[2].astype(str).str.strip()

mask_B = (
    (col0_clean == "Beam B") &
    (col2_clean == "Relative amplitude")
)

rows_found_B = mask_B.sum()

if rows_found_B == 0:
    raise ValueError("No 'Beam B' relative amplitude rows found.")

print(f"\nFound {rows_found_B} Beam B amplitude rows.")

# Ensure numeric once
master_df.loc[mask_B, 3] = pd.to_numeric(master_df.loc[mask_B, 3], errors='coerce')

original_values_B = master_df.loc[mask_B, 3].copy()

for percent in range(100, -10, -10):  # 100 → 0

    scale_factor = percent / 100.0
    percent_str = f"{percent:03d}"

    print(f"\nGenerating Beam B {percent_str}% version...")

    df = master_df.copy()

    df.loc[mask_B, 3] = original_values_B * scale_factor

    new_filename = f"multiBeam_BeamA_100_BeamB_{percent_str}.xlsx"
    new_path = os.path.join(workingDir, new_filename)

    df.to_excel(new_path, header=False, index=False)

    print(f"Saved: {new_filename}")

print("\nBeam B sweep complete.")
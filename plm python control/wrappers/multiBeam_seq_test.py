import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
import os
scriptDir = os.getcwd()
import sys
sys.path.append(os.path.join(scriptDir,"plm python functions" ))
sys.path.append(os.path.join(scriptDir,"basler python functions" ))
from generatePhaseTilt import generate_phase_tilt
from HGMode import HG_mode
import os
import time
#from numba import njit

# @njit
# def add_beams_test(acc, A_amp, A_phase, B_amp, B_phase, global_amp):
# 	rows, cols = A_amp.shape
# 	for i in range(rows):
# 		for j in range(cols):
# 			a = A_amp[i, j] * np.cos(A_phase[i, j]) + 1j * A_amp[i, j] * np.sin(A_phase[i, j])
# 			b = B_amp[i, j] * np.cos(B_phase[i, j]) + 1j * B_amp[i, j] * np.sin(B_phase[i, j])
# 			acc[i, j] += global_amp * (a + b)

diagonals = []

for d in range(13):  # there are 2n - 1 diagonals for n=7
    diagonal = []
    for r in range(7):
        c = d - r
        if 0 <= c < 7:
            label = 7 * c + (r + 1)
            diagonal.append(label)
    diagonals.append(diagonal)

# Print results
for i, diag in enumerate(diagonals):
    print(f"Diagonal {i}: {diag}")

file_path = os.path.join('D:\PLM\plm python control\wrappers\multiBeamData_All_ON.xlsx')
df = pd.read_excel(file_path) 
beamParameters = df.iloc[:, 3].values
beamParameterBlocks = []
	# Walk through the column in steps of 11 (10 data rows + 1 gap row)
for i in range(0, len(beamParameters), 11):  
    chunk = beamParameters[i:i+10]
    if len(chunk) == 10 and not np.any(pd.isna(chunk)):
        beamParameterBlocks.append(chunk)
beamParameterBlocks = np.array(beamParameterBlocks)


for d_index, diagonal in enumerate(diagonals):
    print(f"\nLoading diagonal {d_index}: Beams {diagonal}")
    
    for beam_label in diagonal:
        if 1 <= beam_label <= len(beamParameterBlocks):
            beam_params = beamParameterBlocks[beam_label - 1]  # beam 1 → index 0
            print(f"  Beam {beam_label}: {beam_params}")
        else:
            print(f"  Beam {beam_label}: No parameter block available!")
















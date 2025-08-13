import os
scriptDir = os.getcwd()
import sys
sys.path.append(os.path.join(scriptDir,"plm python functions" ))
sys.path.append(os.path.join(scriptDir,"basler python functions" ))
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
from generatePhaseTilt import generate_phase_tilt
from HGMode import HG_mode
#import os
import time
from numba import njit

# Set a default path (relative or absolute)
default_path = os.path.join('D:\PLM\plm python control\wrappers', "Data")

# # Open file dialog with default path
# file_path = filedialog.askopenfilename(
# 	title="Select the spreadsheet",
# 	filetypes=[("Excel files", "*.xlsx *.xls")],
# 	initialdir=default_path
# )

# if not file_path:
# 	print("No file selected.")

file_path = r'D:\PLM\plm python control\wrappers\Data\2025_06_24\multiBeam A and B 7x7\combined_025.xlsx'
df = pd.read_excel(file_path) 
beamParameters = df.iloc[:, 3].values #.dropna().tolist() 

beamParameterBlocks = []

# Walk through the column in steps of 11 (10 data rows + 1 gap row)
for i in range(0, len(beamParameters), 11):  
    chunk = beamParameters[i:i+10]

    if np.any(pd.isna(chunk)):
        print(chunk)
    if len(chunk) == 10 and not np.any(pd.isna(chunk)):
       beamParameterBlocks.append(chunk)

beamParameterBlocks = np.array(beamParameterBlocks)
print(f"Total blocks found: {len(beamParameterBlocks)}")

#print(chunk)
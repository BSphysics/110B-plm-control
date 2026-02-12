# multibeam optimiser 
import numpy as np
import pandas as pd



file_path = 'D:\PLM\plm python control\wrappers\multiBeamData_FLAT.xlsx'
df = pd.read_excel(file_path)
beamParameters = df.iloc[:, 3].values #.dropna().tolist() 

beamParameterBlocks = []

# Walk through the column in steps of 11 (10 data rows + 1 gap row)
for i in range(0, len(beamParameters), 11):  
    chunk = beamParameters[i:i+10]
    if len(chunk) == 10 and not np.any(pd.isna(chunk)):
        beamParameterBlocks.append(chunk)

beamParameterBlocks = np.array(beamParameterBlocks)

print(beamParameterBlocks[48])

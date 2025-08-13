import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
from generatePhaseTilt import generate_phase_tilt
from HGMode import HG_mode
import os
import time
from numba import njit

@njit
def add_beams_test(acc, A_amp, A_phase, B_amp, B_phase, global_amp):
	rows, cols = A_amp.shape
	for i in range(rows):
		for j in range(cols):
			a = A_amp[i, j] * np.cos(A_phase[i, j]) + 1j * A_amp[i, j] * np.sin(A_phase[i, j])
			b = B_amp[i, j] * np.cos(B_phase[i, j]) + 1j * B_amp[i, j] * np.sin(B_phase[i, j])
			acc[i, j] += global_amp * (a + b)


def load_multibeam_data(self):
	# Hide the root tkinter window
	root = Tk()
	root.withdraw()

	# Set a default path (relative or absolute)
	default_path = os.path.join('D:\PLM\plm python control\wrappers', "Data")

	# Open file dialog with default path
	file_path = filedialog.askopenfilename(
		title="Select the spreadsheet",
		filetypes=[("Excel files", "*.xlsx *.xls")],
		initialdir=default_path
	)

	if not file_path:
		print("No file selected.")
		return None  # Or raise an exception / return default values

	df = pd.read_excel(file_path) 
	beamParameters = df.iloc[:, 3].values #.dropna().tolist() 

	beamParameterBlocks = []

	# Walk through the column in steps of 11 (10 data rows + 1 gap row)
	for i in range(0, len(beamParameters), 11):  
		chunk = beamParameters[i:i+10]
		if len(chunk) == 10 and not np.any(pd.isna(chunk)):
			beamParameterBlocks.append(chunk)

	beamParameterBlocks = np.array(beamParameterBlocks)

	# Normalize the global amplitudes (column index 9)
	#global_amplitudes = beamParameterBlocks[:, 9]
	#beamParameterBlocks[:, 9] = global_amplitudes / np.max(global_amplitudes)
	
	# First pull out all the global amplitudes and normalise them to 1
	#all_global_amplitudes = beamParameters[9::10]
	#beamParameters[9::10] = all_global_amplitudes / np.max(all_global_amplitudes)

	cols, rows = 1358 , 800
	#finalCombined = np.zeros(cols,rows)

	combinedComplexs = []

	beamA_HG_phase, beamA_HG_amplitude = HG_mode(cols, rows, self.user_values[4], self.user_values[5], self.user_values[8], self.user_values[9], self.user_values[12], self.user_values[13])
	beamB_HG_phase, beamB_HG_amplitude = HG_mode(cols, rows, self.user_values[6], self.user_values[7], self.user_values[10], self.user_values[11], self.user_values[14], self.user_values[15])
	if self.clear_beam_A_correction_flag == True:
		self.beam_A_correction_data = np.zeros_like(beamA_phase_tilt) # clear the correction

	if self.clear_beam_B_correction_flag == True:
		self.beam_B_correction_data = np.zeros_like(beamB_phase_tilt)

	#print(beamParameters)
	finalCombined = np.zeros((rows, cols), dtype=np.complex128)
	for i, chunk in enumerate(beamParameterBlocks):
		print('Loading parameters for Beam: ' + str(int(i)))
		#start_time = time.time()

		# Generate phase tilts
		
		#chunk = beamParameters[i:i + chunk_size]
		#print(chunk[2], chunk[3])
		beamA_phase_tilt = generate_phase_tilt(rows, cols, chunk[0], chunk[1], self.button_states[0], self.button_states[1]) 
		beamB_phase_tilt = generate_phase_tilt(rows, cols, chunk[2], chunk[3], self.button_states[2], self.button_states[3])
		#print(f"  Generate phase tilt time: {time.time() - t0:.3f} s")
		#beamA_HG_phase, beamA_HG_amplitude = HG_mode(cols, rows, self.user_values[4], self.user_values[5], self.user_values[8], self.user_values[9], self.user_values[12], self.user_values[13])
		#beamB_HG_phase, beamB_HG_amplitude = HG_mode(cols, rows, self.user_values[6], self.user_values[7], self.user_values[10], self.user_values[11], self.user_values[14], self.user_values[15])

		relative_amplitudes = np.array([chunk[7],chunk[8]])

		max_val = np.max(relative_amplitudes)
		if max_val > 0:
			relative_amplitudes = relative_amplitudes / max_val

		beamA_amplitude = beamA_HG_amplitude * relative_amplitudes[0]
		beamB_amplitude = beamB_HG_amplitude * relative_amplitudes[1]
		global_amplitude = chunk[9]
		if abs(global_amplitude) < 1e-6:
			continue
		#print(global_amplitude)
		
		relative_phase_A = (chunk[4]/100)*2*np.pi  
		relative_phase_B = (chunk[5]/100)*2*np.pi
		#print('Relative Phase Beam A = ' + str(np.round(relative_phase_A,2)))

		global_phase = (chunk[6]/100)*2*np.pi

		beamA_phase = beamA_phase_tilt - self.beam_A_correction_data + beamA_HG_phase + relative_phase_A + global_phase
		beamB_phase = beamB_phase_tilt - self.beam_B_correction_data + beamB_HG_phase + relative_phase_B + global_phase

		add_beams_test(finalCombined, beamA_amplitude, beamA_phase, beamB_amplitude, beamB_phase, global_amplitude)
		finalCombined[np.abs(finalCombined) < 1e-6] = 0

	return finalCombined
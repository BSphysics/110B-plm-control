
# Aim of the function is to generate 24 plm phase maps. Each phase map is selected to give a different multibeam pattern on the 
# camera. This means that ultimately, this function should read in 24 different .xlsx files. To begin with it will only import a few 
# of these but having proven the principle, it should scale up to 24 quite easily. The phase maps need to be acquired with HW timed
# acquistion by the camera. 

import numpy as np
from loadMultibeamData import load_multibeam_data
from ampModPhase import amp_mod_phase

def multi_beam_seq(self):

	multibeam_seq_frame = np.zeros((800, 1358, 24), dtype=np.float32)

	for idx in range(6):
		combinedComplex, multibeam_file_path = load_multibeam_data(self)
		#print('Multibeam file path = ' + str(multibeam_file_path))
		amplitude_modulated_combined_phase = amp_mod_phase(combinedComplex)

		plm_phase_map = (amplitude_modulated_combined_phase + np.pi) / (2*np.pi)
		multibeam_seq_frame[:,:,idx] = plm_phase_map

	multibeam_seq_frame = np.transpose(multibeam_seq_frame, (1, 0, 2)).copy(order='F')

	return  multibeam_seq_frame

	
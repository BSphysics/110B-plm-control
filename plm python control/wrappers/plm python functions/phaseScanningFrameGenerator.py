import numpy as np
from ampModPhase import amp_mod_phase

# Aim of the function is to generate 24 plm phase maps, each with a different value of global phase. 
# These 24 phase maps should then be wrapped into a single plm frame

def phase_scanning_frame_generator (beamA_phase_tilt ,beam_A_correction_data , beamA_HG_phase , beamB_phase, beamA_amplitude, beamB_amplitude):

	phase_values = np.arange(0,100.1,4.33)
	n_frames = len(phase_values)

	H, W = beamA_phase_tilt.shape
	plm_phase_scan_frame = np.zeros((H, W, n_frames), dtype=np.float32)

	for idx, i in enumerate(phase_values):

		global_phase_A = (i/100)*2*np.pi  

		beamA_phase = beamA_phase_tilt - beam_A_correction_data + beamA_HG_phase + global_phase_A

		beamAcomplex = beamA_amplitude * np.exp(1j * beamA_phase)
		beamBcomplex = beamB_amplitude * np.exp(1j * beamB_phase)

		combinedComplex = beamAcomplex + beamBcomplex
        
		amplitude_modulated_combined_phase = amp_mod_phase(combinedComplex) 

		plm_phase_map = (amplitude_modulated_combined_phase + np.pi) / (2*np.pi)

		plm_phase_scan_frame[:,:,idx] = plm_phase_map

	return  plm_phase_scan_frame

# Aim of the function is to generate 24 plm phase maps. Each phase map will use a different combinatiom of 
# global amplitudes defined in amp_ramp. After one amp_ramp sequence, the global phase is flipped by pi rads and then
# a reverse of the amp_ramp sequence is run. If Beam A and Beam B and configured as HG 10 and 01 modes,
# the global phase is configured properly first and a linear polariser is in place in front of the camera, 
# these phase maps will each produce two lobes that have a differnt angle in the camera plane for each phase map
# These 24 phase maps should then be wrapped into a single plm frame - resulting in 24 rotating lobes that refresh at 720 Hz

import numpy as np
from ampModPhase import amp_mod_phase

amp_ramp = []
# BeamB ramps up from 0 to 1.0 while BeamA stays at 1.0
for b in np.arange(0.0, 1.01, 0.2):
    amp_ramp.append([1.0, b])

# BeamA ramps down from 0.9 to 0.0 while BeamB stays at 1.0
for a in np.arange(0.8, -0.01, -0.2):
    amp_ramp.append([a, 1.0])

amp_ramp_array = np.array(amp_ramp)
amp_ramp = np.vstack((amp_ramp_array, np.flipud(amp_ramp_array)))
amp_ramp = np.vstack((amp_ramp, amp_ramp_array[0], amp_ramp_array[0]))

def amp_ramp_frame_generator (beamA_phase, beamB_phase, beamA_HG_amplitude, beamB_HG_amplitude ):

	H, W = beamA_phase.shape
	amp_ramp_scan_frame = np.zeros((H, W, 24), dtype=np.float32)

	for idx in range(len(amp_ramp)):

		global_amplitudes = np.array([amp_ramp[idx,0] , amp_ramp[idx,1]])
		global_amplitudes = global_amplitudes/ np.max(global_amplitudes)
		#print(global_amplitudes)
		
		beamA_amplitude = beamA_HG_amplitude * global_amplitudes[0]
		beamB_amplitude = beamB_HG_amplitude * global_amplitudes[1]
		#print(beamA_amplitude)
		#print(beamB_amplitude)
		if idx < 11:
			beamAcomplex = beamA_amplitude * np.exp(1j * beamA_phase)
			beamBcomplex = beamB_amplitude * np.exp(1j * beamB_phase)
		else: 
			beamAcomplex = beamA_amplitude * np.exp(1j * (beamA_phase + np.pi))
			beamBcomplex = beamB_amplitude * np.exp(1j * beamB_phase)

		combinedComplex = beamAcomplex + beamBcomplex
	    
		amplitude_modulated_combined_phase = amp_mod_phase(combinedComplex) 
	    
		plm_phase_map = (amplitude_modulated_combined_phase + np.pi) / (2*np.pi)

		amp_ramp_scan_frame[:,:,idx] = plm_phase_map

	amp_ramp_scan_frame = np.transpose(amp_ramp_scan_frame, (1, 0, 2)).copy(order='F')
	return  amp_ramp_scan_frame
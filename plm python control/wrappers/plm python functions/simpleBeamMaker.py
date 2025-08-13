import numpy as np
from generatePhaseTilt import generate_phase_tilt
from HGMode import HG_mode
from ampModPhase import amp_mod_phase
import os
# from datetime import datetime
import time
# from basler_centroid import baslerCentroid
# import cv2
# from scipy.interpolate import LinearNDInterpolator
# from openpyxl import Workbook, load_workbook
# from openpyxl.utils import get_column_letter
# import sys
# from PyQt5.QtWidgets import QApplication, QInputDialog
# import matplotlib.pyplot as plt

numHolograms = 24
cols, rows = 1358 , 800


def simple_beam_maker(self, BeamA_amp, BeamB_amp, BeamB_xtilt_shift, BeamB_ytilt_shift, global_phase_A):

    beamA_phase_tilt = generate_phase_tilt(rows, cols, self.user_values[0], self.user_values[1], self.button_states[0], self.button_states[1]) 
    beamB_phase_tilt = generate_phase_tilt(rows, cols, self.user_values[2]+ BeamB_xtilt_shift, self.user_values[3]+BeamB_ytilt_shift, self.button_states[2], self.button_states[3])
    
    beamA_HG_phase, beamA_HG_amplitude = HG_mode(cols, rows, self.user_values[4], self.user_values[5], self.user_values[8], self.user_values[9], self.user_values[12], self.user_values[13])
    beamB_HG_phase, beamB_HG_amplitude = HG_mode(cols, rows, self.user_values[6], self.user_values[7], self.user_values[10], self.user_values[11], self.user_values[14], self.user_values[15])
    
    #global_amplitudes = np.array([self.user_values[16] , self.user_values[17]])
    #global_amplitudes = global_amplitudes/ np.max(global_amplitudes)

    beamA_amplitude = beamA_HG_amplitude * BeamA_amp
    beamB_amplitude = beamB_HG_amplitude * BeamB_amp
            
    #global_phase_A = (self.user_values[18]/100)*2*np.pi  
    global_phase_A = (global_phase_A/100)*2*np.pi
    global_phase_B = (self.user_values[19]/100)*2*np.pi 

    if self.clear_beam_A_correction_flag == True:
        self.beam_A_correction_data = np.zeros_like(beamA_phase_tilt)

    if self.clear_beam_B_correction_flag == True:
        self.beam_B_correction_data = np.zeros_like(beamB_phase_tilt)


    beamA_phase = beamA_phase_tilt - self.beam_A_correction_data + beamA_HG_phase + global_phase_A
    beamB_phase = beamB_phase_tilt - self.beam_B_correction_data + beamB_HG_phase + global_phase_B

    beamAcomplex = beamA_amplitude * np.exp(1j * beamA_phase)
    beamBcomplex = beamB_amplitude * np.exp(1j * beamB_phase)

    combinedComplex = beamAcomplex + beamBcomplex 
    
    amplitude_modulated_combined_phase = amp_mod_phase(combinedComplex)
    
    plm_phase_map = (amplitude_modulated_combined_phase + np.pi) / (2*np.pi)

    return plm_phase_map
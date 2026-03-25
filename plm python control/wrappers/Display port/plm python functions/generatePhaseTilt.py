# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 11:06:32 2025

@author: bs426
"""
import numpy as np

def generate_phase_tilt(rows=1358, cols=800, x_grating_pix=10, y_grating_pix=10, x_tilt_off=False, y_tilt_off=False):
    
    if x_tilt_off == True:
        x = np.linspace(0, 0, cols)
    else:
        x = np.linspace(0, 2 * np.pi * (cols/x_grating_pix), cols)  # X-direction phase
    
    if y_tilt_off == True:
        y = np.linspace(0, 0, rows)
    else:
        y = np.linspace(0, 2 * np.pi * (rows/y_grating_pix), rows)  # Y-direction phase

    X, Y = np.meshgrid(x, y)
    phase_tilt = (X + Y) # Not phase wrapped

    return phase_tilt


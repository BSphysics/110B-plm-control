# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:09:50 2025

@author: bs426
"""
import numpy as np
import math
# import matplotlib.pyplot as plt
# from scipy.special import hermite
from scipy.interpolate import interp1d

def amp_mod_phase(combinedComplex):

    combinedComplex_amplitude = np.abs(combinedComplex)
    combinedComplex_amplitude = combinedComplex_amplitude / np.max(combinedComplex_amplitude)
    
    combinedComplex_phase = np.angle(combinedComplex)
    
    #----------------------------Amplitude modulated phase
    
    x_grid = np.linspace(0, 1, 1000)  # Generate 10,000 points from 0 to 1
    sinc_values = np.sinc(x_grid)      # Compute sinc(x) = sin(pi*x)/(pi*x) and reverse order
    
    # interpolator for inverse sinc
    arcsinc_interp = interp1d(np.flip(sinc_values), np.flip(x_grid), kind='linear', fill_value="extrapolate")
    
    def arcsinc_fast(y):
        """Fast inverse sinc function using interpolation."""
        y = np.asarray(y)
        y = np.clip(y, sinc_values.min(), sinc_values.max())  # Keep within valid range
        return arcsinc_interp(y)
    
    arcSincArray = arcsinc_fast(combinedComplex_amplitude)
    
    combinedAmplitudeModulatedPhase = combinedComplex_phase * (1-arcSincArray)
    
    return combinedAmplitudeModulatedPhase
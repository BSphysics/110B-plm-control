# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 11:52:09 2025

@author: bs426
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.interpolate import interp1d

def HG_mode(cols, rows, m, n, xWaist, yWaist, xshift, yshift):

    def hermite_gaussian(x, n, w0, shift):
        """
        Compute the Hermite-Gaussian mode along one dimension.
        :param x: Array of x values.
        :param n: Hermite mode index.
        :param w0: Beam waist parameter.
        :return: Hermite-Gaussian function values.
        """
        Hn = hermite(int(n))
        # normalization = (2**n * math.factorial(n) * np.sqrt(np.pi) * w0) ** -0.5
        gaussian_envelope = np.exp(-(x-shift)**2 / w0**2)
        return 1 * Hn((x-shift) / w0) * gaussian_envelope
    
    def generate_hermite_gaussian_2d(cols, rows, m, n, xWaist=100, yWaist=100, xshift=0, yshift=0):
        """
        Generate a 2D Hermite-Gaussian mode array with a specified shift.
        :param m: Mode index along x.
        :param n: Mode index along y.
        :param w0: Beam waist parameter.
        :param shift_x: Shift in elements along x-axis.
        :param shift_y: Shift in elements along y-axis.
        :return: 2D array representing the Hermite-Gaussian mode.
        """
        center_x=int(cols/2)
        center_y=int(rows/2)
        
        x = np.linspace(0, cols, cols)
        y = np.linspace(0, rows, rows)
        X, Y = np.meshgrid(x, y)
        
        HG_x = hermite_gaussian(X, m, xWaist, center_x - xshift)
        HG_y = hermite_gaussian(Y, n, yWaist, center_y - yshift)
        
        HG_mode = HG_x * HG_y
    
        return HG_mode
    
    
    
    beam = generate_hermite_gaussian_2d(cols, rows, m, n, xWaist, yWaist, xshift, yshift)
    
    normalisation = np.sqrt(np.dot(np.conj(beam).flatten(), beam.flatten()))
    
    beam = beam/normalisation.astype(complex)
    
    HG_amplitude = np.abs(beam)
    HG_phase = np.angle(beam)

    return HG_phase, HG_amplitude
    
    
    

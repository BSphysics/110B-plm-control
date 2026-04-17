# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 14:22:36 2025

@author: bs426
"""
import numpy as np
from generatePhaseTilt import generate_phase_tilt

def super_pixel_set(sPix_W,sPix_H,phase_main,n_steps):
    """
    Parameters
    ----------
    sPix_W     : super-pixel width in pixels\n
    sPix_H     : super-pixel height in pixels\n
    phase_main : a 2d phase array which will be cut-up into superpixels\n
                 note that this function treats phase as being in range [0 1]
    n_steps    : no. of steps for phase stepping\n
    
    Returns
    -------
    sPix_modes : 3d array containing super-pixel holograms.\n
                 It has the same no. of rows and columns as phase_main,\n
                 and (n_steps+1)*(no. of super-pixels) pages.\n 
                 Each super-pixel has (n_steps+1) holograms associated with it:\n
                 the first one shows only the test super-pixel, the rest include\n
                 the reference super-pixel and phase-stepped test super-pixel
    nx_sPix    : no. of super-pixels along the width of the hologram
    ny_sPix    : no. of super-pixels along the height of the hologram
    """
    
    # height and width of the main phase array
    H, W = np.shape(phase_main)
    
    # no. of superpixels (the -ve signs ensure rounding-up of integers)
    nx_sPix = -(-W//sPix_W)
    ny_sPix = -(-H//sPix_H)
    n_sPix = nx_sPix * ny_sPix

    # no. of phase offsets for phase stepping the test super-pixel
    phase_step = 1/(n_steps)

    # total number of modes which includes phase-stepping and amplitude measurement
    n_modes = n_sPix*(n_steps+1)
     
    #indices of the reference super-pixel (place it in the middle)
    i_ref = ny_sPix//2
    j_ref = nx_sPix//2
    row_idx_ref = i_ref*sPix_H
    col_idx_ref = j_ref*sPix_W
    # phase of the reference super-pixel
    ref_sPix_phase = phase_main[row_idx_ref:row_idx_ref+sPix_H , col_idx_ref:col_idx_ref+sPix_W]

    sPix_modes = np.zeros((H,W,n_modes), dtype=np.float32, order='F')
    page_counter = 0
    for i in range(ny_sPix):
        for j in range(nx_sPix):
            # indices of the i,j-th test super-pixel
            row_idx = i*sPix_H
            col_idx = j*sPix_W
            page_idx = page_counter*(n_steps+1)
            
            # phase of the i,j-th test super-pixel        
            test_sPix_phase = phase_main[row_idx:row_idx+sPix_H , col_idx:col_idx+sPix_W]
            # write this phase to the 3d array with phase of all super-pixels
            sPix_modes[row_idx:row_idx+sPix_H, col_idx:col_idx+sPix_W, page_idx] = test_sPix_phase
            
            # insert the phase of the reference super-pixel
            for k in range(n_steps):
                sPix_modes[row_idx:row_idx+sPix_H, col_idx:col_idx+sPix_W, page_idx+k+1] = np.mod(test_sPix_phase + k*phase_step, 1)
                sPix_modes[row_idx_ref:row_idx_ref+sPix_H, col_idx_ref:col_idx_ref+sPix_W, page_idx+k+1] = ref_sPix_phase 
                
            page_counter += 1
    

    
    # generate_phase_tilt(rows, cols, self.user_values[0], self.user_values[1], self.button_states[0], self.button_states[1]) 

    phase_main_expanded = np.expand_dims(phase_main*0.08, axis=-1)
    phase_zeros = np.zeros((phase_main.shape[0] ,phase_main.shape[1], 1))
    
    alternating_slices = np.concatenate(
    [phase_main_expanded if i % 2 == 0 else phase_zeros for i in range(24)], 
    axis=2)
    
    sPix_modes = np.concatenate((alternating_slices , sPix_modes) , axis = 2)
    
    # sPix_modes = np.concatenate((sPix_modes, alternating_slices) , axis = 2)

    sPix_modes = sPix_modes.astype('float32')

    return sPix_modes, nx_sPix, ny_sPix

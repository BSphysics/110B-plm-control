# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 16:53:11 2025

@author: bs426
"""
import numpy as np
# Divides all the super pixels phase maps into frames (with 24 maps in each)
def super_pixel_frames(sPix_phase):
    
    # Define the number of slices per frame
    slices_per_frame = 24

    # Compute the number of frames needed (including padding)
    num_frames = -(-sPix_phase.shape[2] // slices_per_frame)  
    new_depth = num_frames * slices_per_frame  # Total slices after padding

    # Create a zero-initialized array with the new shape
    padded_data = np.zeros((sPix_phase.shape[0], sPix_phase.shape[1], new_depth), dtype=sPix_phase.dtype)

    # Copy original data into the padded array
    padded_data[:, :, :sPix_phase.shape[2]] = sPix_phase

    # Preallocate frames array
    frames = np.zeros((num_frames, sPix_phase.shape[0], sPix_phase.shape[1], slices_per_frame), dtype=sPix_phase.dtype)

    # Efficient slicing to extract frames
    for i in range(num_frames):
        # print(i)
        frames[i] = padded_data[:, :, i * slices_per_frame : (i + 1) * slices_per_frame] 
        
        # plm_frame = np.asfortranarray(np.transpose(plm_frame, (1, 0, 2)))
        
    return frames
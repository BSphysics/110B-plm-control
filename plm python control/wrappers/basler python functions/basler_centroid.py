# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 17:11:14 2025

@author: bs426
"""

# from camera_config import camConfig
# from pypylon import pylon
# from pypylon import genicam
# import time
import numpy as np
from scipy.ndimage import gaussian_filter
# import os
# import matplotlib.pyplot as plt
# import sys

def baslerCentroid(img, sigma, thresh):
    
    img = gaussian_filter(img, sigma)
    
    mask = img > thresh
    
    rows, cols = np.indices(img.shape)
    
    if np.any(mask):  # Check if any high-intensity pixels exist
    
        centroid_x = int(np.sum(cols * mask * img) / np.sum(img[mask]))
        centroid_y = int(np.sum(rows * mask * img) / np.sum(img[mask]))
        
    else:
        centroid_x = int(48)
        centroid_y = int(48)
        print('These are just placeholder values - beam intensity too low to find proper centroid')
        # return None
       
    return centroid_x, centroid_y
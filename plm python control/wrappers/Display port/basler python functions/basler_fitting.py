# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 16:58:21 2025

@author: bs426
"""
import numpy as np
import os
scriptDir = os.getcwd()
import sys
sys.path.append(os.path.join(scriptDir,"!functions" ))
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.optimize as opt
import scipy.ndimage as ndi

def baslerFitting(zoomed_img):
    
    fig, (ax2,ax3,ax4) = plt.subplots(1,3, figsize=(16, 6))
    
    def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, offset):
        x, y = xy
        return amp * np.exp(-(((x-x0)**2)/(2*sigma_x**2) + ((y-y0)**2)/(2*sigma_y**2))) + offset

    x = np.arange(len(zoomed_img))
    y = np.arange(len(zoomed_img))
    X, Y = np.meshgrid(x, y)

    # Find the center of mass
    com = ndi.center_of_mass(zoomed_img)

    # Fit the Gaussian
    p0 = [np.max(zoomed_img), com[1], com[0], 10, 10, np.min(zoomed_img)]  # Initial guess
    params, _ = opt.curve_fit(gaussian_2d, (X.ravel(), Y.ravel()), zoomed_img.ravel(), p0)

    # Extract width and height
    fwhm_x = 2.355 * params[3]
    fwhm_y = 2.355 * params[4]
    print(f"\nBlob Width (FWHM_x): {fwhm_x:.2f} px, Height (FWHM_y): {fwhm_y:.2f} px")

    ax2.imshow(zoomed_img, cmap='gray', vmin=0, vmax=255)
    ax2.set_axis_off()
    ax2.set_title('2D Gaussian fitting centroid')
    ax2.scatter(params[1],params[2], color='green', s=5, label="Centroid")

    #%%
    def gaussian_1D(x, a, b, c, d):
        return a * np.exp(-(x-b)**2/(2*c**2)) + d

    row_fitting_data = zoomed_img[int(params[1]),:]
    p0 = [np.max(row_fitting_data), np.argmax(row_fitting_data), 3 , 1]  # Initial guess for fitting  
    
    try:
        rowParams1D, pcov = opt.curve_fit(gaussian_1D,x,row_fitting_data, p0=p0)   
        fit_successful = True
    except (RuntimeError, ValueError) as e:
        print(f"Curve fitting failed: {e}")
        fit_successful = False  # Prevents plotting
    
    if fit_successful:
        ax3.plot(row_fitting_data, 'yo')
        ax3.plot(x, gaussian_1D(x, *rowParams1D), 'k--')
        ax3.set_title('x FWHM = ' + str(np.round(rowParams1D[2]* 2.355,2)) + ' pix')
        rowParams1D = None

    fit_successful= False
    col_fitting_data = zoomed_img[:,int(params[2])]
    p0 = [np.max(col_fitting_data), np.argmax(col_fitting_data), 3 , 1]  # Initial guess for fitting 
    
    try: 
        colParams1D, pcov = opt.curve_fit(gaussian_1D,x,col_fitting_data, p0=p0)
        fit_successful = True
    except (RuntimeError, ValueError) as e:
        print(f"Curve fitting failed: {e}")
        fit_successful = False  # Prevents plotting
        colParams1D = None
    
    if fit_successful:    
        ax4.plot(col_fitting_data , 'go')
        ax4.plot(x, gaussian_1D(x, *colParams1D), 'k--')
        ax4.set_title('y FWHM = ' + str(np.round(colParams1D[2]* 2.355,2)) + ' pix')
    
    plt.show()
    plt.pause(0.1)
    return rowParams1D, colParams1D
    
    



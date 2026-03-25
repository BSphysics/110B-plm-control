# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 15:08:57 2026

@author: bs426
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter

plt.close('all')

def find_49_centroids(folder_name):
    images_path = os.path.join(folder_name , "grab50_mean.png")
    N_SPOTS = 49
    ROI_HALF_SIZE = 5   # half-width of each ROI (tweak as needed)
    NROWS, NCOLS = 7, 7  # grid shape
    
    # -----------------------------
    # LOAD IMAGE
    # -----------------------------
    img = cv2.imread(images_path, cv2.IMREAD_GRAYSCALE)
    img = img[150:250,300:400].astype(np.float32)
    
    # Optional smoothing (helps stability)
    img_smooth = gaussian_filter(img, sigma=1)
    
    # -----------------------------
    # FIND LOCAL MAXIMA
    # -----------------------------
    coordinates = peak_local_max(
        img_smooth,
        min_distance=5,     # adjust roughly to spot spacing
        num_peaks=N_SPOTS   # ensures only 49 returned
    )
    
    if len(coordinates) != N_SPOTS:
        print(f"Warning: detected {len(coordinates)} peaks")
    
    # coordinates are (row, col)
    centroids = coordinates
    
    # -----------------------------
    # SORT CENTROIDS INTO 7×7 GRID
    # -----------------------------
    sorted_centroids = []
    
    # Sort by x (column)
    by_x = centroids[centroids[:, 1].argsort()]
    
    # Split into NCOLS column groups
    columns = np.array_split(by_x, NCOLS)
    
    for col in columns:
        # Sort within column by y (top → bottom)
        col_sorted = col[col[:, 0].argsort()]
        sorted_centroids.extend(col_sorted)
    
    sorted_centroids = np.array(sorted_centroids)
    
    # -----------------------------
    # DEFINE ROIs AROUND EACH SPOT
    # -----------------------------
    rois = []
    
    h, w = img.shape
    
    for (y, x) in sorted_centroids:
        x, y = int(x), int(y)
        x1 = max(0, x - ROI_HALF_SIZE)
        x2 = min(w - 1, x + ROI_HALF_SIZE)
        y1 = max(0, y - ROI_HALF_SIZE)
        y2 = min(h - 1, y + ROI_HALF_SIZE)
        rois.append((x1, y1, x2, y2))
        
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for (y, x) in sorted_centroids:
        x = int(round(x))
        y = int(round(y))
        if 0 <= y < img_color.shape[0] and 0 <= x < img_color.shape[1]:
            img_color[y, x] = (255, 0, 0)   # BGR → Red
            
    
    highlight_spot = 49  # 1-based indexing (spot number in your spreadsheet)
    
    roi_sums = []
    
    for i, ((y, x), roi) in enumerate(zip(sorted_centroids, rois), start=1):
        x1, y1, x2, y2 = roi
        
        # Decide color: red for the selected spot, green otherwise
        color = (0, 0, 255) if i == highlight_spot else (0, 255, 0)
        
        # Draw rectangle
        cv2.rectangle(img_color, (x1, y1), (x2, y2), color=color, thickness=1)
        
        # Sum all pixel values inside ROI
        roi_sum = np.sum(img[y1:y2+1, x1:x2+1])
        roi_sums.append(roi_sum)
    
    # Display results
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(img_color.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title("ROIs overlaid")
    plt.axis('off')
    save_path = os.path.join(folder_name , "spot ROIs.png")
    plt.savefig(save_path)
    plt.close(fig)

    def contrast_norm(roi_sums):
        """Compute normalized contrast from a list of ROI sums."""
        if not roi_sums:  # handle empty list
            return 0.0
        max_val = max(roi_sums)
        min_val = min(roi_sums)
        mean_val = sum(roi_sums) / len(roi_sums)
        if mean_val == 0:
            return 0.0
        return (max_val - min_val) / mean_val

    cn = contrast_norm(roi_sums)
    print(f"Brightest - dimmest spot (normalised to the mean) = = {cn:.3f}")
    
    
    return sorted_centroids , roi_sums
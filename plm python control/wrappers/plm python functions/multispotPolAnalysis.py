# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:31:12 2026

@author: bs426
"""

# -*- coding: utf-8 -*-
"""
Automatically process all subfolders:
- Find the single .npy file in each
- Load and analyse it
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import tkinter as tk
from tkinter import filedialog
from scipy.optimize import curve_fit
from matplotlib import patches

def multispot_pol_analysis():

    # ============================================================
    # SETTINGS
    # ============================================================

    ROI_HALF_SIZE = 6
    POLARISER_ANGLES = np.arange(0, 200, 20)
    DEGREES = np.pi / 180
    PEAK_THRESHOLD_FRACTION = 0.15

    # ============================================================
    # SELECT PARENT DIRECTORY
    # ============================================================

    root = tk.Tk()
    root.withdraw()

    start_dir = getattr(multispot_pol_analysis, "last_dir", 
                    r"C:\Users\bs426\OneDrive - University of Exeter\!Work\Work.2026\Lab.2026\110B\multispot patterns")

    parent_folder = filedialog.askdirectory(title="Select parent folder",
                                             initialdir=start_dir)

    multispot_pol_analysis.last_dir = parent_folder

    root.destroy()

    if not parent_folder:
        raise ValueError("No folder selected.")

    print(f"\nProcessing parent folder:\n{parent_folder}")

    # ============================================================
    # FIND ALL SUBFOLDERS
    # ============================================================

    subfolders = [
        os.path.join(parent_folder, name)
        for name in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, name))
    ]

    if not subfolders:
        raise ValueError("No subfolders found.")

    print(f"Found {len(subfolders)} subfolders.\n")

    # ============================================================
    # PROCESS EACH SUBFOLDER
    # ============================================================

    for folder in subfolders:

        print(f"--- Processing folder: {os.path.basename(folder)} ---")

        # Find .npy files in this folder
        npy_files = [f for f in os.listdir(folder) if f.endswith(".npy")]

        if len(npy_files) == 0:
            print("No .npy file found. Skipping.\n")
            continue

        if len(npy_files) > 1:
            print("More than one .npy file found. Skipping.\n")
            continue

        file_path = os.path.join(folder, npy_files[0])
        print(f"Loading: {npy_files[0]}")

        # ========================================================
        # LOAD DATA
        # ========================================================

        imgs = np.load(file_path)

        # Crop region
        imgs_cropped = imgs[:, 100:250, 220:370]

        # Brightest frame
        brightest_index = np.argmax(imgs_cropped.sum(axis=(1, 2)))
        img_ref = np.sum(imgs_cropped , axis=0).astype(np.float32)

        img_smooth = gaussian_filter(img_ref, sigma=1)

        # ========================================================
        # FIND BRIGHT SPOTS
        # ========================================================

        coordinates = peak_local_max(
            img_smooth,
            min_distance=8,
            threshold_abs=np.max(img_smooth) * PEAK_THRESHOLD_FRACTION
        )

        print(f"Detected {len(coordinates)} bright spots\n")

        # ============================================================
        # BUILD ROIs
        # ============================================================
        
        rois = []
        h, w = img_ref.shape
        
        for (y, x) in coordinates:
            x, y = int(x), int(y)
            x1 = max(0, x - ROI_HALF_SIZE)
            x2 = min(w - 1, x + ROI_HALF_SIZE)
            y1 = max(0, y - ROI_HALF_SIZE)
            y2 = min(h - 1, y + ROI_HALF_SIZE)
            rois.append((x1, y1, x2, y2))
        
        # ============================================================
        # EXTRACT POWERS
        # ============================================================
        
        background_vals = np.sum(imgs[:, 200:211, 400:411], axis=(1, 2))
        
        powers_all = []
        
        for (x1, y1, x2, y2) in rois:
            roi_stack = imgs_cropped[:, y1:y2+1, x1:x2+1]
            roi_sums = roi_stack.sum(axis=(1, 2))
            powers = np.clip(roi_sums - background_vals, 0, None)
            powers_all.append(powers)
        
        powers_all = np.array(powers_all)
        
        # ============================================================
        # FIT MODEL (FULLY POLARISED STOKES RECONSTRUCTION)
        # ============================================================
        
        fit_results = []
        
        theta = POLARISER_ANGLES * DEGREES
        
        # Design matrix
        M = np.column_stack([
            np.ones_like(theta),
            np.cos(2*theta),
            np.sin(2*theta)
        ])
        
        for powers in powers_all:
        
            try:
                # Linear least squares
                coeffs, _, _, _ = np.linalg.lstsq(M, powers, rcond=None)
                A, B, C = coeffs
        
                # Stokes parameters
                S0 = 2*A
                S1 = 2*B
                S2 = 2*C
        
                # Orientation
                alpha = 0.5 * np.arctan2(S2, S1)
        
                # Reconstruct |S3| (fully polarised assumption)
                S3_abs_sq = S0**2 - S1**2 - S2**2
                S3_abs_sq = max(S3_abs_sq, 0)  # avoid small negative from noise
                S3_abs = np.sqrt(S3_abs_sq)
        
                # Ellipticity angle magnitude
                chi = 0.5 * np.arcsin(S3_abs / S0) if S0 != 0 else 0
        
                # Semi-axis ratio
                axis_ratio = np.tan(abs(chi))
        
                # Define ellipse axes for plotting
                Ex = 1.0
                Ey = axis_ratio
        
                ellipticity = axis_ratio
        
                print("Orientation (deg):", np.round(alpha*180/np.pi,2),
                      " Ellipticity b/a:", np.round(axis_ratio,4))
        
                fit_results.append((Ex, Ey, alpha, ellipticity))
        
            except Exception:
                fit_results.append((np.nan,)*4)
              

        # ============================================================
        # MAP DETECTED SPOTS TO TRUE 7x7 GRID POSITIONS
        # ============================================================
        
        # import matplotlib.patches as patches
        
        NROWS, NCOLS = 7, 7
        
        coords = np.array(coordinates)  # (y, x)
        
        ys = coords[:, 0]
        xs = coords[:, 1]
        
        # --- Find approximate grid lines from detected data ---
        # Since grid is regular, we can cluster x and y positions
        
        # Estimate column positions
        sorted_x = np.sort(xs)
        sorted_y = np.sort(ys)
        
        # Because only 5 beams exist, we estimate grid spacing
        # using min and max extent
        
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        
        # Build full 7-column and 7-row grid coordinates
        x_grid = np.linspace(x_min, x_max, NCOLS)
        y_grid = np.linspace(y_min, y_max, NROWS)
        
        # Prepare container for full 49 beams
        full_fit_results = [(np.nan,)*4 for _ in range(49)]
        
        for idx, (y, x) in enumerate(coords):
        
            # Find nearest column index
            col_idx = np.argmin(np.abs(x_grid - x))
        
            # Find nearest row index
            row_idx = np.argmin(np.abs(y_grid - y))
        
            beam_number = col_idx * NROWS + row_idx  # 0-based
        
            full_fit_results[beam_number] = fit_results[idx]
        
        # ============================================================
        # PLOT FULL 7x7 GRID
        # ============================================================
        
        fig, axes = plt.subplots(7, 7, figsize=(12, 12))
        fig.suptitle("Intensity Ellipses")
        
        for idx in range(49):
        
            col = idx // NROWS
            row = idx % NROWS
            ax = axes[row, col]
        
            Ex, Ey, alpha, eE = full_fit_results[idx]
        
            if not np.isnan(Ex):
        
                scale = 1 / max(Ex**2, Ey**2)
        
                ellipse = patches.Ellipse(
                    (0, 0),
                    Ex**2 * scale*2.5,
                    Ey**2 * scale*2.5,
                    angle=np.degrees(alpha),
                    fill=False,
                    linewidth=2
                )
        
                ax.add_patch(ellipse)
        
            ax.text(0, -1.3, str(idx + 1),
                    ha='center',
                    fontsize=7)
        
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            ax.axis('off')
        
        
        
        plt.tight_layout()
        plt.savefig(os.path.join(parent_folder,os.path.basename(folder) + '__' + os.path.splitext(os.path.basename(file_path))[0] ))
        plt.close('all')
    # plt.show()
    #%%
    # ============================================================
    # CREATE ANIMATED GIF FROM PNGs
    # ============================================================

    from PIL import Image

    def make_gif_from_folder(folder, output_name="animation.gif", duration=500):
        """
        Makes an animated GIF from all .png files in a folder.
        
        Parameters:
            folder (str): Path to folder containing .png files
            output_name (str): Name of output GIF
            duration (int): Frame duration in milliseconds
        """
        
        # Find all PNG files
        png_files = [f for f in os.listdir(folder) if f.endswith(".png")]
        if not png_files:
            print("No PNG files found for GIF creation.")
            return
        
        # Sort files (important for correct order)
        png_files.sort()
        
        # Load images
        images = [Image.open(os.path.join(folder, f)) for f in png_files]
        
        # Save as GIF
        output_path = os.path.join(folder, output_name)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        
        print(f"Animated GIF saved to: {output_path}")


    # Run GIF creation in parent folder
    make_gif_from_folder(parent_folder, output_name="summary_animation.gif", duration=100)
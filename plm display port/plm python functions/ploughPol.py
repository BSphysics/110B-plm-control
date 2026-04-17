# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:57:50 2026

@author: bs426
"""

def plough_pol():

    import numpy as np
    import matplotlib.pyplot as plt
    import tkinter as tk
    from tkinter import filedialog
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max
    import scipy.ndimage
    from matplotlib import patches

    # ============================================================
    # SETTINGS
    # ============================================================
    plt.close('all')
    ROI_HALF_SIZE = 6
    POLARISER_ANGLES = np.arange(0, 200, 20)
    DEGREES = np.pi / 180
    PEAK_THRESHOLD_FRACTION = 0.2

    # ============================================================
    # SELECT FILE
    # ============================================================

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select .npy file",
        filetypes=[("NumPy files", "*.npy")]
    )

    root.destroy()

    if not file_path:
        raise ValueError("No file selected.")

    print(f"\nProcessing file:\n{file_path}")

    # ============================================================
    # LOAD DATA
    # ============================================================

    imgs = np.load(file_path)

    # ============================================================
    # DEFINE MAIN ROI (your crop)
    # ============================================================
    
    y1_sig, y2_sig = 80, 230
    x1_sig, x2_sig = 230, 390
    
    roi_height = y2_sig - y1_sig
    roi_width  = x2_sig - x1_sig
    
    # ============================================================
    # DEFINE BACKGROUND ROI (same size, shifted)
    # ============================================================
    
   
    dx = 100   # shift in x (columns)
    dy = 200     # shift in y (rows)
    
    x1_bg = x1_sig + dx
    x2_bg = x2_sig + dx
    y1_bg = y1_sig + dy
    y2_bg = y2_sig + dy
    
    # ============================================================
    # SAFETY CHECKS
    # ============================================================
    
    h, w = imgs.shape[1], imgs.shape[2]
    
    # Check bounds
    if x2_bg > w or y2_bg > h:
        raise ValueError("Background ROI goes outside image bounds. Adjust dx/dy.")
    
    # Check overlap
    overlap = not (
        x2_sig < x1_bg or x2_bg < x1_sig or
        y2_sig < y1_bg or y2_bg < y1_sig
    )
    
    if overlap:
        print("WARNING: Signal and background ROIs overlap!")
    
    # ============================================================
    # CREATE VISUALISATION
    # ============================================================
    
    import os
    # from matplotlib import patches
    
    img_full = np.sum(imgs, axis=0)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_full, cmap='gray')
    
    # Signal ROI (green)
    rect_sig = patches.Rectangle(
        (x1_sig, y1_sig),
        roi_width,
        roi_height,
        linewidth=2,
        edgecolor='lime',
        facecolor='none',
        label='Signal ROI'
    )
    ax.add_patch(rect_sig)
    
    # Background ROI (red)
    rect_bg = patches.Rectangle(
        (x1_bg, y1_bg),
        roi_width,
        roi_height,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        label='Background ROI'
    )
    ax.add_patch(rect_bg)
    
    ax.legend()
    ax.set_title("ROI sanity check")
    ax.axis('off')
    
    # ============================================================
    # SAVE IMAGE
    # ============================================================
    
    save_path = os.path.join(
        os.path.dirname(file_path),
        os.path.splitext(os.path.basename(file_path))[0] + "_ROI_check.png"
    )
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ROI check image to:\n{save_path}")
    
    # ============================================================
    # USE THESE FOR ANALYSIS
    # ============================================================
    
    imgs_cropped = imgs[:, y1_sig:y2_sig, x1_sig:x2_sig]

    img_ref = np.sum(imgs_cropped, axis=0).astype(np.float32)
    img_smooth = gaussian_filter(img_ref, sigma=1)

    # ============================================================
    # DETECT SPOTS
    # ============================================================

    threshold = np.max(img_smooth) * PEAK_THRESHOLD_FRACTION

    coordinates = peak_local_max(
        img_smooth,
        min_distance=8,
        threshold_abs=threshold
    )

    print(f"Detected {len(coordinates)} spots")

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

    # background_vals = np.sum(imgs[:, 200:211, 400:411], axis=(1, 2))
    background_vals = []

    for (x1, y1, x2, y2) in rois:
    
        # shift ROI into background region
        x1b = x1 + dx
        x2b = x2 + dx
        y1b = y1 + dy
        y2b = y2 + dy
    
        bg_stack = imgs[:, y1b:y2b+1, x1b:x2b+1]
        bg_sums = bg_stack.sum(axis=(1, 2))
    
        background_vals.append(bg_sums)
    
    background_vals = np.array(background_vals)

    powers_all = []
    
    for i, (x1, y1, x2, y2) in enumerate(rois):
    
        roi_stack = imgs_cropped[:, y1:y2+1, x1:x2+1]
        roi_sums = roi_stack.sum(axis=(1, 2))
    
        powers = np.clip(roi_sums - background_vals[i]*1, 0, None)
        powers_all.append(powers)
        
        # print(background_vals[i])

    powers_all = np.array(powers_all)

    # ============================================================
    # FIT POLARISATION
    # ============================================================

    theta = POLARISER_ANGLES * DEGREES

    M = np.column_stack([
        np.ones_like(theta),
        np.cos(2*theta),
        np.sin(2*theta)
    ])

    fit_results = []

    for i, powers in enumerate(powers_all):

        try:
            coeffs, _, _, _ = np.linalg.lstsq(M, powers, rcond=None)
            A, B, C = coeffs

            S0 = 2*A
            S1 = 2*B
            S2 = 2*C

            alpha = 0.5 * np.arctan2(S2, S1)

            S3_abs_sq = S0**2 - S1**2 - S2**2
            S3_abs_sq = max(S3_abs_sq, 0)
            S3_abs = np.sqrt(S3_abs_sq)

            chi = 0.5 * np.arcsin(S3_abs / S0) if S0 != 0 else 0
            axis_ratio = np.tan(abs(chi))

            print(f"Spot {i+1}: "
                  f"Angle = {np.round(alpha*180/np.pi,2)} deg, "
                  f"Ellipticity = {np.round((axis_ratio),3)}")

            fit_results.append((1.0, (axis_ratio), alpha))

        except Exception:
            fit_results.append((np.nan,)*3)
            
            
    # ============================================================
    # PLOT: Raw data + fit for ALL spots
    # ============================================================
    
    n_spots = len(powers_all)
    
    if n_spots == 0:
        print("No spots to plot.")
    else:
        # Grid size (nice square-ish layout)
        n_cols = int(np.ceil(np.sqrt(n_spots)))
        n_rows = int(np.ceil(n_spots / n_cols))
    
        theta = POLARISER_ANGLES * DEGREES
        theta_fine = np.linspace(0, np.max(theta), 300)
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = np.array(axes).reshape(-1)  # flatten in case of 2D grid
    
        for i in range(n_spots):
    
            ax = axes[i]
            powers = powers_all[i]
    
            try:
                # Recompute fit coefficients (safe + simple)
                M = np.column_stack([
                    np.ones_like(theta),
                    np.cos(2*theta),
                    np.sin(2*theta)
                ])
    
                coeffs, _, _, _ = np.linalg.lstsq(M, powers, rcond=None)
                A, B, C = coeffs
    
                fit_curve = A + B * np.cos(2 * theta_fine) + C * np.sin(2 * theta_fine)
    
                # Extract parameters again (for display)
                S0 = 2*A
                S1 = 2*B
                S2 = 2*C
    
                alpha = 0.5 * np.arctan2(S2, S1)
    
                S3_abs_sq = max(S0**2 - S1**2 - S2**2, 0)
                S3_abs = np.sqrt(S3_abs_sq)
    
                chi = 0.5 * np.arcsin(S3_abs / S0) if S0 != 0 else 0
                axis_ratio = np.tan(abs(chi))
    
                # --- Plot ---
                ax.plot(POLARISER_ANGLES, powers, 'o', markersize=5)
                ax.plot(theta_fine / DEGREES, fit_curve, '-', linewidth=1.5)
    
                ax.set_title(
                    f"Spot {i+1}\n"
                    f"{np.round(alpha*180/np.pi,1)}°, b/a={np.round(axis_ratio,3)}",
                    fontsize=9
                )
    
            except Exception:
                ax.set_title(f"Spot {i+1} (fit failed)", fontsize=9)
    
            ax.set_xlabel("Angle")
            ax.set_ylabel("Intensity")
            ax.tick_params(labelsize=7)
    
        # Hide unused subplots
        for j in range(n_spots, len(axes)):
            axes[j].axis('off')
    
        plt.tight_layout()
    
        # Save alongside data
        save_path = os.path.join(
            os.path.dirname(file_path),
            os.path.splitext(os.path.basename(file_path))[0] + "_all_fits.png"
        )
    
        plt.savefig(save_path, dpi=150)
        plt.close()
    
        print(f"Saved fit grid to:\n{save_path}")        
    

    # ============================================================
    # PLOT RESULT (simple, no grid reconstruction)
    # ============================================================

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_ref, cmap='gray')

    for i, ((y, x), (Ex, Ey, alpha)) in enumerate(zip(coordinates, fit_results)):

        if np.isnan(Ex):
            continue

        ellipse = patches.Ellipse(
            (x, y),
            width=20,
            height=20 * Ey,
            angle=np.degrees(alpha),
            fill=False,
            color='red',
            linewidth=2
        )

        ax.add_patch(ellipse)
        ax.text(x, y, str(i+1), color='yellow', fontsize=8)

    ax.set_title("Detected Spots + Polarisation")
    ax.axis('off')

    plt.tight_layout()
    # Save alongside data
    save_path = os.path.join(
        os.path.dirname(file_path),
        os.path.splitext(os.path.basename(file_path))[0] + "_ellipse_overlay.png"
    )

    plt.savefig(save_path, dpi=150)
    
# -*- coding: utf-8 -*-
"""
plough_pol.py  —  Polarisation analysis for plough (Big Dipper) spot array.

Spot detection strategy
-----------------------
The script looks for a  plough_centroids.npy  file in the script directory.
  • If found   → uses those fixed centroids (robust, order-preserving).
  • If missing → launches the interactive centroid-picker GUI so you can
                 click each spot once, then saves the result for future runs.

Run centroid_picker.py separately any time you want to re-pick the spots.
"""

def plough_pol(file_path=None):

    import numpy as np
    import matplotlib.pyplot as plt
    import tkinter as tk
    from tkinter import filedialog
    from scipy.ndimage import gaussian_filter
    import scipy.ndimage
    from matplotlib import patches
    import os
    import subprocess
    import sys

    scriptDir = os.getcwd()

    # ============================================================
    # SETTINGS
    # ============================================================
    plt.close('all')
    ROI_HALF_SIZE = 8
    POLARISER_ANGLES = np.arange(0, 200, 20)
    DEGREES = np.pi / 180
    N_SPOTS = 7

    # ============================================================
    # SELECT FILE
    # ============================================================

    if file_path is None:
        root = tk.Tk()
        root.withdraw()

        default_dir = r"C:\Users\bs426\Downloads\ben\ben\wrappers\Data"
        file_path = filedialog.askopenfilename(
            title="Select .npy file",
            initialdir=default_dir,
            filetypes=[("NumPy files", "*.npy")]
        )
        root.destroy()
    
        if not file_path:
            raise ValueError("No file selected.")

    print(f"\nProcessing file:\n{file_path}")

    data_dir  = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # ============================================================
    # LOAD DATA
    # ============================================================

    imgs = np.load(file_path)

    # ============================================================
    # DEFINE MAIN ROI (crop)
    # ============================================================

    y1_sig, y2_sig = 80, 230
    x1_sig, x2_sig = 230, 390

    roi_height = y2_sig - y1_sig
    roi_width  = x2_sig - x1_sig

    # ============================================================
    # DEFINE BACKGROUND ROI (same size, shifted)
    # ============================================================

    dx = 100
    dy = 200

    x1_bg = x1_sig + dx
    x2_bg = x2_sig + dx
    y1_bg = y1_sig + dy
    y2_bg = y2_sig + dy

    # ============================================================
    # SAFETY CHECKS
    # ============================================================

    h_full, w_full = imgs.shape[1], imgs.shape[2]

    if x2_bg > w_full or y2_bg > h_full:
        raise ValueError("Background ROI goes outside image bounds. Adjust dx/dy.")

    overlap = not (
        x2_sig < x1_bg or x2_bg < x1_sig or
        y2_sig < y1_bg or y2_bg < y1_sig
    )
    if overlap:
        print("WARNING: Signal and background ROIs overlap!")

    # ============================================================
    # ROI VISUALISATION
    # ============================================================

    img_full = np.sum(imgs, axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_full, cmap='gray')

    rect_sig = patches.Rectangle(
        (x1_sig, y1_sig), roi_width, roi_height,
        linewidth=2, edgecolor='lime', facecolor='none', label='Signal ROI'
    )
    rect_bg = patches.Rectangle(
        (x1_bg, y1_bg), roi_width, roi_height,
        linewidth=2, edgecolor='red', facecolor='none', label='Background ROI'
    )
    ax.add_patch(rect_sig)
    ax.add_patch(rect_bg)
    ax.legend()
    ax.set_title("ROI sanity check")
    ax.axis('off')

    save_path_roi = os.path.join(data_dir, base_name + "_ROI_check.png")
    plt.savefig(save_path_roi, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROI check image to:\n{save_path_roi}")

    # ============================================================
    # CROP TO SIGNAL ROI
    # ============================================================

    imgs_cropped = imgs[:, y1_sig:y2_sig, x1_sig:x2_sig]
    img_ref = np.sum(imgs_cropped, axis=0).astype(np.float32)

    # ============================================================
    # LOAD OR PICK CENTROIDS
    # ============================================================

    centroid_path = os.path.join(scriptDir, "plough_centroids.npy")

    if os.path.exists(centroid_path):
        centroids_xy = np.load(centroid_path)   # shape (N, 2): columns = x, y
        print(f"\nLoaded {len(centroids_xy)} centroids from:\n  {centroid_path}")

        if len(centroids_xy) != N_SPOTS:
            raise ValueError(
                f"Centroid file has {len(centroids_xy)} entries but N_SPOTS={N_SPOTS}. "
                "Delete the centroid file and re-run to re-pick."
            )

    else:
        print(f"\nNo centroid file found at:\n  {centroid_path}")
        print("Launching centroid picker — please click each spot in order.")

        from centroid_picker import pick_centroids

        centroids_xy = pick_centroids(file_path, n_spots=N_SPOTS)

        if centroids_xy is None:
            raise RuntimeError("Centroid picking was cancelled. Cannot proceed.")

        np.save(centroid_path, centroids_xy)
        print(f"\nSaved centroids to:\n  {centroid_path}")

    # ── convert full-image coords → cropped-ROI coords ──────────────────────
    coordinates_cropped = centroids_xy.copy()
    coordinates_cropped[:, 0] -= x1_sig   # x offset
    coordinates_cropped[:, 1] -= y1_sig   # y offset

    # (y, x) pairs — same convention as peak_local_max
    coordinates = [(int(round(cy)), int(round(cx)))
                   for cx, cy in coordinates_cropped]

    print(f"\nUsing {N_SPOTS} spots (in cropped-ROI coordinates):")
    for i, (y, x) in enumerate(coordinates):
        print(f"  Spot {i+1}: x={x}, y={y}")

    # ============================================================
    # BUILD ROIs
    # ============================================================

    rois = []
    h_crop, w_crop = img_ref.shape

    for (y, x) in coordinates:
        x1 = max(0, x - ROI_HALF_SIZE)
        x2 = min(w_crop - 1, x + ROI_HALF_SIZE)
        y1 = max(0, y - ROI_HALF_SIZE)
        y2 = min(h_crop - 1, y + ROI_HALF_SIZE)
        rois.append((x1, y1, x2, y2))

    # ============================================================
    # SPOT ROI OVERLAY — visual check on the summed (cropped) image
    # ============================================================

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_ref, cmap='gray', interpolation='nearest')

    for i, (x1, y1, x2, y2) in enumerate(rois):

        # Signal ROI box (green)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)

        # Background ROI box shifted back into cropped coords (red dashed)
        # Only draw if it fits inside the cropped image
        if (x1 + dx + (x2 - x1) <= w_crop) and (y1 + dy + (y2 - y1) <= h_crop):
            rect_bg_spot = patches.Rectangle(
                (x1 + dx, y1 + dy), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect_bg_spot)

        # Spot number label just above the box
        ax.text(x1, y1 - 2, str(i + 1),
                color='yellow', fontsize=8, fontweight='bold', va='bottom')

    # Legend proxies — use Rectangle as a concrete stand-in for Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='lime',  linewidth=1.5, linestyle='-',  label='Signal ROIs'),
        Line2D([0], [0], color='red',   linewidth=1.5, linestyle='--', label='Background ROIs'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7)
    ax.set_title("Spot ROI locations (summed image)", fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    save_path_spots = os.path.join(data_dir, base_name + "_spot_ROIs.png")
    plt.savefig(save_path_spots, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spot ROI overlay to:\n{save_path_spots}")

    # ============================================================
    # EXTRACT POWERS
    # ============================================================

    background_vals = []

    for (x1, y1, x2, y2) in rois:
        x1b = x1 + dx
        x2b = x2 + dx
        y1b = y1 + dy
        y2b = y2 + dy

        bg_stack = imgs[:, y1b:y2b+1, x1b:x2b+1]
        bg_sums  = bg_stack.sum(axis=(1, 2))
        background_vals.append(bg_sums)

    background_vals = np.array(background_vals)

    powers_all = []

    for i, (x1, y1, x2, y2) in enumerate(rois):
        roi_stack = imgs_cropped[:, y1:y2+1, x1:x2+1]
        roi_sums  = roi_stack.sum(axis=(1, 2))
        powers    = np.clip(roi_sums - background_vals[i], 0, None)
        powers_all.append(powers)

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

            S3_abs_sq = max(S0**2 - S1**2 - S2**2, 0)
            S3_abs    = np.sqrt(S3_abs_sq)

            chi        = 0.5 * np.arcsin(S3_abs / S0) if S0 != 0 else 0
            axis_ratio = np.tan(abs(chi))

            print(f"Spot {i+1}: "
                  f"Angle = {np.round(alpha*180/np.pi, 2)} deg, "
                  f"Ellipticity = {np.round(axis_ratio, 3)}")

            fit_results.append((1.0, axis_ratio, alpha))

        except Exception:
            fit_results.append((np.nan,)*3)

    # ============================================================
    # PLOT: Raw data + fit for ALL spots
    # ============================================================

    n_spots = len(powers_all)

    if n_spots == 0:
        print("No spots to plot.")
    else:
        n_cols = int(np.ceil(np.sqrt(n_spots)))
        n_rows = int(np.ceil(n_spots / n_cols))

        theta_fine = np.linspace(0, np.max(theta), 300)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = np.array(axes).reshape(-1)

        for i in range(n_spots):

            ax = axes[i]
            powers = powers_all[i]

            try:
                coeffs, _, _, _ = np.linalg.lstsq(M, powers, rcond=None)
                A, B, C = coeffs

                fit_curve = A + B * np.cos(2*theta_fine) + C * np.sin(2*theta_fine)

                S0 = 2*A
                S1 = 2*B
                S2 = 2*C

                alpha = 0.5 * np.arctan2(S2, S1)

                S3_abs_sq = max(S0**2 - S1**2 - S2**2, 0)
                S3_abs    = np.sqrt(S3_abs_sq)

                chi        = 0.5 * np.arcsin(S3_abs / S0) if S0 != 0 else 0
                axis_ratio = np.tan(abs(chi))

                ax.plot(POLARISER_ANGLES, powers, 'o', markersize=5)
                ax.plot(theta_fine / DEGREES, fit_curve, '-', linewidth=1.5)
                ax.set_title(
                    f"Spot {i+1}\n"
                    f"{np.round(alpha*180/np.pi, 1)}°, b/a={np.round(axis_ratio, 3)}",
                    fontsize=9
                )

            except Exception:
                ax.set_title(f"Spot {i+1} (fit failed)", fontsize=9)

            ax.set_xlabel("Angle")
            ax.set_ylabel("Intensity")
            ax.tick_params(labelsize=7)

        for j in range(n_spots, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()

        save_path_fits = os.path.join(data_dir, base_name + "_all_fits.png")
        plt.savefig(save_path_fits, dpi=150)
        plt.close()
        print(f"Saved fit grid to:\n{save_path_fits}")

    # ============================================================
    # ELLIPSE OVERLAY PLOT
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
        # ax.text(x, y, str(i+1), color='yellow', fontsize=8)

    ax.set_title("Detected Spots + Polarisation")
    ax.axis('off')

    plt.tight_layout()

    save_path_ell = os.path.join(data_dir, base_name + "_ellipse_overlay.png")
    plt.savefig(save_path_ell, dpi=150)
    plt.close()
    print(f"Saved ellipse overlay to:\n{save_path_ell}")

    # ============================================================
    # OPEN OUTPUT FOLDER IN FILE EXPLORER
    # ============================================================

    # print(f"\nOpening output folder:\n{data_dir}")

    # if sys.platform == "win32":
    #     os.startfile(data_dir)
    # elif sys.platform == "darwin":
    #     subprocess.Popen(["open", data_dir])
    # else:
    #     subprocess.Popen(["xdg-open", data_dir])

    
    if __name__ == "__main__":
        plough_pol()
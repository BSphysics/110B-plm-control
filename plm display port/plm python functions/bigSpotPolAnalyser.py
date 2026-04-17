# -*- coding: utf-8 -*-
"""
Big spot analyser
"""

def big_spot_pol_analyser():

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max
    import tkinter as tk
    from tkinter import filedialog
    from scipy.optimize import curve_fit
    from matplotlib import patches

    # ============================================================
    # SETTINGS
    # ============================================================

    POLARISER_ANGLES = np.arange(0, 200, 20)
    DEGREES = np.pi / 180
    PEAK_THRESHOLD_FRACTION = 0.15

    # ============================================================
    # SELECT PARENT DIRECTORY
    # ============================================================

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

    folder = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]


    imgs = np.load(file_path)

    # First find spot location in full frame
    img_full = np.sum(imgs, axis=0).astype(np.float32)
    img_full_smooth = gaussian_filter(img_full, sigma=2)
    img_full_smooth = img_full_smooth - np.min(img_full_smooth)
    img_full_smooth[img_full_smooth < 0] = 0

    # ----- centroid using image moments -----
    from scipy.ndimage import center_of_mass

    # ------------------------------------------------------------
    # Robust beam centre detection
    # ------------------------------------------------------------

    # Step 1: find brightest pixel (just to locate the beam roughly)
    py, px = np.unravel_index(np.argmax(img_full_smooth), img_full_smooth.shape)

    # Step 2: take a local window around it
    WINDOW = 90   # pixels (adjust if needed)

    y1 = max(0, py - WINDOW)
    y2 = min(img_full_smooth.shape[0], py + WINDOW)
    x1 = max(0, px - WINDOW)
    x2 = min(img_full_smooth.shape[1], px + WINDOW)

    local = img_full_smooth[y1:y2, x1:x2]

    # Step 3: compute centroid in the local window
    cy_local, cx_local = center_of_mass(local)

    # Convert back to full-frame coordinates
    cy_full = int(round(cy_local + y1))
    cx_full = int(round(cx_local + x1))

    print(f"Spot centre in full frame: y={cy_full}, x={cx_full}")

    print(f"Spot centre in full frame: y={cy_full}, x={cx_full}")

    # Crop around detected centre
    CROP_HALF = 60  # adjust to taste
    y1c = max(0, cy_full - CROP_HALF)
    y2c = min(imgs.shape[1], cy_full + CROP_HALF)
    x1c = max(0, cx_full - CROP_HALF)
    x2c = min(imgs.shape[2], cx_full + CROP_HALF)

    imgs_cropped = imgs[:, y1c:y2c, x1c:x2c]

    # ============================================================
    # BACKGROUND ROI - same size as crop, displaced to the left
    # ============================================================
    crop_height = y2c - y1c
    crop_width = x2c - x1c

    # Right edge of background ROI is 1 pixel left of crop left edge
    bg_x2 = x1c - 1
    bg_x1 = max(0, bg_x2 - crop_width)
    bg_y1 = y1c
    bg_y2 = y2c

    # Warn if background ROI has been clipped by image edge
    actual_bg_width = bg_x2 - bg_x1
    if actual_bg_width < crop_width:
        print(f"Warning: background ROI clipped by image edge ({actual_bg_width} vs {crop_width} pixels wide)")


    delta = -150
    imgs_background = imgs[:, bg_y1 - delta:bg_y2 - delta, bg_x1 - delta:bg_x2-delta]

    # ============================================================
    # DIAGNOSTIC: Full frame with ROI rectangles overlaid
    # ============================================================
    img_full_display = np.sum(imgs, axis=0).astype(np.float32)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_full_display, cmap='gray')

    # Red rectangle: spot crop ROI
    spot_rect = patches.Rectangle(
        (x1c, y1c),          # (x, y) of top-left corner
        x2c - x1c,           # width
        y2c - y1c,           # height
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        label='Spot ROI'
    )
    ax.add_patch(spot_rect)

    # Blue rectangle: background ROI
    bg_rect = patches.Rectangle(
        (bg_x1 - delta, bg_y1 - delta),  # shifted position
        bg_x2 - bg_x1,                    # width unchanged
        bg_y2 - bg_y1,                    # height unchanged
        linewidth=2,
        edgecolor='blue',
        facecolor='none',
        label='Background ROI'
    )
    ax.add_patch(bg_rect)

    ax.legend(loc='upper right')
    ax.set_title(f"Full frame with ROIs  |  Spot centre: y={cy_full}, x={cx_full}")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, os.path.basename(folder) + '__roi_diagnostic.png'), dpi=150)
    # plt.show()

    # ============================================================
    # BUILD BEAM MASK (5% intensity threshold)
    # ============================================================

    beam_sum = np.sum(imgs_cropped, axis=0).astype(np.float32)

    # smooth slightly to avoid noisy mask edges
    beam_sum_smooth = gaussian_filter(beam_sum, sigma=2)

    # normalise
    beam_sum_smooth -= beam_sum_smooth.min()
    beam_sum_smooth /= beam_sum_smooth.max()

    # threshold mask
    THRESH = 0.1
    beam_mask = beam_sum_smooth > THRESH

    # ============================================================
    # EXTRACT Intensities
    # ============================================================

    background_intensity = np.sum(imgs_background * beam_mask, axis=(1,2))

    img_intensity = np.sum(imgs_cropped * beam_mask, axis=(1,2))

    spot_intensity = np.clip(img_intensity - background_intensity, 0, None)
    # spot_intensity = img_intensity - background_intensity

    plt.imshow(np.sum(imgs_cropped, axis=0)  - np.sum(imgs_background, axis=0))

    plt.figure(figsize=(6,6))
    plt.imshow(beam_sum, cmap='gray')
    plt.contour(beam_mask, colors='red', linewidths=1)
    plt.title("Beam mask (5% threshold)")
    plt.savefig(os.path.join(folder, os.path.basename(folder) + '__intensity_mask.png'), dpi=300)
    # plt.show()

    #%%
      
    # ============================================================
    # FIT MODEL (FULLY POLARISED STOKES RECONSTRUCTION)
    # ============================================================

    theta = POLARISER_ANGLES * DEGREES

    # Design matrix
    M = np.column_stack([
        np.ones_like(theta),
        np.cos(2*theta),
        np.sin(2*theta)
    ])

    # Fit to the full spot_intensity vector (one value per polariser angle)
    coeffs, _, _, _ = np.linalg.lstsq(M, spot_intensity, rcond=None)
    A, B, C = coeffs

    # Stokes parameters
    S0 = 2*A
    S1 = 2*B
    S2 = 2*C

    # Orientation
    alpha = 0.5 * np.arctan2(S2, S1)

    # Reconstruct |S3| (fully polarised assumption)
    S3_abs_sq = max(S0**2 - S1**2 - S2**2, 0)
    S3_abs = np.sqrt(S3_abs_sq)

    # Ellipticity
    chi = 0.5 * np.arcsin(S3_abs / S0) if S0 != 0 else 0
    axis_ratio = np.tan(abs(chi))

    print(f"Orientation (deg): {np.round(alpha*180/np.pi, 2)}")
    print(f"Ellipticity b/a:   {np.round(axis_ratio, 4)}")

    # ============================================================
    # PLOT: Raw data + fitted curve
    # ============================================================

    theta_fine = np.linspace(0, np.max(POLARISER_ANGLES) * DEGREES, 300)
    fit_curve = A + B * np.cos(2 * theta_fine) + C * np.sin(2 * theta_fine)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: data + fit ---
    ax1.plot(POLARISER_ANGLES, spot_intensity, 'o', markersize=7, label='Raw data')
    ax1.plot(theta_fine / DEGREES, fit_curve, '-', linewidth=2, label='Fit')
    ax1.set_ylim(0, np.max(spot_intensity)*1.1)
    ax1.set_xlabel('Polariser angle (degrees)')
    ax1.set_ylabel('Integrated intensity (counts)')
    ax1.set_title(f"Orientation: {np.round(alpha*180/np.pi, 1)}°  |  Ellipticity b/a: {np.round(axis_ratio, 4)}")
    ax1.legend()

    # --- Right panel: polarisation ellipse ---
    ellipse = patches.Ellipse(
        (0, 0),
        width=3 * axis_ratio,   # minor axis along x
        height=3,               # major axis along y
        angle=np.degrees(alpha),
        fill=False,
        linewidth=2,
        edgecolor='blue'
    )
    ax2.add_patch(ellipse)

    # # Draw orientation line through centre
    # dx = np.cos(alpha)
    # dy = np.sin(alpha)
    # ax2.plot([-dx, dx], [-dy, dy], 'r-', linewidth=1.5, label='Orientation')

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    # ax2.axhline(0, color='gray', linewidth=0.5)
    # ax2.axvline(0, color='gray', linewidth=0.5)
    ax2.set_title('E-field polarisation ellipse')
    # ax2.legend()
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, os.path.basename(folder) + '__fit_and_global_ellipse.png'), dpi=300)

    #%%

    # ============================================================
    # SPATIALLY RESOLVED POLARISATION ANALYSIS
    # ============================================================

    ROI_SIZE = 4  # pixels

    img_crop_sum = np.sum(imgs_cropped, axis=0)
    crop_h, crop_w = img_crop_sum.shape

    n_rows_grid = crop_h // ROI_SIZE
    n_cols_grid = crop_w // ROI_SIZE

    # Storage for results
    orientation_map = np.full((n_rows_grid, n_cols_grid), np.nan)
    ellipticity_map = np.full((n_rows_grid, n_cols_grid), np.nan)
    intensity_map   = np.full((n_rows_grid, n_cols_grid), np.nan)

    for row_idx in range(n_rows_grid):
        for col_idx in range(n_cols_grid):

            y1r = row_idx * ROI_SIZE
            y2r = y1r + ROI_SIZE
            x1r = col_idx * ROI_SIZE
            x2r = x1r + ROI_SIZE

            # Extract intensity vs polariser angle for this sub-ROI
            roi_stack = imgs_cropped[:, y1r:y2r, x1r:x2r]
            roi_intensity = roi_stack.sum(axis=(1, 2)).astype(np.float64)

            # Background subtraction - scale background by sub-ROI area vs full background area
            sub_roi_area = ROI_SIZE * ROI_SIZE
            bg_area = imgs_background.shape[1] * imgs_background.shape[2]
            scaled_bg = background_intensity * (sub_roi_area / bg_area)
            roi_intensity = np.clip(roi_intensity - scaled_bg, 0, None)

            # Skip if sub-ROI is too dim (likely outside the beam)
            roi_mean_per_pixel = np.mean(roi_intensity) / (ROI_SIZE * ROI_SIZE)
            img_mean_per_pixel = np.max(img_crop_sum) / imgs.shape[0]  # max pixel, per frame
            
            # Skip only truly dark/empty ROIs
            if roi_mean_per_pixel < 0.1 * img_mean_per_pixel:
                continue

            try:
                coeffs, _, _, _ = np.linalg.lstsq(M, roi_intensity, rcond=None)
                Ar, Br, Cr = coeffs

                S0r = 2*Ar
                S1r = 2*Br
                S2r = 2*Cr

                alphar = 0.5 * np.arctan2(S2r, S1r)

                S3_sq = max(S0r**2 - S1r**2 - S2r**2, 0)
                S3r = np.sqrt(S3_sq)

                chir = 0.5 * np.arcsin(S3r / S0r) if S0r != 0 else 0
                axis_ratio_r = np.tan(abs(chir))

                orientation_map[row_idx, col_idx] = np.degrees(alphar)
                ellipticity_map[row_idx, col_idx] = axis_ratio_r
                intensity_map[row_idx, col_idx]   = np.mean(roi_intensity)

            except Exception:
                pass

    # ============================================================
    # PLOT SPATIAL MAPS
    # ============================================================

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Spatially resolved polarisation  |  ROI size: {ROI_SIZE}x{ROI_SIZE} px")

    im0 = axes[0].imshow(intensity_map, cmap='hot', origin='upper')
    axes[0].set_title('Intensity')
    plt.colorbar(im0, ax=axes[0], label='Counts')

    im1 = axes[1].imshow(orientation_map, cmap='hsv', origin='upper', vmin=-90, vmax=90)
    axes[1].set_title('Orientation (degrees)')
    plt.colorbar(im1, ax=axes[1], label='Degrees')

    im2 = axes[2].imshow(ellipticity_map, cmap='viridis', origin='upper', vmin=0, vmax=1)
    axes[2].set_title('Ellipticity b/a')
    plt.colorbar(im2, ax=axes[2], label='b/a ratio')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, os.path.basename(folder) + '__spatial_pol_map.png'), dpi=150)
    # plt.show()

    # ============================================================
    # ELLIPSE PLOT - polarisation ellipses overlaid on intensity
    # ============================================================

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img_crop_sum, cmap='gray', origin='upper')

    for row_idx in range(n_rows_grid):
        for col_idx in range(n_cols_grid):
            if np.isnan(orientation_map[row_idx, col_idx]):
                continue

            cx_r = col_idx * ROI_SIZE + ROI_SIZE / 2
            cy_r = row_idx * ROI_SIZE + ROI_SIZE / 2

            angle_deg = orientation_map[row_idx, col_idx]
            b_over_a  = ellipticity_map[row_idx, col_idx]

            # Scale ellipse to fit within ROI, semi-major axis = 45% of ROI_SIZE
            semi_major = ROI_SIZE * 0.45
            semi_minor = semi_major * b_over_a **1

            ellipse = patches.Ellipse(
                (cx_r, cy_r),
                width=2 * semi_major,
                height=2 * semi_minor,
                angle=angle_deg,
                fill=False,
                linewidth=1.5,
                edgecolor='red'
            )
            ax.add_patch(ellipse)

    ax.set_title('Field ellipses overlaid on intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, os.path.basename(folder) + '__spatial_pol_ellipses.png'), dpi=150)
    # plt.show()
    plt.close('all')

    #%%
    # ============================================================
    # PUBLICATION-QUALITY IMAGE PREPARATION (donut-safe)
    # ============================================================

    img_display = img_crop_sum.copy()

    # Background subtraction
    bg = np.percentile(img_display, 5)
    img_display = img_display - bg
    img_display[img_display < 0] = 0

    from scipy.ndimage import gaussian_filter

    # Compute summed image
    img_display = img_crop_sum.copy()

    # Background subtraction
    bg = np.percentile(img_display, 5)
    img_display -= bg
    img_display[img_display < 0] = 0

    # Apply Gaussian smoothing only inside beam mask
    beam_mask_bool = beam_mask.astype(bool)

    img_smooth = np.zeros_like(img_display)

    temp = img_display * beam_mask_bool
    img_smooth[beam_mask_bool] = gaussian_filter(temp, sigma=1.2)[beam_mask_bool]

    # Normalise
    img_smooth /= np.max(img_smooth)

    # Optional gamma
    img_smooth = img_smooth**0.8

    # Normalise for display
    img_display = img_display / np.max(img_display)

    # Optional mild gamma for nicer contrast
    img_display = img_display**0.8


    # ============================================================
    # ELLIPSE PLOT — polarisation ellipses overlaid on HQ image
    # ============================================================

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img_display, cmap='gray', origin='upper')

    # Normalise intensity map for alpha mapping (ignore NaNs)
    int_map_valid = intensity_map.copy()
    int_map_min = np.nanmin(int_map_valid)
    int_map_max = np.nanmax(int_map_valid)
    int_map_norm = (int_map_valid - int_map_min) / (int_map_max - int_map_min + 1e-12)

    # Optional: apply a power to increase contrast between dim and bright ellipses
    ALPHA_GAMMA = 0.5  # < 1 boosts mid-range; > 1 suppresses dim ellipses more aggressively
    ALPHA_MIN   = 0.0 # set to 0 to hide faint ellipses

    for row_idx in range(n_rows_grid):
        for col_idx in range(n_cols_grid):
            if np.isnan(orientation_map[row_idx, col_idx]):
                continue
                            
            # if not beam_mask_bool[int(cy_r), int(cx_r)]:
                # continue

            cx_r = col_idx * ROI_SIZE + ROI_SIZE / 2
            cy_r = row_idx * ROI_SIZE + ROI_SIZE / 2

            angle_deg = orientation_map[row_idx, col_idx]
            b_over_a  = ellipticity_map[row_idx, col_idx]

            INTENSITY_GAMMA = 0.5   # same idea as alpha gamma (tune this)
            SIZE_MIN = 0.05         # prevents ellipses disappearing completely
            
            local_int = float(int_map_norm[row_idx, col_idx]) ** INTENSITY_GAMMA
            scale = max(SIZE_MIN, local_int)
            
            semi_major = ROI_SIZE * 0.45 * scale
            semi_minor = semi_major * b_over_a ** 1

            # Alpha scaled by normalised intensity
            raw_alpha = float(int_map_norm[row_idx, col_idx]) ** ALPHA_GAMMA
            ellipse_alpha = 0.9 #max(ALPHA_MIN, raw_alpha)

            ellipse = patches.Ellipse(
                (cx_r, cy_r),
                width=2 * semi_major,
                height=2 * semi_minor,
                angle=angle_deg,
                fill=False,
                linewidth=1.0,
                edgecolor='red',
                alpha=ellipse_alpha
            )
            ax.add_patch(ellipse)

    # ax.set_title('E-field ellipses overlaid on intensity (HQ)')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, os.path.basename(folder) + '__spatial_pol_ellipses_HQ.png'), dpi=150)
    # plt.show()

    # ============================================================
    # FIT PLOT FOR CENTRAL ROI ONLY
    # ============================================================
    #%%
    # Find the grid cell whose centre is closest to the beam centre
    # Beam centre in crop coordinates
    cx_crop = cx_full - x1c
    cy_crop = cy_full - y1c

    # Convert to grid indices
    centre_col = int(cx_crop // ROI_SIZE)
    centre_row = int(cy_crop // ROI_SIZE)

    # Clamp to valid grid range
    centre_col = np.clip(centre_col, 0, n_cols_grid - 1)
    centre_row = np.clip(centre_row, 0, n_rows_grid - 1)

    # Re-extract intensity for that ROI
    y1r = centre_row * ROI_SIZE
    y2r = y1r + ROI_SIZE
    x1r = centre_col * ROI_SIZE
    x2r = x1r + ROI_SIZE

    roi_stack_centre = imgs_cropped[:, y1r:y2r, x1r:x2r]
    roi_intensity_centre = roi_stack_centre.sum(axis=(1, 2)).astype(np.float64)

    # Background subtraction
    sub_roi_area = ROI_SIZE * ROI_SIZE
    bg_area = imgs_background.shape[1] * imgs_background.shape[2]
    scaled_bg_centre = background_intensity * (sub_roi_area / bg_area)
    roi_intensity_centre = np.clip(roi_intensity_centre - scaled_bg_centre, 0, None)

    # Fit
    coeffs_c, _, _, _ = np.linalg.lstsq(M, roi_intensity_centre, rcond=None)
    Ac, Bc, Cc = coeffs_c

    S0c = 2 * Ac
    S1c = 2 * Bc
    S2c = 2 * Cc

    alphac = 0.5 * np.arctan2(S2c, S1c)

    S3c_sq = max(S0c**2 - S1c**2 - S2c**2, 0)
    S3c = np.sqrt(S3c_sq)

    chic = 0.5 * np.arcsin(S3c / S0c) if S0c != 0 else 0
    axis_ratio_c = np.tan(abs(chic))

    print(f"Central ROI — Orientation (deg): {np.round(alphac * 180 / np.pi, 2)}")
    print(f"Central ROI — Ellipticity b/a:   {np.round(axis_ratio_c, 4)}")

    # Plot
    fit_curve_c = Ac + Bc * np.cos(2 * theta_fine) + Cc * np.sin(2 * theta_fine)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    # fig.suptitle(f"Central ROI  (grid [{centre_row}, {centre_col}],  crop px [{y1r}:{y2r}, {x1r}:{x2r}])")

    ax1.plot(POLARISER_ANGLES, roi_intensity_centre, 'o', markersize=7, label='Raw data')
    ax1.plot(theta_fine / DEGREES, fit_curve_c, '-', linewidth=2, label='Fit')
    ax1.set_ylim(0, np.max(roi_intensity_centre) * 1.1)
    ax1.set_xlabel('Polariser angle (degrees)')
    ax1.set_ylabel('Integrated intensity (counts)')
    ax1.set_title(f"Orientation: {np.round(alphac * 180 / np.pi, 1)}°  |  Ellipticity b/a: {np.round(axis_ratio_c, 4)}")
    ax1.legend()

    ellipse_c = patches.Ellipse(
        (0, 0),
        width=3 * axis_ratio_c,
        height=3,
        angle=np.degrees(alphac),
        fill=False,
        linewidth=2,
        edgecolor='blue'
    )


    plt.tight_layout()
    plt.savefig(os.path.join(folder, os.path.basename(folder) + '__central_roi_fit.png'), dpi=300)
    # plt.show()


    os.startfile(folder)

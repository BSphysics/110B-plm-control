# # Uses fitting to analyse polarisation ellipse of 49 beams

def pol_analyse_49_beams(manual_path=None):

    import numpy as np
    import cv2
    import os
    import matplotlib.pyplot as plt
    from tkinter import Tk, filedialog
    from skimage.feature import peak_local_max
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import curve_fit
    from matplotlib import patches

    degrees = np.pi / 180
    polariserAngles = np.arange(0, 200, 20)
    N_SPOTS = 49
    ROI_HALF_SIZE = 5
    NROWS, NCOLS = 7, 7

    # ============================================================
    # LOAD FILE
    # ============================================================

    if manual_path is None:
        root = Tk()
        root.withdraw()
        default_path = r'D:\PLM\plm python control\wrappers\Data'
        file_path = filedialog.askopenfilename(
            title="Select the npy file",
            initialdir=default_path)
        root.destroy()
    else:
        file_path = manual_path

    if not file_path or not os.path.exists(file_path):
        print("No valid file path.")
        return None

    print(f"Analysing images saved here: {file_path}")
    imgs = np.load(file_path)

    # ============================================================
    # CROP ONCE
    # ============================================================

    imgs_cropped = imgs[:, 100:250, 220:370]
    brightest_index = np.argmax(imgs_cropped.sum(axis=(1, 2)))
    img_ref = imgs_cropped[brightest_index].astype(np.float32)

    img_smooth = gaussian_filter(img_ref, sigma=1)

    # ============================================================
    # FIND CENTROIDS
    # ============================================================

    coordinates = peak_local_max(
        img_smooth,
        min_distance=5,
        num_peaks=N_SPOTS
    )

    by_x = coordinates[coordinates[:, 1].argsort()]
    columns = np.array_split(by_x, NCOLS)

    sorted_centroids = []
    for col in columns:
        col_sorted = col[col[:, 0].argsort()]
        sorted_centroids.extend(col_sorted)

    sorted_centroids = np.array(sorted_centroids)

    # ============================================================
    # DEFINE ROIs
    # ============================================================

    rois = []
    h, w = img_ref.shape

    for (y, x) in sorted_centroids:
        x, y = int(x), int(y)
        x1 = max(0, x - ROI_HALF_SIZE)
        x2 = min(w - 1, x + ROI_HALF_SIZE)
        y1 = max(0, y - ROI_HALF_SIZE)
        y2 = min(h - 1, y + ROI_HALF_SIZE)
        rois.append((x1, y1, x2, y2))

    # ============================================================
    # PRECOMPUTE BACKGROUND + POWERS
    # ============================================================

    background_vals = np.sum(imgs[:, 200:211, 400:411], axis=(1, 2))

    powers_all = np.zeros((N_SPOTS, len(imgs)))

    for idx, (x1, y1, x2, y2) in enumerate(rois):
        roi_stack = imgs_cropped[:, y1:y2+1, x1:x2+1]
        roi_sums = roi_stack.sum(axis=(1, 2))
        powers_all[idx] = np.clip(roi_sums - background_vals, 0, None)

    # ============================================================
    # FIT FUNCTION (ONLY ONCE)
    # ============================================================

    def model_f(theta, p1, p2, p3, p4):
        return (p1*np.cos(theta*degrees-p3))**2 + \
               (p2*np.sin(theta*degrees-p3))**2 + p4

    fit_results = []

    for idx in range(N_SPOTS):

        powers = powers_all[idx]

        try:
            popt, _ = curve_fit(
                model_f,
                polariserAngles,
                powers,
                bounds=(
                    [0, 0, 0, 0],
                    [np.max(powers)*1.5,
                     np.max(powers)*1.5,
                     2*np.pi,
                     0.01]
                )
            )

            Ex, Ey, alpha, offset = popt

            if Ey > Ex:
                Ex, Ey = Ey, Ex
                alpha += np.pi/2

            alpha = alpha % np.pi

            ellipticity_E = Ey/Ex if Ex != 0 else 0
            ellipticity_I = (Ey/Ex)**2 if Ex != 0 else 0

            fit_results.append(
                (Ex, Ey, alpha, ellipticity_E, ellipticity_I)
            )

        except RuntimeError:
            fit_results.append((np.nan,)*5)

    # ============================================================
    # E-FIELD ELLIPSES GRID (WITH BEAM NUMBERS)
    # ============================================================

    fig, axes = plt.subplots(7, 7, figsize=(12, 12))
    fig.suptitle("E-field Ellipses")

    for idx in range(N_SPOTS):

        col = idx // NROWS
        row = idx % NROWS
        ax = axes[row, col]

        Ex, Ey, alpha, eE, _ = fit_results[idx]

        if not np.isnan(Ex):
            scale = 1/max(Ex, Ey)
            ellipse = patches.Ellipse(
                (0, 0),
                Ex*scale,
                Ey*scale,
                angle=alpha*180/np.pi,
                fill=False,
                linewidth=2
            )
            ax.add_patch(ellipse)

        # Beam number under ellipse
        ax.text(0, -1.3, str(idx+1),
                ha='center',
                fontsize=7)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(file_path),
                             "E-field ellipses ROIs.png"))
    plt.close()

    # ============================================================
    # INTENSITY ELLIPSES GRID
    # ============================================================

    fig, axes = plt.subplots(7, 7, figsize=(12, 12))
    fig.suptitle("Intensity Ellipses")

    for idx in range(N_SPOTS):

        col = idx // NROWS
        row = idx % NROWS
        ax = axes[row, col]
        Ex, Ey, alpha, _, eI = fit_results[idx]

        if not np.isnan(Ex):
            Ex2, Ey2 = Ex**2, Ey**2
            scale = 1/max(Ex2, Ey2)
            ellipse = patches.Ellipse(
                (0, 0),
                Ex2*scale,
                Ey2*scale,
                angle=alpha*180/np.pi,
                fill=False,
                linewidth=2
            )
            ax.add_patch(ellipse)

        ax.text(0, -1.3, str(idx+1),
                ha='center',
                fontsize=7)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(file_path),
                             "Intensity ellipses ROIs.png"))
    plt.close()

    # ============================================================
    # BAR CHART (E-FIELD)
    # ============================================================

    ellipticities_E = [r[3] for r in fit_results]
    ellipticities_I = [r[4] for r in fit_results]

    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(1, 50), ellipticities_E)
    plt.ylim(0, 0.5)
    plt.xlabel("Beam Number")
    plt.ylabel("Ellipticity (Ey/Ex)")
    plt.title("E-field Ellipticity")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(file_path),
                             "Ellipticity bar chart.png"))
    plt.close()

    # ============================================================
    # PRINT MEANS
    # ============================================================

    print("\nMean E-field ellipticity:",
          np.round(np.nanmean(ellipticities_E), 4))

    print("Mean Intensity ellipticity:",
          np.round(np.nanmean(ellipticities_I), 4))

    print("\nBack to free streaming")

    return ellipticities_I



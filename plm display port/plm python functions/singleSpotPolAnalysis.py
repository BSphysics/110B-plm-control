"""
singleSpotPolAnalysis.py

Polarisation analysis of the brightest spot in a pol_measure() stack
(rotating linear analyser, angles 0:20:180 deg by default).

What it reports, and what each quantity assumes
-----------------------------------------------
Measured directly (no assumptions beyond Malus's law):
    S0, S1, S2           from I(theta) = 1/2 [S0 + S1 cos 2theta + S2 sin 2theta]
    azimuth              1/2 atan2(S2, S1), in the analyser frame (+ phi0 if given)
    p_lin                sqrt(S1^2 + S2^2) / S0   (degree of LINEAR polarisation)
    extinction           Imin/Imax over analyser angle = (1 - p_lin)/(1 + p_lin)
Inferred, only valid if the beam is fully polarised (DOP = 1):
    |S3|, |chi|, ellipticity b/a = tan|chi|
    A rotating linear analyser cannot tell circular from unpolarised light, so
    "ellipticity" here is an upper bound that includes any depolarisation.

Background: the per-pixel background (median of a border region in each frame) is
multiplied by the number of ROI pixels.  Subtracting a fixed-size patch sum from a
larger ROI sum leaves an offset b, and because |S3| = sqrt(S0^2 - S1^2 - S2^2),
the spurious ellipticity grows like sqrt(b/S0): a 1 % offset reads as b/a ~ 0.07.

Usage
-----
    from singleSpotPolAnalysis import single_spot_pol_analysis
    res = single_spot_pol_analysis(filename)          # e.g. the path pol_measure returns
    res = single_spot_pol_analysis()                  # opens a file dialog
Optional: angles_deg (if not 0:20:180), phi0_deg (analyser axis offset relative to
your channel-A axis), roi_radius (px; default from the spot size).
Saves <stem>__single_spot_pol.png and .json next to the data file.
"""
import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter


class NoSignalError(RuntimeError):
    """Raised when the stack contains no usable spot (laser off, shutter closed...)."""


def _pick_file():
    from PyQt5.QtWidgets import QApplication, QFileDialog
    app = QApplication.instance()
    owns = app is None
    if owns:
        app = QApplication([])
    path, _ = QFileDialog.getOpenFileName(None, "Select pol_measure .npy", "", "NumPy (*.npy)")
    if owns:
        app.quit()
    return path


def _border_background(img, width=10):
    """Per-pixel background from the image border (robust to the spot)."""
    b = np.concatenate([img[:width].ravel(), img[-width:].ravel(),
                        img[:, :width].ravel(), img[:, -width:].ravel()])
    return float(np.median(b))


def fit_malus(theta, I):
    """Linear least squares for I = 1/2 [S0 + S1 cos2t + S2 sin2t]. Returns S, resid."""
    M = 0.5 * np.column_stack([np.ones_like(theta), np.cos(2 * theta), np.sin(2 * theta)])
    S, *_ = np.linalg.lstsq(M, I, rcond=None)
    resid = I - M @ S
    return S, resid, M


def single_spot_pol_analysis(file_path=None, angles_deg=None, phi0_deg=0.0,
                             roi_radius=None, roi_threshold=0.05, bg_border_px=10,
                             min_snr=20.0, saturation=None, max_resid=0.05,
                             save=True, show=False):
    if file_path is None:
        file_path = _pick_file()
        if not file_path:
            raise ValueError("No file selected.")

    stack = np.load(file_path).astype(np.float64)          # (n_angles, H, W)
    n = stack.shape[0]
    # pol_measure (QWP version) writes <stem>_meta.json with the angles and channel-A offset
    meta_path = os.path.splitext(file_path)[0] + "_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if angles_deg is None:
            angles_deg = meta.get("angles_deg")
        if not phi0_deg:
            phi0_deg = meta.get("phi0_deg", 0.0)
    if angles_deg is None:
        angles_deg = np.arange(0, 20 * n, 20)[:n]
    angles_deg = np.asarray(angles_deg, dtype=float)
    if len(angles_deg) != n:
        raise ValueError(f"{n} frames but {len(angles_deg)} angles")
    theta = np.deg2rad(angles_deg)

    # --- data sanity checks -----------------------------------------------------
    warnings_ = []
    frame_max = stack.reshape(n, -1).max(axis=1)
    empty = np.where(frame_max == 0)[0]
    if len(empty):
        warnings_.append(f"frames {empty.tolist()} (angles {angles_deg[empty].tolist()} deg) are "
                         f"all zero - pol_measure probably stopped early; they are excluded")
        keep = frame_max > 0
        stack, angles_deg, theta, n = stack[keep], angles_deg[keep], theta[keep], int(keep.sum())
        if n < 4:
            raise NoSignalError(f"only {n} usable frames")
    full = saturation if saturation is not None else (255.0 if stack.max() <= 255 else 4095.0)
    # pol_measure saves means of many frames, so a mean near full scale already
    # implies clipping in individual frames
    sat = np.where(stack.reshape(n, -1).max(axis=1) >= 0.95 * full)[0]
    if len(sat):
        warnings_.append(f"frames at {angles_deg[sat].tolist()} deg reach >=95 % of full scale "
                         f"({full:g}) - likely saturated; reduce exposure")

    # --- background, per frame, per pixel ------------------------------------
    bg = np.array([_border_background(f, bg_border_px) for f in stack])
    sub = stack - bg[:, None, None]

    # --- brightest spot and ROI ----------------------------------------------
    ref = gaussian_filter(sub.mean(axis=0), 1.5)          # analyser-averaged ~ S0/2
    noise = 1.4826 * np.median(np.abs(ref - np.median(ref)))
    if not np.isfinite(ref).all() or ref.max() < min_snr * max(noise, 1e-6):
        raise NoSignalError(f"No spot found: peak {ref.max():.2f} DN vs noise {noise:.3f} DN "
                            f"(need > {min_snr:g}x). Laser off or shutter closed?")
    cy, cx = np.unravel_index(np.argmax(ref), ref.shape)
    yy, xx = np.indices(ref.shape)
    r = np.hypot(yy - cy, xx - cx)
    if roi_radius is None:
        # radius enclosing everything above roi_threshold of the peak, + 3 px margin
        above = (ref > roi_threshold * ref[cy, cx])
        roi_radius = float(r[above & (r < 0.5 * min(ref.shape))].max()) + 3
    roi = r <= roi_radius
    # recentre on the intensity-weighted centroid inside the first ROI
    w = np.clip(ref, 0, None) * roi
    cy, cx = np.sum(w * yy) / w.sum(), np.sum(w * xx) / w.sum()
    r = np.hypot(yy - cy, xx - cx)
    roi = r <= roi_radius

    I = sub[:, roi].sum(axis=1)                            # background-corrected ROI sums

    # --- fit (with single-outlier rejection) -------------------------------------
    S, resid, M = fit_malus(theta, I)
    excluded = []
    if n >= 6:
        rms_all = np.sqrt(np.sum(resid ** 2) / max(n - 3, 1))
        best = None
        for j in range(n):
            k = np.arange(n) != j
            Sj, rj, _ = fit_malus(theta[k], I[k])
            rms_j = np.sqrt(np.sum(rj ** 2) / max(n - 4, 1))
            if best is None or rms_j < best[1]:
                best = (j, rms_j)
        j, rms_j = best
        if rms_all > max_resid * S[0] / 2 and rms_j < rms_all / 3:
            excluded = [float(angles_deg[j])]
            warnings_.append(f"frame at {angles_deg[j]:g} deg is an outlier (fit rms drops "
                             f"{rms_all / rms_j:.1f}x without it) - excluded. Likely a laser/PLM "
                             f"power transient or a mis-positioned stage")
            keep = np.arange(n) != j
            theta, I, angles_deg, n = theta[keep], I[keep], angles_deg[keep], n - 1
            S, resid, M = fit_malus(theta, I)
    S0, S1, S2 = S
    if not np.isfinite(S).all() or S0 <= 0:
        raise NoSignalError(f"Fit gave S0 = {S0:.3g}; no usable signal in the ROI.")
    L = np.hypot(S1, S2)
    p_lin = L / S0
    azimuth = (0.5 * np.degrees(np.arctan2(S2, S1)) - phi0_deg + 90) % 180 - 90
    S3_abs = np.sqrt(max(S0 ** 2 - L ** 2, 0.0))
    chi = 0.5 * np.degrees(np.arcsin(min(S3_abs / S0, 1.0)))
    ellip = np.tan(np.radians(chi))
    extinction = (1 - p_lin) / (1 + p_lin)

    # uncertainties from fit residuals (standard least-squares covariance)
    dof = max(n - 3, 1)
    s2 = np.sum(resid ** 2) / dof
    cov = s2 * np.linalg.inv(M.T @ M)
    dS = np.sqrt(np.diag(cov))
    d_az = np.degrees(0.5 * np.hypot(S1 * dS[2], S2 * dS[1]) / max(L ** 2, 1e-300))
    d_p = np.hypot(dS[1] * S1 / L, dS[2] * S2 / L) / S0 if L > 0 else np.nan

    res = dict(excluded_angles_deg=excluded, file=file_path, centre_xy=[float(cx), float(cy)], roi_radius_px=float(roi_radius),
               roi_pixels=int(roi.sum()), angles_deg=angles_deg.tolist(), roi_sums=I.tolist(),
               background_per_px=bg.tolist(), phi0_deg=float(phi0_deg),
               S0=float(S0), S1=float(S1), S2=float(S2),
               azimuth_deg=float(azimuth), azimuth_err_deg=float(d_az),
               p_lin=float(p_lin), p_lin_err=float(d_p), extinction=float(extinction),
               S3_abs_if_fully_polarised=float(S3_abs), chi_deg_if_fully_polarised=float(chi),
               ellipticity_if_fully_polarised=float(ellip),
               fit_rms_residual_frac=float(np.sqrt(s2) / (S0 / 2)))

    print(f"Spot at (x, y) = ({cx:.1f}, {cy:.1f}), ROI r = {roi_radius:.0f} px "
          f"({roi.sum()} px), background {np.mean(bg):.2f} DN/px")
    print(f"Azimuth            {azimuth:7.2f} +/- {d_az:.2f} deg"
          + (f"  (relative to channel A; phi0 = {phi0_deg:.2f} deg)" if phi0_deg else "  (analyser frame)"))
    print(f"Linear DOP p_lin    {p_lin:7.4f} +/- {d_p:.4f}   extinction Imin/Imax = {extinction:.4f}")
    print(f"If fully polarised: |chi| = {chi:.2f} deg, b/a = {ellip:.4f}  "
          f"(upper bound: includes depolarisation)")
    print(f"Fit residual        {res['fit_rms_residual_frac'] * 100:.2f} % of mean")
    if res["fit_rms_residual_frac"] > max_resid:
        warnings_.append(f"fit residual {res['fit_rms_residual_frac'] * 100:.1f} % > "
                         f"{max_resid * 100:g} %: data do not follow Malus - check roi_sums, "
                         f"saturation, early stop, or try a larger roi_radius")
    for w_ in warnings_:
        print("  WARNING: " + w_)
    res["warnings"] = warnings_

    if save or show:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import patches
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
        h = int(roi_radius * 1.6)
        y0, x0 = int(round(cy)), int(round(cx))
        sl = (slice(max(y0 - h, 0), y0 + h + 1), slice(max(x0 - h, 0), x0 + h + 1))
        ax[0].imshow(sub.mean(axis=0)[sl], cmap="gray")
        ax[0].add_patch(patches.Circle((cx - sl[1].start, cy - sl[0].start), roi_radius,
                                       fill=False, ec="r"))
        ax[0].set_title("Mean over analyser angles + ROI")
        ax[0].axis("off")
        tf = np.linspace(theta.min(), theta.max(), 300)
        fit = 0.5 * (S0 + S1 * np.cos(2 * tf) + S2 * np.sin(2 * tf))
        ax[1].plot(angles_deg, I, "o", label="ROI sum (bg-corrected)")
        ax[1].plot(np.degrees(tf), fit, "-", label="Malus fit")
        ax[1].set_xlabel("Analyser angle (deg)")
        ax[1].set_ylabel("Counts")
        ax[1].legend()
        ax[1].set_title(f"az {azimuth:.1f}°, p_lin {p_lin:.4f}, ext {extinction:.4f}")
        a = 1.0
        b = ellip
        ax[2].add_patch(patches.Ellipse((0, 0), 2 * a, 2 * b, angle=azimuth, fill=False, lw=2))
        ax[2].set_xlim(-1.2, 1.2)
        ax[2].set_ylim(-1.2, 1.2)
        ax[2].set_aspect("equal")
        ax[2].axhline(0, color="0.8", lw=0.5)
        ax[2].axvline(0, color="0.8", lw=0.5)
        ax[2].set_title(f"Ellipse (assumes DOP=1): b/a {ellip:.3f}")
        fig.tight_layout()
        if save:
            stem = os.path.splitext(file_path)[0]
            fig.savefig(stem + "__single_spot_pol.png", dpi=150)
            with open(stem + "__single_spot_pol.json", "w") as f:
                json.dump(res, f, indent=2)
        if show:
            plt.show()
        plt.close(fig)
    return res


if __name__ == "__main__":
    single_spot_pol_analysis(show=True)

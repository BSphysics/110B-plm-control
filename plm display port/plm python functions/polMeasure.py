# -*- coding: utf-8 -*-
"""
polMeasure.py - polarisation measurements with the QWP + polariser analyser.

Analyser: light -> QWP (ELL14 on COM6, addr 0, s/n 11402101)
                -> linear polariser (ELL14 on COM5, addr 1, s/n 11400476) -> camera.

Two measurements. NEITHER homes the stages:

pol_measure(...)     ROTATING LINEAR ANALYSER - drop-in for the old function (same call and
                     return).  The QWP turns with the polariser, parallel to it (Q = P + theta0),
                     so it has no effect.  Saves the same mean-image stack as before plus a
                     _meta.json (angles, phi0); singleSpotPolAnalysis reads it, so azimuths
                     come out relative to the channel-A axis.

stokes_measure(...)  FULL STOKES. Polariser at 4 fixed angles, QWP stepped through 180 deg.
                     Fits theta0 each time, so a switch of the polariser's zero is detected and
                     corrected in analyser_cal.json, then per-pixel S0..S3, DOP, azimuth
                     (relative to channel A) and ellipticity maps.

Homing: the polariser ELL14 can home to either of two zeros ~10 deg apart.  Home ONLY at the
start of a session, via the "QWP calibration" button (homes both stages and calibrates on
channel A).  Azimuths are referenced through the QWP stage (psiQ_A in analyser_cal.json),
which has been stable, so they survive a polariser zero switch.
"""
import os
import json
import time
import shutil
from datetime import datetime

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from pypylon import pylon

from elliptec import ELL14

CAL_FILE = "analyser_cal.json"
POL = dict(port="COM5", address="1", serial="11400476")
QWP = dict(port="COM6", address="0", serial="11402101")


# ======================================================================= helpers
def _folder_tag(global_amplitudes, beam_phases, beamAPhaseShift, config_path):
    """Short, Windows-safe description of the beam settings for the folder name."""
    if config_path is not None:
        tag = 'cfg ' + os.path.splitext(os.path.basename(config_path))[0][-30:]
    else:
        tag = 'A{:.2f} B{:.2f}'.format(float(global_amplitudes[0]), float(global_amplitudes[1]))
        if beam_phases is not None:
            tag += ' phA{:.1f} phB{:.1f}'.format(float(beam_phases[0]), float(beam_phases[1]))
    if beamAPhaseShift is not None:
        tag += ' shiftA{:.2f}'.format(float(beamAPhaseShift))
    for ch in '<>:"/\\|?*':
        tag = tag.replace(ch, '-')
    return tag


def load_cal(path=CAL_FILE):
    if not os.path.exists(path):
        raise RuntimeError(f"{path} not found - run 'QWP calibration' (channel A) first")
    with open(path) as f:
        cal = json.load(f)
    for k in ("theta0_deg", "delta_deg", "phi_A_deg"):
        if k not in cal:
            raise RuntimeError(f"{path} has no {k} - run 'QWP calibration' (channel A) first")
    cal.setdefault("psiQ_A_deg", (cal["phi_A_deg"] + cal["theta0_deg"]) % 180)
    return cal


def _open_mounts():
    pol = ELL14(port=POL["port"], address=POL["address"])
    qwp = ELL14(port=QWP["port"], address=QWP["address"])
    for d, ref in ((pol, POL), (qwp, QWP)):
        if d.info["serial"] != ref["serial"]:
            raise RuntimeError(f"{d.port} addr {d.addr} is serial {d.info['serial']}, "
                               f"expected {ref['serial']} - ports swapped?")
    return pol, qwp


def _camera_sw_trigger(camera, exposure_time):
    if camera.IsGrabbing():
        camera.StopGrabbing()
    camera.TriggerSelector.SetValue('FrameStart')
    camera.TriggerMode.SetValue('On')
    camera.TriggerSource.SetValue('Software')
    if exposure_time:
        camera.ExposureTimeAbs.SetValue(float(exposure_time))


def _camera_free_run(camera, exposure_time):
    if camera.IsGrabbing():
        camera.StopGrabbing()
    if exposure_time:
        camera.ExposureTimeAbs.SetValue(float(exposure_time))
    camera.TriggerMode.SetValue("Off")
    camera.OffsetX.SetValue(0)
    camera.OffsetY.SetValue(0)
    camera.Width.SetValue(512)
    camera.Height.SetValue(512)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


def _grab_mean(camera, n):
    acc = None
    camera.StartGrabbingMax(n)
    for _ in range(n):
        camera.TriggerSoftware.Execute()
        r = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        try:
            if not r.GrabSucceeded():
                raise RuntimeError("camera grab failed")
            a = r.Array.astype(np.float64)
        finally:
            r.Release()
        acc = a if acc is None else acc + a
    if camera.IsGrabbing():
        camera.StopGrabbing()
    return (acc / n).astype(np.float32)


def _border_bg(img, w=10):
    return float(np.median(np.concatenate([img[:w].ravel(), img[-w:].ravel(),
                                           img[:, :w].ravel(), img[:, -w:].ravel()])))


def _output_folder(global_amplitudes, beam_phases, beamAPhaseShift, config_path, suffix=""):
    now = datetime.now()
    folder = os.path.join(os.getcwd(), 'Data', now.strftime("%Y_%m_%d"),
                          now.strftime("%H_%M_%S") + ' '
                          + _folder_tag(global_amplitudes, beam_phases, beamAPhaseShift, config_path)
                          + suffix)
    os.makedirs(folder, exist_ok=True)
    if config_path is not None:
        if os.path.exists(config_path):
            shutil.copy2(config_path, os.path.join(folder, os.path.basename(config_path)))
        else:
            print("Warning: config file path does not exist.")
    return folder


def bisector_angle(cal):
    """Polariser stage angle on the bisector of channels A and B (45 deg to A if B unknown)."""
    phiA = (cal["psiQ_A_deg"] - cal["theta0_deg"]) % 180
    if "psiQ_B_deg" in cal:
        phiB = (cal["psiQ_B_deg"] - cal["theta0_deg"]) % 180
        return (phiA + ((phiB - phiA) % 180) / 2) % 180
    return (phiA + 45) % 180


def _park(pol, qwp, cal):
    """Leave the analyser as a plain polariser on the channel bisector (QWP parallel)."""
    P = bisector_angle(cal)
    pol.move_abs(P)
    qwp.move_abs(P + cal["theta0_deg"])
    return P


# ===================================================== rotating linear analyser
def pol_measure(camera, global_amplitudes, beamName, exposure_time, *, beamAPhaseShift=None,
                multiBeamFilePath=None, poincareBeamFilePath=None, beam_phases=None,
                angles_deg=None, images_per_angle=50, cal_path=CAL_FILE):
    """Rotating linear analyser, QWP co-rotating parallel.  Returns (npy filename, folder).
    The stack is in polariser-stage angles; the _meta.json records phi0 (channel A's azimuth
    in that frame) so singleSpotPolAnalysis reports azimuths relative to channel A."""
    cal = load_cal(cal_path)
    theta0 = cal["theta0_deg"]
    phi0 = (cal["psiQ_A_deg"] - theta0) % 180
    if angles_deg is None:
        angles_deg = np.arange(0, 181, 20)
    angles_deg = np.asarray(angles_deg, dtype=float)
    config_path = poincareBeamFilePath or multiBeamFilePath

    pol = qwp = None
    exp0 = camera.ExposureTimeAbs.GetValue()
    try:
        pol, qwp = _open_mounts()
        _camera_sw_trigger(camera, exposure_time)
        H, W = camera.Height.Value, camera.Width.Value
        stack = np.zeros((len(angles_deg), H, W), np.float32)
        P_rb, Q_rb = [], []
        t0 = time.time()
        for i, P in enumerate(angles_deg):
            P_rb.append(pol.move_abs(P))
            Q_rb.append(qwp.move_abs(P + theta0))
            stack[i] = _grab_mean(camera, images_per_angle)
        print(f"Linear analyser scan: {len(angles_deg)} angles in {time.time() - t0:.1f} s "
              f"(QWP parallel to polariser, theta0 = {theta0:.2f} deg)")
        park = _park(pol, qwp, cal)
        print(f"Analyser parked on the channel bisector: polariser {park:.2f} deg, QWP parallel")
    finally:
        for d in (pol, qwp):
            if d is not None:
                d.release()
        _camera_free_run(camera, exp0)

    folder = _output_folder(global_amplitudes, beam_phases, beamAPhaseShift, config_path)
    name = ('Multibeam file__' + os.path.splitext(os.path.basename(multiBeamFilePath))[0][-20:]
            if multiBeamFilePath else 'Singlebeam__')
    filename = os.path.join(folder, name + ".npy")
    np.save(filename, stack)
    meta = dict(kind="linear", angles_deg=angles_deg.tolist(), polariser_readback_deg=P_rb,
                qwp_readback_deg=Q_rb, phi0_deg=phi0, theta0_deg=theta0,
                psiQ_A_deg=cal["psiQ_A_deg"], exposure_us=exposure_time,
                images_per_angle=images_per_angle, beam_name=beamName,
                global_amplitudes=[float(a) for a in global_amplitudes],
                beam_phases=None if beam_phases is None else [float(p) for p in beam_phases],
                date=datetime.now().isoformat(timespec="seconds"))
    with open(os.path.splitext(filename)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print('Saved ' + filename)
    return filename, folder


# ================================================================ full Stokes
def analyser_rows(P, Q, theta0, delta):
    """Vectorised analyser rows (N, 4): I = rows @ S.  All angles in radians."""
    t = Q - theta0
    c, s = np.cos(2 * t), np.sin(2 * t)
    cd, sd = np.cos(delta), np.sin(delta)
    c2P, s2P = np.cos(2 * P), np.sin(2 * P)
    return 0.5 * np.stack([np.ones_like(t),
                           c2P * (c * c + s * s * cd) + s2P * c * s * (1 - cd),
                           c2P * c * s * (1 - cd) + s2P * (s * s + c * c * cd),
                           -c2P * s * sd + s2P * c * sd], axis=-1)


def _state_vec(psi, chi, dop):
    return np.array([1.0, dop * np.cos(2 * chi) * np.cos(2 * psi),
                     dop * np.cos(2 * chi) * np.sin(2 * psi), dop * np.sin(2 * chi)])


def fit_roi(P, Q, I, block, delta, theta0_guess, fit_theta0=True):
    """Fit ROI-integrated data for (theta0), psi, chi, DOP and a scale per block
    (radians).  The fast/slow branch is chosen nearest theta0_guess."""
    nb = int(block.max()) + 1
    sc0 = [2 * I[block == k].max() for k in range(nb)]

    def unpack(p):
        if fit_theta0:
            return p[0], p[1], p[2], p[3], np.asarray(p[4:])
        return theta0_guess, p[0], p[1], p[2], np.asarray(p[3:])

    def resid(p):
        th0, psi, chi, dop, sc = unpack(p)
        return sc[block] * (analyser_rows(P, Q, th0, delta) @ _state_vec(psi, chi, dop)) - I

    lb = ([-np.inf] if fit_theta0 else []) + [-np.inf, -np.pi / 4, 0] + [0] * nb
    ub = ([np.inf] if fit_theta0 else []) + [np.inf, np.pi / 4, 1] + [np.inf] * nb
    best = None
    th_starts = theta0_guess + np.radians([-12, 0, 12]) if fit_theta0 else [None]
    for th in th_starts:
        for psi in np.radians(np.arange(0, 180, 30)):
            for chi in (-0.4, 0.0, 0.4):
                p0 = ([th] if fit_theta0 else []) + [psi, chi, 0.95] + sc0
                r = least_squares(resid, p0, bounds=(lb, ub))
                if best is None or r.cost < best.cost:
                    best = r
    th0, psi, chi, dop, sc = unpack(best.x)
    if fit_theta0:      # (theta0, chi) and (theta0 + 90, -chi) are the same solution
        d = (th0 - theta0_guess + np.pi / 2) % np.pi - np.pi / 2
        if abs(d) > np.pi / 4:
            th0, chi = th0 + np.pi / 2, -chi      # azimuth is unchanged by this equivalence
        th0 = theta0_guess + (th0 - theta0_guess + np.pi / 2) % np.pi - np.pi / 2
    J = best.jac
    cov = np.linalg.pinv(J.T @ J) * 2 * best.cost / max(len(I) - len(best.x), 1)
    err = np.sqrt(np.diag(cov))
    off = 1 if fit_theta0 else 0
    return dict(theta0=float(th0), theta0_err=float(err[0]) if fit_theta0 else 0.0,
                psi=float(psi), psi_err=float(err[off]), chi=float(chi), chi_err=float(err[off + 1]),
                dop=float(dop), dop_err=float(err[off + 2]), scales=np.asarray(sc),
                rms=float(np.sqrt(np.mean(best.fun ** 2)) / np.mean(I)))


def stokes_measure(camera, global_amplitudes, beamName, exposure_time, *, beamAPhaseShift=None,
                   multiBeamFilePath=None, poincareBeamFilePath=None, beam_phases=None,
                   lp_offsets=(0, 45, 90, 135), qwp_step=15.0, n_avg=10, bin_px=2,
                   cal_path=CAL_FILE, frame_tol_deg=1.5, roi_radius=None):
    """Full-Stokes measurement. Returns (summary dict, folder)."""
    cal = load_cal(cal_path)
    delta = np.radians(cal["delta_deg"])
    th0_cal = np.radians(cal["theta0_deg"])
    phiA = (cal["psiQ_A_deg"] - cal["theta0_deg"]) % 180
    config_path = poincareBeamFilePath or multiBeamFilePath
    qwp_angles = np.arange(0, 180, qwp_step)

    pol = qwp = None
    exp0 = camera.ExposureTimeAbs.GetValue()
    rows, imgs = [], []
    try:
        pol, qwp = _open_mounts()
        _camera_sw_trigger(camera, exposure_time)
        t0 = time.time()
        for k, off in enumerate(lp_offsets):
            Pk = pol.move_abs(phiA + off)
            for q in (qwp_angles if k % 2 == 0 else qwp_angles[::-1]):
                Qk = qwp.move_abs(q)
                imgs.append(_grab_mean(camera, n_avg))
                rows.append((k, Pk, Qk))
        print(f"Stokes scan: {len(rows)} settings in {time.time() - t0:.0f} s")
        _park(pol, qwp, cal)
    finally:
        for d in (pol, qwp):
            if d is not None:
                d.release()
        _camera_free_run(camera, exp0)

    stack = np.stack(imgs)
    data = np.array(rows)
    block = data[:, 0].astype(int)
    P, Q = np.radians(data[:, 1]), np.radians(data[:, 2])
    full = 255.0 if stack.max() <= 255 else 4095.0
    nsat = int(np.sum(stack.reshape(len(stack), -1).max(axis=1) >= 0.95 * full))
    if nsat:
        print(f"WARNING: {nsat} of {len(stack)} frames near saturation - reduce exposure")

    sub = stack - np.array([_border_bg(f) for f in stack])[:, None, None]
    ref = gaussian_filter(sub.mean(axis=0), 1.5)
    cy, cx = np.unravel_index(np.argmax(ref), ref.shape)
    yy, xx = np.indices(ref.shape)
    r = np.hypot(yy - cy, xx - cx)
    if roi_radius is None:
        roi_radius = float(r[(ref > 0.05 * ref[cy, cx]) & (r < 60)].max()) + 3
    roi = r <= roi_radius
    I = sub[:, roi].sum(axis=1)

    # 1. brightest spot, theta0 free: detects (and corrects) a switch of the polariser's zero
    f = fit_roi(P, Q, I, block, delta, th0_cal, fit_theta0=True)
    frame_note = ""
    if f["dop"] < 0.5:
        f = fit_roi(P, Q, I, block, delta, th0_cal, fit_theta0=False)
        frame_note = " (DOP < 0.5: theta0 held at calibration)"
    d_th = float(np.degrees((f["theta0"] - th0_cal + np.pi / 2) % np.pi - np.pi / 2))
    if abs(d_th) > frame_tol_deg:
        print(f"WARNING: theta0 moved {d_th:+.2f} deg since calibration - the polariser's zero has "
              f"switched. Results use the new value; updating {cal_path}.")
        cal["theta0_deg"] = float(np.degrees(f["theta0"]) % 180)
        cal["phi_A_deg"] = float((cal["psiQ_A_deg"] - cal["theta0_deg"]) % 180)
        cal["frame_resync_date"] = datetime.now().isoformat(timespec="seconds")
        with open(cal_path, "w") as fh:
            json.dump(cal, fh, indent=2)

    # 2. per-pixel Stokes with the fitted theta0, frames normalised for laser drift
    scale = f["scales"] / f["scales"].mean()
    A = analyser_rows(P, Q, f["theta0"], delta)
    img = sub / scale[block][:, None, None]
    if bin_px > 1:
        n, H, W = img.shape
        H2, W2 = H // bin_px, W // bin_px
        img = img[:, :H2 * bin_px, :W2 * bin_px].reshape(n, H2, bin_px, W2, bin_px).sum(axis=(2, 4))
    S = np.linalg.lstsq(A, img.reshape(len(img), -1), rcond=None)[0].reshape((4,) + img.shape[1:])
    S0 = S[0]
    mask = gaussian_filter(S0, 1) > 0.05 * gaussian_filter(S0, 1).max()
    Lpol = np.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2)
    ref_az = np.degrees(f["theta0"]) - cal["psiQ_A_deg"]          # stage frame -> channel-A frame
    az_map = (0.5 * np.degrees(np.arctan2(S[2], S[1])) + ref_az + 90) % 180 - 90
    chi_map = 0.5 * np.degrees(np.arcsin(np.clip(S[3] / np.maximum(Lpol, 1e-12), -1, 1)))
    dop_map = Lpol / np.maximum(S0, 1e-12)

    az_rel = float((np.degrees(f["psi"]) + ref_az + 90) % 180 - 90)
    summary = dict(azimuth_rel_A_deg=az_rel, azimuth_err_deg=float(np.degrees(f["psi_err"])),
                   chi_deg=float(np.degrees(f["chi"])), chi_err_deg=float(np.degrees(f["chi_err"])),
                   ellipticity=float(np.tan(f["chi"])), dop=f["dop"], dop_err=f["dop_err"],
                   theta0_fit_deg=float(np.degrees(f["theta0"]) % 180), theta0_change_deg=d_th,
                   rms_resid_frac=f["rms"], roi_centre_xy=[int(cx), int(cy)],
                   roi_radius_px=float(roi_radius), n_settings=int(len(I)), saturated_frames=nsat,
                   bin_px=bin_px, fast_axis_offset_deg=cal.get("fast_axis_offset_deg", 0.0),
                   date=datetime.now().isoformat(timespec="seconds"), beam_name=beamName,
                   global_amplitudes=[float(a) for a in global_amplitudes],
                   beam_phases=None if beam_phases is None else [float(p) for p in beam_phases])
    print(f"Brightest spot: azimuth {az_rel:+.2f} +/- {summary['azimuth_err_deg']:.2f} deg (rel. channel A), "
          f"chi {summary['chi_deg']:+.2f} +/- {summary['chi_err_deg']:.2f} deg "
          f"(b/a {abs(summary['ellipticity']):.3f}), DOP {f['dop']:.3f} +/- {f['dop_err']:.3f}, "
          f"rms {100 * f['rms']:.1f}%{frame_note}")
    if f["rms"] > 0.05:
        print("WARNING: fit residual > 5% - check saturation / signal level")

    folder = _output_folder(global_amplitudes, beam_phases, beamAPhaseShift, config_path, " Stokes")
    np.save(os.path.join(folder, "stokes_raw_stack.npy"), stack)
    np.savez_compressed(os.path.join(folder, "stokes_maps.npz"), S=S, mask=mask, azimuth_rel_A=az_map,
                        chi=chi_map, dop=dop_map, P_deg=np.degrees(P), Q_deg=np.degrees(Q), block=block)
    with open(os.path.join(folder, "stokes_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    _stokes_figure(folder, S0, az_map, chi_map, dop_map, mask, summary)
    print('Saved ' + folder)
    return summary, folder


def _stokes_figure(folder, S0, az, chi, dop, mask, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return
    pad = 4
    sl = (slice(max(ys.min() - pad, 0), ys.max() + pad + 1),
          slice(max(xs.min() - pad, 0), xs.max() + pad + 1))
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    panels = [(S0, "S0", "gray", None),
              (np.where(mask, az, np.nan), "Azimuth rel. channel A (deg)", "twilight", (-90, 90)),
              (np.where(mask, chi, np.nan), "Ellipticity angle chi (deg)", "RdBu_r", (-45, 45)),
              (np.where(mask, dop, np.nan), "DOP", "viridis", (0, 1.1))]
    for a, (d, t, cm, lim) in zip(ax, panels):
        h = a.imshow(d[sl], cmap=cm, vmin=None if lim is None else lim[0],
                     vmax=None if lim is None else lim[1])
        a.set_title(t, fontsize=10)
        a.axis("off")
        fig.colorbar(h, ax=a, fraction=0.046)
    fig.suptitle(f"Spot: az {summ['azimuth_rel_A_deg']:+.1f} deg, chi {summ['chi_deg']:+.2f} deg, "
                 f"DOP {summ['dop']:.3f}  (sign of chi set by fast-axis assignment)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "stokes_maps.png"), dpi=130)
    plt.close(fig)

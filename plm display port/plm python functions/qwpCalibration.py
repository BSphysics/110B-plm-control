"""
qwpCalibration.py - find the QWP's axis orientation (in ELL14 stage degrees) and its
true retardance, using channel A as a known linear input.

Everything is expressed in the LINEAR POLARISER'S STAGE FRAME (stage reading after
homing).  Model for each measurement:
    I = s_k * a(phi_P, theta_Q - theta0, delta) . [1, cos 2phi_A, sin 2phi_A, 0]
where a() is the polariser-after-retarder analyser row (same as in fit_stokes),
phi_P is the polariser stage angle, theta_Q the QWP stage angle, s_k a scale per
polariser setting (absorbs slow laser drift between blocks).
Fitted: theta0 (QWP stage angle at which its axis lies along the polariser stage's 0),
delta (retardance), and channel A's full state: azimuth phi_A, ellipticity angle chi_A
and degree of polarisation.  Channel A is NOT assumed linear (on this rig it isn't).

Scan: polariser at phi_A + {0, 45, 90, 135} deg, QWP stepped through 180 deg at each.
Crossed (phi_A + 90) gives the sharpest determination of theta0; 45/135 pin down delta.

Limitation: (theta0, chi_A) and (theta0 + 90, -chi_A) fit identically, so the fast and
slow axes - and hence the sign of S3 - cannot be told apart from the data alone.
Resolve this once from the axis engraving on the mount and set `fast_axis_offset_deg`
to 0 or 90 accordingly (90 also flips the sign of chi_A).

GUI use:
    # 1. calibrate (channel A only on the PLM, B amplitude 0, bitpacked)
    qwp_calibration(self, self.camera, phi_A_guess=93.9)
    # 2. channel B's state with the QWP held at that calibration (A amplitude 0, bitpacked)
    qwp_calibration(self, self.camera, phi_A_guess=0.4, channel="B")
"""
import os
import json
import time
from datetime import datetime

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from pypylon import pylon

from elliptec import ELL14

POL_SERIAL = "11400476"
QWP_SERIAL = "11402101"


# ------------------------------------------------------------------ model
def analyser_row(phi, theta, delta):
    """Row r with I = r . S for a linear polariser at phi after a retarder with fast
    axis theta and retardance delta (radians, all in the same frame)."""
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    cd, sd = np.cos(delta), np.sin(delta)
    M = np.array([[1, 0, 0, 0],
                  [0, c * c + s * s * cd, c * s * (1 - cd), -s * sd],
                  [0, c * s * (1 - cd), s * s + c * c * cd, c * sd],
                  [0, s * sd, -c * sd, cd]])
    lp = 0.5 * np.array([1, np.cos(2 * phi), np.sin(2 * phi), 0])
    return lp @ M


def model(params, phiP, thQ, block, n_blocks):
    th0, delta, phiA, chi, dop = params[:5]
    scales = params[5:5 + n_blocks]
    S = np.array([1, dop * np.cos(2 * chi) * np.cos(2 * phiA),
                  dop * np.cos(2 * chi) * np.sin(2 * phiA), dop * np.sin(2 * chi)])
    rows = np.array([analyser_row(p, t - th0, delta) for p, t in zip(phiP, thQ)])
    return scales[block] * (rows @ S)


def fit_calibration(phiP_deg, thQ_deg, I, block, phi_A_guess_deg):
    phiP, thQ = np.radians(phiP_deg), np.radians(thQ_deg)
    nb = int(block.max()) + 1
    scale0 = [2 * I[block == k].max() for k in range(nb)]
    best = None
    lb = [-np.inf, 0, -np.inf, -np.pi / 4, 0] + [0] * nb
    ub = [np.inf, np.pi, np.inf, np.pi / 4, 1] + [np.inf] * nb
    for th0g in np.radians(np.arange(0, 180, 7.5)):            # coarse start grid
        for chig in (0.05, -0.05):
            p0 = [th0g, np.pi / 2, np.radians(phi_A_guess_deg), chig, 0.99] + scale0
            r = least_squares(lambda p: model(p, phiP, thQ, block, nb) - I, p0, bounds=(lb, ub),
                              x_scale=[0.1, 0.1, 0.1, 0.05, 0.01] + [max(scale0)] * nb)
            if best is None or r.cost < best.cost:
                best = r
    p = best.x
    J = best.jac
    dof = max(len(I) - len(p), 1)
    cov = np.linalg.pinv(J.T @ J) * 2 * best.cost / dof
    err = np.sqrt(np.diag(cov))
    th0, delta, phiA, chi, dop = p[:5]
    return dict(chi_A_deg=float(np.degrees(chi)), chi_A_err_deg=float(np.degrees(err[3])),
                ellipticity_A=float(np.tan(chi)), dop_A=float(dop), dop_A_err=float(err[4]),theta0_deg=float(np.degrees(th0) % 180), theta0_err_deg=float(np.degrees(err[0])),
                delta_deg=float(np.degrees(delta)), delta_err_deg=float(np.degrees(err[1])),
                phi_A_deg=float(np.degrees(phiA) % 180), phi_A_err_deg=float(np.degrees(err[2])),
                scales=p[5:].tolist(),
                rms_resid_frac=float(np.sqrt(np.mean(best.fun ** 2)) / np.mean(I)),
                fitted=model(p, phiP, thQ, block, nb).tolist())


def fit_state(phiP_deg, thQ_deg, I, block, theta0_deg, delta_deg, phi_guess_deg):
    """Fit only the input state (azimuth, ellipticity, DOP) with the QWP's theta0 and
    delta held fixed from a previous calibration.  Using the same theta0 for every
    channel keeps the SIGN of chi consistent between channels."""
    phiP, thQ = np.radians(phiP_deg), np.radians(thQ_deg)
    nb = int(block.max()) + 1
    th0, dl = np.radians(theta0_deg), np.radians(delta_deg)
    scale0 = [2 * I[block == k].max() for k in range(nb)]

    def f(q):
        return model(np.r_[th0, dl, q], phiP, thQ, block, nb) - I
    lb = [-np.inf, -np.pi / 4, 0] + [0] * nb
    ub = [np.inf, np.pi / 4, 1] + [np.inf] * nb
    best = None
    for chig in (0.05, -0.05):
        r = least_squares(f, [np.radians(phi_guess_deg), chig, 0.99] + scale0, bounds=(lb, ub),
                          x_scale=[0.1, 0.05, 0.01] + [max(scale0)] * nb)
        if best is None or r.cost < best.cost:
            best = r
    q = best.x
    cov = np.linalg.pinv(best.jac.T @ best.jac) * 2 * best.cost / max(len(I) - len(q), 1)
    err = np.sqrt(np.diag(cov))
    return dict(theta0_deg=float(theta0_deg), theta0_err_deg=0.0, delta_deg=float(delta_deg),
                delta_err_deg=0.0, phi_A_deg=float(np.degrees(q[0]) % 180),
                phi_A_err_deg=float(np.degrees(err[0])), chi_A_deg=float(np.degrees(q[1])),
                chi_A_err_deg=float(np.degrees(err[1])), ellipticity_A=float(np.tan(q[1])),
                dop_A=float(q[2]), dop_A_err=float(err[2]), scales=q[3:].tolist(),
                rms_resid_frac=float(np.sqrt(np.mean(best.fun ** 2)) / np.mean(I)),
                fitted=(f(q) + I).tolist())


# ------------------------------------------------------------------ acquisition
def _grab_mean(camera, n):
    acc = None
    camera.StartGrabbingMax(n)
    for _ in range(n):
        camera.TriggerSoftware.Execute()
        r = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        try:
            a = r.Array.astype(np.float64)
        finally:
            r.Release()
        acc = a if acc is None else acc + a
    if camera.IsGrabbing():
        camera.StopGrabbing()
    return acc / n


def _border_bg(img, w=10):
    return float(np.median(np.concatenate([img[:w].ravel(), img[-w:].ravel(),
                                           img[:, :w].ravel(), img[:, -w:].ravel()])))


def qwp_calibration(gui, camera, phi_A_guess=93.9, channel="A", fit_qwp=None,
                    lp_offsets=(0, 45, 90, 135),
                    qwp_step=10.0, n_avg=10, exposure_us=None, roi_radius=None,
                    fast_axis_offset_deg=0.0, pol_serial=POL_SERIAL, qwp_serial=QWP_SERIAL,
                    pol_port="COM5", pol_addr="1", qwp_port="COM6", qwp_addr="0",
                    frame_tol_deg=1.5, home=None,
                    out_root=None, save_cal="analyser_cal.json"):
    """channel: which channel is on the PLM ('A' or 'B'); phi_A_guess is that channel's
    approximate azimuth in the polariser stage frame (A ~93.9, B ~0.4).
    fit_qwp: True = fit theta0 and delta too (a calibration); False = hold them at the
    values in analyser_cal.json and fit only the channel's state.  Default: True for A,
    False for B."""
    channel = channel.upper()
    if fit_qwp is None:
        fit_qwp = channel == "A"
    if home is None:          # home only when calibrating (session start); never for state runs
        home = bool(fit_qwp)
    if gui is not None:
        amp_A, amp_B = float(gui.inputs[16].text()), float(gui.inputs[17].text())
        on, off = (amp_A, amp_B) if channel == "A" else (amp_B, amp_A)
        if off != 0 or on == 0:
            print(f"Needs channel {channel} ONLY on the PLM (other amplitude = 0, then "
                  f"Bitpack). Aborting.")
            return None
    fixed = None
    if not fit_qwp:
        if not (save_cal and os.path.exists(save_cal)):
            print(f"fit_qwp=False needs theta0/delta from {save_cal}; run channel A first.")
            return None
        with open(save_cal) as f:
            c = json.load(f)
        fixed = (c["theta0_deg"], c["delta_deg"])
        print(f"Using QWP calibration from {save_cal}: theta0 {fixed[0]:.2f}, delta {fixed[1]:.2f} deg")

    pol = qwp = None
    exp0 = camera.ExposureTimeAbs.GetValue()
    try:
        if pol_port:          # direct (fast); set pol_port=None to search by serial (slow)
            pol, qwp = ELL14(port=pol_port, address=pol_addr), ELL14(port=qwp_port, address=qwp_addr)
            for d_, sn in ((pol, pol_serial), (qwp, qwp_serial)):
                if d_.info["serial"] != sn:
                    raise RuntimeError(f"{d_.port} addr {d_.addr} is serial {d_.info['serial']}, "
                                       f"expected {sn} - ports swapped?")
        else:
            pol, qwp = ELL14(serial_no=pol_serial), ELL14(serial_no=qwp_serial)
        print(f"Polariser: {pol}\nQWP:       {qwp}")
        if home:      # the polariser ELL14 can home to either of two zeros ~10 deg apart, so
            for d in (pol, qwp):              # home ONLY here (session start); measurements never home
                d.home()
                for a in (30, 60, 0):          # let the position-error compensation settle
                    d.move_abs(a)

        if camera.IsGrabbing():
            camera.StopGrabbing()
        camera.TriggerSelector.SetValue("FrameStart")
        camera.TriggerMode.SetValue("On")
        camera.TriggerSource.SetValue("Software")
        if exposure_us:
            camera.ExposureTimeAbs.SetValue(float(exposure_us))

        # ROI from the brightest configuration: polariser along channel A, QWP parallel-ish
        pol.move_abs(phi_A_guess)
        img = _grab_mean(camera, n_avg)
        full = 255.0 if img.max() <= 255 else 4095.0
        if img.max() >= 0.95 * full:
            print(f"WARNING: peak {img.max():.0f}/{full:.0f} at maximum transmission - "
                  f"reduce exposure and re-run")
        sub = gaussian_filter(img - _border_bg(img), 1.5)
        cy, cx = np.unravel_index(np.argmax(sub), sub.shape)
        yy, xx = np.indices(sub.shape)
        r = np.hypot(yy - cy, xx - cx)
        if roi_radius is None:
            roi_radius = float(r[(sub > 0.05 * sub[cy, cx]) & (r < 60)].max()) + 3
        roi = r <= roi_radius
        print(f"Spot at ({cx}, {cy}), ROI radius {roi_radius:.1f} px")

        rows = []
        qwp_angles = np.arange(0, 180 + 1e-9, qwp_step)
        t0 = time.time()
        for k, off in enumerate(lp_offsets):
            phiP = pol.move_abs(phi_A_guess + off)
            for th in (qwp_angles if k % 2 == 0 else qwp_angles[::-1]):   # serpentine
                thQ = qwp.move_abs(th)
                img = _grab_mean(camera, n_avg)
                I = float(np.sum((img - _border_bg(img))[roi]))
                rows.append((k, phiP, thQ, I, img.max()))
            print(f"  polariser {phiP:7.2f} deg done ({time.time() - t0:.0f} s)")
        data = np.array(rows)
        block = data[:, 0].astype(int)
        phiP, thQ, I, pk = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        if pk.max() >= 0.95 * full:
            print(f"WARNING: {np.sum(pk >= 0.95 * full)} points near saturation")

        prev = {}
        if save_cal and os.path.exists(save_cal):
            with open(save_cal) as f:
                prev = json.load(f)
        if fixed is None:
            res = fit_calibration(phiP, thQ, I, block, phi_A_guess)
            res["theta0_deg"] = (res["theta0_deg"] + fast_axis_offset_deg) % 180
            if round(fast_axis_offset_deg) % 180 == 90:
                res["chi_A_deg"], res["ellipticity_A"] = -res["chi_A_deg"], -res["ellipticity_A"]
            # stay on the same fast/slow branch as the previous calibration, so the sign
            # of chi (and theta0) is comparable from run to run
            if "theta0_deg" in prev:
                d = (res["theta0_deg"] - prev["theta0_deg"] + 90) % 180 - 90
                if abs(d) > 45:
                    res["theta0_deg"] = (res["theta0_deg"] + 90) % 180
                    res["chi_A_deg"], res["ellipticity_A"] = -res["chi_A_deg"], -res["ellipticity_A"]
        else:
            # theta0 fitted (delta held): copes with a switch of the polariser's zero.
            from polMeasure import fit_roi
            fr = fit_roi(np.radians(phiP), np.radians(thQ), I, block, np.radians(fixed[1]),
                         np.radians(fixed[0]), fit_theta0=True)
            if fr["dop"] < 0.5:
                fr = fit_roi(np.radians(phiP), np.radians(thQ), I, block, np.radians(fixed[1]),
                             np.radians(fixed[0]), fit_theta0=False)
            model_I = None
            res = dict(theta0_deg=float(np.degrees(fr["theta0"]) % 180),
                       theta0_err_deg=float(np.degrees(fr["theta0_err"])),
                       delta_deg=float(fixed[1]), delta_err_deg=0.0,
                       phi_A_deg=float(np.degrees(fr["psi"]) % 180),
                       phi_A_err_deg=float(np.degrees(fr["psi_err"])),
                       chi_A_deg=float(np.degrees(fr["chi"])), chi_A_err_deg=float(np.degrees(fr["chi_err"])),
                       ellipticity_A=float(np.tan(fr["chi"])), dop_A=fr["dop"], dop_A_err=fr["dop_err"],
                       scales=fr["scales"].tolist(), rms_resid_frac=fr["rms"])
            from polMeasure import analyser_rows, _state_vec
            model_I = fr["scales"][block] * (analyser_rows(np.radians(phiP), np.radians(thQ), fr["theta0"],
                                                           np.radians(fixed[1]))
                                             @ _state_vec(fr["psi"], fr["chi"], fr["dop"]))
            res["fitted"] = model_I.tolist()
        # checks against the previous calibration:
        #  theta0 change  -> the polariser's zero switched (expected occasionally after homing)
        #  psiQ change    -> the beam's polarisation itself changed (azimuth vs the QWP stage)
        res["frame_change_deg"] = {}
        if "theta0_deg" in prev:
            dth = (res["theta0_deg"] - prev["theta0_deg"] + 90) % 180 - 90
            res["frame_change_deg"]["theta0"] = float(dth)
            if abs(dth) > frame_tol_deg:
                print(f"NOTE: polariser zero has switched (theta0 {prev['theta0_deg']:.2f} -> "
                      f"{res['theta0_deg']:.2f}, {dth:+.2f} deg). Handled: angles are referenced "
                      f"to the QWP stage.")
        psiQ = (res["phi_A_deg"] + res["theta0_deg"]) % 180
        pk = f"psiQ_{channel}_deg"
        if pk in prev:
            dps = (psiQ - prev[pk] + 90) % 180 - 90
            res["frame_change_deg"][pk] = float(dps)
            if abs(dps) > frame_tol_deg:
                print(f"WARNING: channel {channel}'s azimuth (vs QWP stage) changed by {dps:+.2f} deg "
                      f"since last time - the beam's polarisation has changed.")
        res.update(channel=channel, qwp_fitted=bool(fit_qwp),
                   date=datetime.now().isoformat(timespec="seconds"),
                   pol_serial=pol_serial, qwp_serial=qwp_serial,
                   fast_axis_offset_deg=fast_axis_offset_deg, roi_radius_px=roi_radius,
                   n_points=len(I))
        ch = channel
        if fixed is None:
            print(f"\nQWP axis at stage angle theta0 = {res['theta0_deg']:.2f} +/- "
                  f"{res['theta0_err_deg']:.2f} deg  (branch kept consistent with previous calibration)")
            print(f"Retardance delta = {res['delta_deg']:.2f} +/- {res['delta_err_deg']:.2f} deg "
                  f"(expect ~90.9 at 801 nm)")
        else:
            print(f"\nQWP: delta held at {fixed[1]:.2f} deg; theta0 fitted = {res['theta0_deg']:.2f} "
                  f"(calibration {fixed[0]:.2f})")
        print(f"Channel {ch} azimuth = {res['phi_A_deg']:.2f} +/- {res['phi_A_err_deg']:.2f} deg "
              f"(polariser stage); vs QWP stage psiQ_{ch} = "
              f"{(res['phi_A_deg'] + res['theta0_deg']) % 180:.2f} deg")
        print(f"Channel {ch} ellipticity: chi = {res['chi_A_deg']:.2f} +/- {res['chi_A_err_deg']:.2f} deg "
              f"(b/a = {abs(res['ellipticity_A']):.3f}; sign depends on fast-axis assignment), "
              f"DOP = {res['dop_A']:.3f} +/- {res['dop_A_err']:.3f}")
        print(f"Fit rms residual = {100 * res['rms_resid_frac']:.2f} % of mean")
        print(f"QWP 'transparent' park angle for polariser at P: theta = P + {res['theta0_deg']:.2f}")

        now = datetime.now()
        folder = os.path.join(out_root or os.path.join(os.getcwd(), "Data", now.strftime("%Y_%m_%d")),
                              now.strftime("%H_%M_%S") + (" QWP calibration" if fixed is None
                                                           else f" channel {channel} state"))
        os.makedirs(folder, exist_ok=True)
        np.savez(os.path.join(folder, "qwp_cal_data.npz"), block=block, phiP=phiP, thQ=thQ, I=I,
                 peak=pk, fitted=np.array(res["fitted"]))
        summary = {k: v for k, v in res.items() if k != "fitted"}
        with open(os.path.join(folder, "qwp_cal.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if save_cal:
            cal = {}
            if os.path.exists(save_cal):
                with open(save_cal) as f:
                    cal = json.load(f)
            if fixed is not None and abs(res["frame_change_deg"].get("theta0", 0)) > frame_tol_deg:
                cal.update(theta0_deg=res["theta0_deg"],
                           phi_A_deg=float((cal.get("psiQ_A_deg", cal["phi_A_deg"] + fixed[0])
                                            - res["theta0_deg"]) % 180))
            if fixed is None:
                cal.update(theta0_deg=res["theta0_deg"], delta_deg=res["delta_deg"],
                           fast_axis_offset_deg=fast_axis_offset_deg, qwp_cal_date=res["date"],
                           qwp_cal_folder=folder)
            cal.update({f"phi_{channel}_deg": res["phi_A_deg"], f"chi_{channel}_deg": res["chi_A_deg"],
                        f"dop_{channel}": res["dop_A"], f"state_{channel}_date": res["date"],
                        # azimuth in QWP-stage coordinates: survives polariser zero switches
                        f"psiQ_{channel}_deg": (res["phi_A_deg"] + res["theta0_deg"]) % 180})
            with open(save_cal, "w") as f:
                json.dump(cal, f, indent=2)
            print(f"Updated {os.path.abspath(save_cal)}")
        _plot(folder, phiP, thQ, I, block, res, lp_offsets)

        # leave the analyser as a plain polariser at 45 deg to this channel
        pol.move_abs(phi_A_guess + 45)
        qwp.move_abs(phi_A_guess + 45 + res["theta0_deg"])
        return res
    finally:
        for d in (pol, qwp):
            if d is not None:
                d.release()
        camera.ExposureTimeAbs.SetValue(exp0)
        if camera.IsGrabbing():
            camera.StopGrabbing()
        camera.TriggerMode.SetValue("Off")
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


def _plot(folder, phiP, thQ, I, block, res, lp_offsets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    fit = np.array(res["fitted"])
    for k, off in enumerate(lp_offsets):
        m = block == k
        o = np.argsort(thQ[m])
        ax.plot(thQ[m][o], I[m][o], "o", ms=4, label=f"polariser = phi_A + {off} deg")
        ax.plot(thQ[m][o], fit[m][o], "-", color=ax.lines[-1].get_color())
    ax.axvline(res["theta0_deg"] % 180, color="k", ls=":", lw=1)
    ax.set_xlabel("QWP stage angle (deg)")
    ax.set_ylabel("ROI sum (counts)")
    ax.set_title(f"theta0 {res['theta0_deg']:.2f}, delta {res['delta_deg']:.1f}, "
                 f"phi_A {res['phi_A_deg']:.2f}, chi_A {res['chi_A_deg']:.2f} deg "
                 f"(rms {100 * res['rms_resid_frac']:.1f}%)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "qwp_cal.png"), dpi=130)
    plt.close(fig)


def refit_saved(npz_path, phi_A_guess=93.9, fast_axis_offset_deg=0.0):
    """Re-analyse a saved qwp_cal_data.npz with the current model."""
    d = np.load(npz_path)
    res = fit_calibration(d["phiP"], d["thQ"], d["I"], d["block"].astype(int), phi_A_guess)
    res["theta0_deg"] = (res["theta0_deg"] + fast_axis_offset_deg) % 180
    if round(fast_axis_offset_deg) % 180 == 90:
        res["chi_A_deg"], res["ellipticity_A"] = -res["chi_A_deg"], -res["ellipticity_A"]
    for k in ("theta0_deg", "delta_deg", "phi_A_deg", "chi_A_deg", "dop_A", "rms_resid_frac"):
        print(f"{k:16s} {res[k]:.4f}")
    return res

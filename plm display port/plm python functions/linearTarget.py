"""
linearTarget.py - make the A + B superposition a pure LINEAR state at a chosen azimuth
==========================================================================================

Principle
---------
Park the QWP parallel to the polariser (Q = P + theta0) so the analyser is a plain
linear polariser, and set that polariser CROSSED to the target azimuth.  The light
that gets through is then exactly the part of the beam that is NOT the target state:

    eps = I_crossed / (I_crossed + I_parallel)
        = (1 - s.t) / 2  ~=  (1 - DOP)/2 + dpsi^2 + chi^2        (angles in rad)

eps = 0 only if the beam is fully polarised, linear, and at the target azimuth.
overlapOptimiser_v2 already minimises the transmitted power through "the analyser"
by adjusting B's tilt, low-order Zernikes, amplitude ratio and relative phase.  With
the analyser crossed to the target, that minimum IS the most-linear state.

Why every target is reachable, and why 45 deg is the hard one
---------------------------------------------------------------
* Two controls (amplitude ratio rho, relative phase beta) against two conditions
  (real and imaginary parts of the field leaking through the crossed polariser), so
  for perfectly overlapping, mutually coherent beams the null is exact at ANY
  analyser angle.  This does not depend on A and B's ellipticities (~1 deg) or on
  their 3.5 deg non-orthogonality: those change the rho and beta needed, not
  whether a solution exists.  So "45 deg between the inputs" is not chosen because
  the inputs are perfect; it is just a point where rho, beta come out near (1, pi).
* What rho and beta cannot fix is loss of local fringe visibility V (imperfect
  spatial-mode / wavefront overlap, chromatic walk-off, group-delay mismatch).
  For orthogonal inputs and a target at angle theta from A, the balanced null is
      eps(theta) = (1 - V)/2 * sin^2(2 theta)
  so 45 deg is the worst case (equal powers, maximum reliance on interference).
  Every other target inherits the 45 deg floor scaled by sin^2(2 theta).  This
  holds as long as the spatial correction found at 45 deg still applies with
  different amplitudes; target_sweep() measures eps directly, so it will show if
  that assumption fails.

What this module does
---------------------
linearise(gui, plm, camera, target="D")
    1. Analyser crossed to the target (the default "D" is the bisector of A and B).
    2. Runs overlapOptimiser_v2 (spatial overlap, Zernikes, rho, beta) against
       that analyser, and writes the result into the GUI as v2 does.
    3. Measures eps properly: crossed AND parallel, each at its own exposure
       (8-bit camera), with a blank-hologram dark at that same exposure.  This
       replaces v2's I_min/I_max, which equals eps only at 45 deg.
    4. Optional target sweep (see below) and optional full Stokes check of the
       final state with polMeasure.stokes_measure.

target_sweep(...)
    Keeps the spatial correction and re-solves only rho and beta for each target
    azimuth (seconds per target).  Returns, for each target, the GUI amplitudes and
    Beam A phase plus the measured eps.  The GUI is left on the "D" solution;
    apply_target(gui, row) loads any other row.

Targets are given in degrees relative to channel A (the same frame as
stokes_measure's azimuth_rel_A_deg), or as one of these tokens:
    "A" (0), "B" (psiQ_B - psiQ_A), "D" (the A/B bisector, ~43.2 deg today),
    "AD" (D + 90, the other bisector).

Nothing here homes the stages.  The azimuth frame comes from analyser_cal.json
(psiQ_A, psiQ_B, theta0) exactly as in polMeasure.

GUI usage (replaces the old overlap_optimiser call):

    from linearTarget import linear_optimiser
    newBxTilt, newByTilt, zero_phase = linear_optimiser(self, plm, camera)
    # or, with the sweep and a Stokes check:
    linear_optimiser(self, plm, camera, sweep=True, verify_stokes=True)
    # full result: self.last_linear_result
"""
import os
import json
import time
from datetime import datetime

import numpy as np

import polMeasure as pm
import overlapOptimiser_v2 as ov


# ----------------------------------------------------------------------- geometry
def target_rel_A(cal, target):
    """Target azimuth in degrees relative to channel A."""
    dAB = (cal["psiQ_B_deg"] - cal["psiQ_A_deg"]) % 180
    if isinstance(target, str):
        t = target.upper()
        table = {"A": 0.0, "B": dAB, "D": dAB / 2, "AD": (dAB / 2 + 90) % 180}
        if t not in table:
            raise ValueError(f"unknown target {target!r}; use a number or one of {list(table)}")
        return float(table[t])
    return float(target) % 180


def analyser_angles(cal, t_relA, crossed=True):
    """(P, Q) stage angles: polariser crossed to (or parallel with) the target, QWP parallel."""
    psiQ = cal["psiQ_A_deg"] + t_relA                  # target azimuth in the QWP-stage frame
    P = (psiQ + (90.0 if crossed else 0.0) - cal["theta0_deg"]) % 180
    return P, P + cal["theta0_deg"]


class Analyser:
    def __init__(self, cal):
        self.cal = cal
        self.pol, self.qwp = pm._open_mounts()
        self.state = None

    def set(self, t_relA, crossed=True):
        P, Q = analyser_angles(self.cal, t_relA, crossed)
        Pr = self.pol.move_abs(P)
        Qr = self.qwp.move_abs(Q)
        self.state = (t_relA, crossed, Pr, Qr)
        return Pr, Qr

    def release(self):
        for d in (self.pol, self.qwp):
            if d is not None:
                d.release()
        self.pol = self.qwp = None


# ------------------------------------------------------------------ measurements
def roi_rate(opt, holo, max_exposure_us, lo=0.15, hi=0.9, tries=8):
    """Dark-subtracted ROI sum per microsecond of exposure for one hologram.  The
    exposure is adjusted until the ROI peak is between lo and hi of full scale (or
    max_exposure_us is reached), with a blank-hologram dark at the same exposure.
    Returns (rate, rate_noise, exposure_used, peak_fraction)."""
    rig, roi = opt.rig, opt.roi
    t0 = rig.exposure
    blank = opt.holo(kind="blank")
    try:
        for _ in range(tries):
            raw, dark = rig.measure([holo, blank])
            fs = rig.full_scale
            pk = raw[roi].max()
            if pk >= 0.98 * fs:
                rig.exposure = rig.exposure / 4
                continue
            sig = max(pk - np.median(dark[roi]), 0.5)
            if pk < lo * fs and rig.exposure < max_exposure_us:
                rig.exposure = min(max_exposure_us, rig.exposure * min(50.0, 0.6 * fs / sig))
                continue
            break
        t = rig.exposure
        s = float(np.sum((raw - dark)[roi]))
        noise = float(np.std(dark[roi]) * np.sqrt(roi.sum()))
        return s / t, noise / t, t, float(pk / fs)
    finally:
        rig.exposure = t0


def measure_eps(opt, an, holo, t_relA, max_exposure_us):
    """eps = crossed / (crossed + parallel), each at its own exposure."""
    an.set(t_relA, crossed=True)
    perp, perp_n, t_perp, pk_perp = roi_rate(opt, holo, max_exposure_us)
    an.set(t_relA, crossed=False)
    par, par_n, t_par, pk_par = roi_rate(opt, holo, max_exposure_us)
    an.set(t_relA, crossed=True)
    tot = perp + par
    eps = perp / tot
    return dict(eps=float(eps), eps_noise=float(np.hypot(perp_n, eps * par_n) / tot),
                exp_crossed_us=t_perp, exp_parallel_us=t_par,
                peak_crossed=pk_perp, peak_parallel=pk_par,
                angle_bound_deg=float(np.degrees(np.sqrt(max(eps, 0.0)))))


def balance_hdr(opt, max_exposure_us):
    """rho from single-channel powers through the (crossed) analyser, each measured
    at its own exposure, so a weak channel (targets near A or B) is still measured."""
    PA, PAn, _, _ = roi_rate(opt, opt.holo(kind="A"), max_exposure_us)
    PB, PBn, _, _ = roi_rate(opt, opt.holo(kind="B"), max_exposure_us)
    PA_, PB_ = max(PA, PAn), max(PB, PBn)              # noise-limited floor, never 0
    opt.rho *= np.sqrt(PA_ / PB_)
    return PA, PB


def solve_phase(opt, n_iter=2):
    """Four-step through the crossed analyser, centred on the current null estimate."""
    e = None
    for _ in range(n_iter):
        betas = opt.beta_null + opt.BETAS
        imgs, IA, IB = opt.run_batch([(opt.p, b, "both") for b in betas])
        A, B, phi = ov.fourstep([np.sum(i[opt.roi]) for i in imgs], betas)   # absolute phase
        e = (A - B) / (A + B)
        opt.beta_null = (phi + np.pi) % (2 * np.pi)
    return e


def gui_values(hb, rho, beta):
    """GUI fields 16, 17 (amplitudes) and 18 (Beam A global phase, %) for (rho, beta)
    relative to the hologram hb was built from (same convention as v2.write_back)."""
    aA, aB = hb.ampA0, hb.ampB0 * rho
    m = max(aA, aB)
    return float(aA / m), float(aB / m), float((hb.uv[18] - beta / (2 * np.pi) * 100) % 100)


def apply_target(gui, row):
    """Load one sweep row into the GUI fields (press the GUI's update to display it)."""
    gui.inputs[16].setText(f"{row['amp_A']:.4g}")
    gui.inputs[17].setText(f"{row['amp_B']:.4g}")
    gui.inputs[18].setText(f"{row['phase_A_pct']:.2f}")


# -------------------------------------------------------------------- procedures
def _fresh_opt_state(opt, gui):
    opt.hb = ov.HologramBuilder(gui)          # hologram exactly as the GUI now builds it
    opt.p, opt.rho, opt.beta_null = {}, 1.0, np.pi


def solve_target(opt, an, gui_hb, t_relA, max_exposure_us, log):
    """Re-solve rho and beta (spatial correction kept) for one target azimuth."""
    an.set(t_relA, crossed=True)
    opt.p, opt.rho, opt.beta_null = {}, 1.0, np.pi
    PA, PB = balance_hdr(opt, max_exposure_us)
    opt.auto_exposure()
    solve_phase(opt)
    balance_hdr(opt, max_exposure_us)                   # amp_mod_phase is not linear: iterate
    solve_phase(opt)
    m = measure_eps(opt, an, opt.holo(beta=opt.beta_null), t_relA, max_exposure_us)
    aA, aB, ph = gui_values(gui_hb, opt.rho, opt.beta_null)
    row = dict(target_rel_A_deg=t_relA, rho=float(opt.rho), beta_rad=float(opt.beta_null),
               amp_A=aA, amp_B=aB, phase_A_pct=ph,
               PA_through=float(PA), PB_through=float(PB), **m)
    log(f"  target {t_relA:6.2f} deg: eps = {m['eps'] * 100:.3f} % "
        f"(+/- {m['eps_noise'] * 100:.3f}), |dpsi|,|chi| <= {m['angle_bound_deg']:.2f} deg; "
        f"GUI amps A {aA:.3f} B {aB:.3f}, phase A {ph:.2f} %")
    return row


def target_sweep(opt, an, gui, targets, max_exposure_us, log):
    gui_hb = ov.HologramBuilder(gui)
    rows = []
    for t in targets:
        rows.append(solve_target(opt, an, gui_hb, float(t) % 180, max_exposure_us, log))
    return rows


def linearise(gui, plm, camera, target="D", cfg=None, sweep=False,
              sweep_targets=tuple(np.arange(0, 180, 22.5)), verify_stokes=False,
              max_exposure_us=200000.0, cal_path=pm.CAL_FILE):
    cfg = cfg or ov.OverlapConfig(bandwidth_nm=6.6, wavelength_nm=801.0)
    cal = pm.load_cal(cal_path)
    if "psiQ_B_deg" not in cal:
        raise RuntimeError("analyser_cal.json has no psiQ_B_deg - run the channel-B "
                           "QWP calibration first")
    t_relA = target_rel_A(cal, target)
    dAB = (cal["psiQ_B_deg"] - cal["psiQ_A_deg"]) % 180
    lines = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        lines.append(s)
        print(s)

    log(f"Channels: A at 0, B at {dAB:.2f} deg (rel. A).  Target linear azimuth "
        f"{t_relA:.2f} deg rel. A.  Analyser: polariser crossed, QWP parallel (transparent).")
    an = Analyser(cal)
    res = dict(target_rel_A_deg=t_relA, dAB_deg=dAB, cal=cal)
    opt = None
    try:
        P, Q = an.set(t_relA, crossed=True)
        log(f"Analyser at P = {P:.2f}, Q = {Q:.2f} deg (stage read-back)")

        # 1. full v2 optimisation against the crossed analyser
        opt = ov.OverlapOptimiser(gui, plm, camera, cfg)
        v2 = opt.run()                         # writes GUI fields, leaves null on display slot
        res["v2"] = {k: v for k, v in v2.items() if k not in ("images", "corrB_new", "budget",
                                                              "initial", "new_uv")}
        res["v2_folder"] = v2.get("folder")
        res["roi_px"] = int(opt.roi.sum())

        # 2. proper eps (crossed vs parallel) for the GUI hologram
        opt.rig.start_sw_trigger()
        opt.rig.exposure = v2["exposure_us"]
        _fresh_opt_state(opt, gui)
        m = measure_eps(opt, an, opt.holo(), t_relA, max_exposure_us)
        res["final"] = m
        log(f"RESULT at {t_relA:.2f} deg: eps = {m['eps'] * 100:.3f} % (+/- "
            f"{m['eps_noise'] * 100:.3f}) of the power is not in the target linear state "
            f"-> |dpsi|, |chi| <= {m['angle_bound_deg']:.2f} deg, DOP >= {1 - 2 * m['eps']:.4f}")
        if cfg.bandwidth_nm:
            fl = opt.hb.predicted_chromatic_floor(cfg.bandwidth_nm, cfg.wavelength_nm)[0]
            res["chromatic_floor_pred"] = fl
            log(f"  predicted chromatic floor at 45 deg: {fl * 100:.3f} %")

        # 3. other targets: keep the spatial correction, re-solve rho and beta
        if sweep:
            log("Target sweep (spatial correction fixed; rho, beta re-solved per target)")
            res["sweep"] = target_sweep(opt, an, gui, sweep_targets, max_exposure_us, log)
            _fresh_opt_state(opt, gui)

        # leave the PLM on the target solution
        opt.rig.show(cfg.display_slot)
        res["exposure_parallel_us"] = m["exp_parallel_us"]
    finally:
        if opt is not None:
            opt.rig.restore_free_run()
        try:
            an.set(t_relA, crossed=False)      # park parallel to the target (camera sees the beam)
        finally:
            an.release()

    # 4. independent check of the displayed state
    if verify_stokes:
        uv = [float(b.text()) for b in gui.inputs]
        summ, folder = pm.stokes_measure(camera, (uv[16], uv[17]), f"linear {t_relA:.1f}",
                                         0.7 * res["exposure_parallel_us"],
                                         beam_phases=(uv[18], uv[19]), cal_path=cal_path)
        res["stokes"] = summ
        res["stokes_folder"] = folder
        log(f"Stokes check: azimuth {summ['azimuth_rel_A_deg']:+.2f} deg rel. A "
            f"(target {((t_relA + 90) % 180) - 90:+.2f}), chi {summ['chi_deg']:+.2f} deg, "
            f"DOP {summ['dop']:.3f} (bounded at 1)")
    res["log"] = lines
    _save(res, cfg)
    gui.last_linear_result = res
    return res


def _save(res, cfg):
    now = datetime.now()
    root = cfg.save_root or os.path.join(os.getcwd(), "Data", now.strftime("%Y_%m_%d"))
    folder = os.path.join(root, now.strftime("%H_%M_%S")
                          + f" linear {res['target_rel_A_deg']:.1f}deg")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "linear_summary.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "log"}, f, indent=2, default=float)
    with open(os.path.join(folder, "linear_log.txt"), "w") as f:
        f.write("\n".join(res["log"]))
    if res.get("sweep"):
        _sweep_figure(res, os.path.join(folder, "linear_sweep.png"))
    res["folder"] = folder
    print(f"Saved {folder}")


def _sweep_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["sweep"]
    t = np.array([r["target_rel_A_deg"] for r in rows])
    e = np.array([r["eps"] for r in rows])
    en = np.array([r["eps_noise"] for r in rows])
    dAB = res["dAB_deg"]
    tt = np.linspace(0, 180, 361)
    th = np.where(tt <= dAB, tt * 90 / dAB, 90 + (tt - dAB) * 90 / (180 - dAB))
    e45 = res["final"]["eps"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].errorbar(t, e * 100, en * 100, fmt="o", label="measured")
    ax[0].plot(tt, e45 * np.sin(np.radians(2 * th)) ** 2 * 100, "--",
               label=r"$\varepsilon_{45}\,\sin^2 2\theta$ (visibility-limited)")
    ax[0].set_ylim(0, 1.2 * max(np.max(e), e45) * 100)
    for x, lab in ((0, "A"), (dAB, "B"), (dAB / 2, "D")):
        ax[0].axvline(x, color="0.8", lw=0.8)
        ax[0].text(x, ax[0].get_ylim()[1] * 0.95, lab, ha="center")
    ax[0].set_xlabel("target azimuth rel. channel A (deg)")
    ax[0].set_ylabel(r"$\varepsilon$ = power not in target state (%)")
    ax[0].legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax[1].plot(t, [r["amp_A"] for r in rows], "o-", label="GUI amp A")
    ax[1].plot(t, [r["amp_B"] for r in rows], "s-", label="GUI amp B")
    a2 = ax[1].twinx()
    a2.plot(t, [r["phase_A_pct"] for r in rows], "^:", color="C3", label="phase A (%)")
    a2.set_ylabel("Beam A phase (%)", color="C3")
    ax[1].set_xlabel("target azimuth rel. channel A (deg)")
    ax[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def linear_optimiser(gui, plm, camera, target="D", **kw):
    """Drop-in for the old overlap_optimiser(): returns (newBx, newBy, zero_phase)."""
    res = linearise(gui, plm, camera, target=target, **kw)
    v2 = res["v2"]
    return v2["newBx"], v2["newBy"], v2["zero_phase"]


# ------------------------------------------------------------ live delay meter
def _visibility_floor(I4, betas, IA, IB, roi, smooth_px=1.0):
    """Intensity-weighted mean local fringe visibility V from a four-step set.
    V is the per-pixel modulation divided by its maximum possible value, so it does not depend
    on overlap (amplitude ratio) or on the local relative phase - only on mutual
    coherence.  Returns (1 - V)/2, the null it implies at 45 deg."""
    sm = (lambda x: ov.gaussian_filter(x, smooth_px)) if smooth_px else (lambda x: x)
    I4s = np.stack([sm(i) for i in I4])
    Ap, Bp, _ = ov.fourstep(I4s, betas)
    resid = (I4s[0] + I4s[2] - I4s[1] - I4s[3]) / 2
    sig2 = (1.4826 * np.median(np.abs(resid - np.median(resid)))) ** 2
    Bp = np.sqrt(np.clip(Bp ** 2 - sig2, 0, None))          # remove noise bias
    IAs, IBs = np.clip(sm(IA), 0, None), np.clip(sm(IB), 0, None)
    g = np.sqrt(IAs * IBs)
    sel = roi & (g > 0.05 * g[roi].max())
    # Single-channel frames give only the local A:B ratio; the absolute level comes
    # from the four-step mean Ap (= I_A' + I_B' in the combined hologram), so a
    # common change of PLM efficiency between single-channel and combined holograms
    # cancels: 2 sqrt(I_A' I_B') = Ap * 2 sqrt(IA IB) / (IA + IB).
    denom = Ap * 2 * g / np.maximum(IAs + IBs, 1e-12)
    V = np.sum(Bp[sel]) / np.sum(denom[sel])
    return float((1 - V) / 2), int(sel.sum())


def delay_meter(gui, plm, camera, duration_s=300.0, target="D", n_avg=2,
                realign_px=0.5, cal_path=pm.CAL_FILE):
    """Live A/B coherence read-out for hand-tuning the group-delay compensator.

    Analyser crossed to the target (default: A/B bisector).  The current GUI
    hologram is uploaded as a four-step set (B phase 0, pi/2, pi, 3pi/2) plus
    A-only, B-only and blank frames and cycled for duration_s.  Each cycle prints

      null%    - total leakage (overlap + coherence)
      COH%     - (1 - V)/2 with V the intensity-weighted local fringe visibility:
                 the null you would get with perfect overlap.  Independent of the
                 A-B phase; only weakly dependent on displacement (simulation:
                 +~10 % relative per px of A-B offset), hence the re-centring
                 below.  MINIMISE THIS.
      amp%, wavefr%  - overlap terms of the error budget
      |A-B|px  - centroid separation

    If the glass displaces a beam by more than realign_px, one Newton step on B's
    tilt (Jacobian measured at start) re-centres it and the holograms are
    re-uploaded.  The GUI is frozen while this runs; it stops after duration_s and
    leaves the GUI fields unchanged (re-run the linear optimiser at the best glass
    position)."""
    cal = pm.load_cal(cal_path)
    t_relA = target_rel_A(cal, target)
    cfg = ov.OverlapConfig(n_avg=n_avg, verbose=False)
    opt = ov.OverlapOptimiser(gui, plm, camera, cfg)
    an = Analyser(cal)
    rows = []

    def upload():
        holos = [opt.holo(beta=b) for b in opt.BETAS]
        holos += [opt.holo(kind="A"), opt.holo(kind="B"), opt.holo(kind="blank")]
        return opt.rig.upload(holos)

    try:
        an.set(t_relA, crossed=True)
        opt.rig.start_sw_trigger()
        opt.auto_exposure()
        opt.coarse_align()                        # measures opt.J, sets opt.roi
        opt.auto_exposure()                       # overlap changed: re-expose
        slots = upload()
        fs = opt.rig.full_scale or 255.0
        print(f"Delay meter: analyser crossed to {t_relA:.2f} deg rel. A, exposure "
              f"{opt.rig.exposure:.0f} us, ROI {int(opt.roi.sum())} px, {duration_s:.0f} s")
        print("   t(s)   null%   COH%    amp%  wavefr%  |A-B|px  peak")
        stop_file = os.path.join(os.getcwd(), "STOP_DELAY_METER")
        if os.path.exists(stop_file):
            os.remove(stop_file)
        print(f"  (to stop early: create an empty file {stop_file})")
        t0 = time.time()
        while time.time() - t0 < duration_s:
            if os.path.exists(stop_file):
                os.remove(stop_file)
                print("  stop file found - stopping")
                break
            raw, steps = [], []
            tc = time.time()
            try:
                for s in slots:
                    ts = time.time()
                    opt.rig.show(s)
                    tg = time.time()
                    raw.append(opt.rig.grab())
                    steps.append((s, tg - ts, time.time() - tg))
            except KeyboardInterrupt:
                print("  interrupted - stopping cleanly")
                break
            if time.time() - tc > 5.0:
                worst = max(steps, key=lambda x: max(x[1], x[2]))
                print(f"  ! slow cycle {time.time() - tc:.1f} s - slowest: slot {worst[0]}: "
                      f"PLM show {worst[1]:.1f} s, camera grab {worst[2]:.1f} s")
            dark = raw[-1]
            I4 = [r - dark for r in raw[:4]]
            IA, IB = raw[4] - dark, raw[5] - dark
            b = ov.error_budget(I4, opt.BETAS, IA, IB, opt.roi)
            coh, npx = _visibility_floor(I4, opt.BETAS, IA, IB, opt.roi)
            try:
                cA, cB = ov.centroid(IA), ov.centroid(IB)
                sep = float(np.linalg.norm(cA - cB))
            except RuntimeError:
                cA = cB = None
                sep = np.nan
            pk = max(r[opt.roi].max() for r in raw[:6]) / fs
            t = time.time() - t0
            rows.append((t, b["eps"], coh, b["N_amp"], b["N_phase"], sep, pk))
            note = ""
            if pk >= 0.98:
                opt.rig.exposure = opt.rig.exposure * 0.6
                note = f"  SATURATED -> exposure {opt.rig.exposure:.0f} us (ignore this line)"
            if cA is not None and sep > realign_px:
                d = np.linalg.solve(opt.J, cA - cB)
                opt.p["dnu_x"] += d[0]
                opt.p["dnu_y"] += d[1]
                slots = upload()
                opt.roi = ov.make_roi(IA, IB, cfg.roi_threshold, cfg.roi_dilate_px)
                note += "  (B re-centred)"
            print(f"{t:7.1f}  {b['eps'] * 100:6.2f}  {coh * 100:6.2f}  {b['N_amp'] * 100:5.2f}  "
                  f"{b['N_phase'] * 100:6.2f}   {sep:6.2f}  {pk:4.2f}{note}")
    finally:
        try:
            an.set(t_relA, crossed=False)
        finally:
            an.release()
            opt.rig.restore_free_run()
    if rows:
        r = np.array(rows)
        k = int(np.nanargmin(r[:, 2]))
        print(f"Best COH floor {r[k, 2] * 100:.2f} % at t = {r[k, 0]:.0f} s "
              f"(null {r[k, 1] * 100:.2f} %)")
        now = datetime.now()
        folder = os.path.join(os.getcwd(), "Data", now.strftime("%Y_%m_%d"))
        os.makedirs(folder, exist_ok=True)
        np.savetxt(os.path.join(folder, now.strftime("%H_%M_%S") + " delay_meter.csv"), r,
                   delimiter=",", header="t_s,null,coh_floor,N_amp,N_phase,AB_sep_px,peak_frac")
    return rows

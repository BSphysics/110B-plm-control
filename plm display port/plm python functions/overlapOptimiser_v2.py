"""
overlapOptimiser_v2.py
======================

Drop-in replacement for overlapOptimiser.overlap_optimiser() that optimises the
quantity you actually care about -- the depth of the polarisation null -- rather
than a proxy for it (centroid coincidence).

Key differences from v1
-----------------------
1.  Metric: four-step phase shifting between channels A and B through the analyser.
    For each candidate hologram, B's phase is stepped by 0, pi/2, pi, 3pi/2 and the
    ROI signal is fitted to  I(beta) = A + B cos(beta - phi).  This gives the null
    depth  eps = (A - B)/(A + B)  and the null phase directly (no 1-D phase scan,
    no sensitivity to where the scan happened to start).  pi/2 steps are 4 levels
    of a 16-level PLM, so they are as exact as the device allows.

2.  Background: every batch includes a blank hologram (no grating -> all light in
    the zero order), so the dark/stray level in the ROI is measured under identical
    conditions and subtracted.  With an 8-bit camera and a 64x64 ROI, an unsubtracted
    or rectified dark level alone can masquerade as a few-% "extinction".

3.  Parameters: tilt is perturbed as a spatial frequency (cycles/px), which moves
    the spot linearly and identically in x and y (the GUI's *period* parameter
    does not -- that's why v1 needed a 20x different gain in x).  In addition to
    tilt, the search covers low-order Zernike terms on channel B relative to A
    (defocus, astigmatism x2, coma x2) and, optionally, B's envelope centre and
    waist.  Amplitude ratio is balanced analytically from single-channel powers
    measured through the analyser.

4.  Search: one Jacobian-calibrated Newton step on centroids for the coarse
    overlap (replaces the slow proportional loops), then coordinate-wise parabolic
    fits on eps (the null is quadratic in every small parameter error, so a
    5-point parabola lands on the minimum in one go).  Candidate holograms are
    bitpacked in batches into spare frame slots, so the UI is paused once per
    batch rather than once per measurement.

5.  Diagnosis: the same four images, plus single-channel images, give a per-pixel
    budget that splits the residual null into
        N_amp   - local amplitude mismatch   (fixable: ratio / position / size)
        N_phase - local wavefront mismatch   (fixable: tilt / Zernikes)
        N_coh   - local loss of fringe visibility (NOT fixable with the PLM:
                  chromatic walk-off between channels, group-delay mismatch,
                  drift during the measurement).
    If N_coh dominates at the end, you are at the dispersion floor.

GUI integration (plm_display_port.py, in the overlap_optimiser_flag branch):

    from overlapOptimiser_v2 import overlap_optimiser_v2, OverlapConfig
    newBxTilt, newByTilt, zero_phase = overlap_optimiser_v2(
        self, plm, camera, OverlapConfig(bandwidth_nm=4.0))

It writes the new B periods, amplitudes and null phase into the GUI fields,
folds the Zernike correction into self.beam_B_correction_data (saved as a new
.npy and registered via save_phase_file), and leaves the PLM showing the verified
null hologram in `display_slot`.  The full result is in self.last_overlap_result.

NOT yet run on the real rig -- it was validated against a simulated rig
(test_overlap_v2.py).  Check the CONFIG notes (frame slots, dump direction)
before first use.
"""

import os
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label
from pypylon import pylon

from generatePhaseTilt import generate_phase_tilt
from HGMode import HG_mode
from ampModPhase import amp_mod_phase

ROWS, COLS = 800, 1358
NUM_HOLO = 24


# --------------------------------------------------------------------------- config
@dataclass
class OverlapConfig:
    # Frame slots used for batches.  Must avoid slots the GUI uses
    # (0 main, 1 multibeam, 2 optimiser display, 60 blank, 66 multibeam seq).
    work_slots: tuple = tuple(range(10, 34))          # 24 slots
    display_slot: int = 2

    n_avg: int = 5                 # camera frames averaged per hologram
    settle_s: float = 0.15         # wait after set_frame before grabbing
    target_peak: float = 0.6       # auto-exposure: brightest pixel / full scale
    full_scale: float = None       # None -> 255 for uint8 images, 4095 otherwise

    roi_threshold: float = 0.01    # ROI = pixels above this fraction of A/B peak ...
    roi_dilate_px: int = 3         # ... dilated by this many px

    # Parameters searched (B relative to A).  Units: dnu in cycles/PLM-px,
    # z_* in rad (polynomial on PLM coords normalised to half-height = 1),
    # cx/cy in PLM px (B envelope centre), waist as fractional change.
    params: tuple = ("dnu_x", "dnu_y", "z_defocus", "z_astig0", "z_astig45",
                     "z_coma_x", "z_coma_y")
    init_step: dict = field(default_factory=lambda: {
        "dnu_x": 0.15 / COLS, "dnu_y": 0.15 / ROWS,
        "z_defocus": 0.25, "z_astig0": 0.25, "z_astig45": 0.25,
        "z_coma_x": 0.2, "z_coma_y": 0.2,
        "cx": 30.0, "cy": 30.0, "waist": 0.08})
    max_passes: int = 6
    tol_rel: float = 0.05          # stop when a full pass improves eps by < 5 % (relative)

    # Where the unwanted channel is sent for single-channel measurements.  Keeping
    # both channels in the hologram keeps amp_mod_phase's normalisation identical to
    # the combined case.  CHECK on the rig that this spot misses the camera ROI.
    dump_dnu: tuple = (0.0, 0.05)

    hdr_factor: float = 10.0       # final null measured at this x exposure
    # Exposures are rounded to whole PLM sub-hologram periods (1.44 kHz) so every
    # frame integrates the same number of PLM refresh cycles; an exposure of ~1.15
    # periods catches a varying number of PLM transitions -> frame-to-frame flicker.
    # Set to None to disable.
    exposure_quantum_us: float = 1e6 / 1440
    bandwidth_nm: float = None     # FWHM laser spectrum; enables floor prediction
    wavelength_nm: float = 800.0

    apply_to_gui: bool = True
    save: bool = True
    save_root: str = None          # default: ./Data/<date>/<time> overlap v2
    verbose: bool = True


# --------------------------------------------------------------------- hologram model
def _freq_of_tilt(px, py, offx, offy):
    """Spatial frequency (cycles/px) produced by generate_phase_tilt for given periods.
    Measured from the function itself so no assumption about its conventions."""
    T = np.asarray(generate_phase_tilt(16, 16, px, py, offx, offy), dtype=float)
    fx = np.median(np.angle(np.exp(1j * np.diff(T, axis=1)))) / (2 * np.pi)
    fy = np.median(np.angle(np.exp(1j * np.diff(T, axis=0)))) / (2 * np.pi)
    return fx, fy


def _period_for_freq(p0, target, axis, other, offx, offy):
    """Solve generate_phase_tilt period giving `target` cycles/px on `axis` (secant)."""
    def f(p):
        args = (p, other) if axis == 0 else (other, p)
        return _freq_of_tilt(*args, offx, offy)[axis] - target
    a, b = p0, p0 * 1.01
    fa, fb = f(a), f(b)
    for _ in range(30):
        if abs(fb) < 1e-10 or fb == fa:
            break
        a, b, fa = b, b - fb * (b - a) / (fb - fa), fb
        fb = f(b)
    return b


class HologramBuilder:
    """Reproduces InteractiveGUI.update_value()'s two-beam hologram, with
    perturbations applied to channel B."""

    ZERN = ("z_defocus", "z_astig0", "z_astig45", "z_coma_x", "z_coma_y")

    def __init__(self, gui, uv=None, corrB=None):
        uv = list(uv) if uv is not None else [float(b.text()) for b in gui.inputs]
        self.uv = uv
        self.bs = list(gui.button_states)
        self.corrA = np.asarray(gui.beam_A_correction_data, dtype=float)
        self.corrB = np.asarray(gui.beam_B_correction_data if corrB is None else corrB,
                                dtype=float)

        y, x = np.mgrid[0:ROWS, 0:COLS].astype(np.float32)
        self.X = x - COLS / 2
        self.Y = y - ROWS / 2
        u, v = self.X / (ROWS / 2), self.Y / (ROWS / 2)
        r2 = u * u + v * v
        self.zbasis = {"z_defocus": 2 * r2 - 1, "z_astig0": u * u - v * v,
                       "z_astig45": 2 * u * v, "z_coma_x": (3 * r2 - 2) * u,
                       "z_coma_y": (3 * r2 - 2) * v}

        self.tiltA = generate_phase_tilt(ROWS, COLS, uv[0], uv[1], self.bs[0], self.bs[1])
        self.tiltB = generate_phase_tilt(ROWS, COLS, uv[2], uv[3], self.bs[2], self.bs[3])
        self.nuA = _freq_of_tilt(uv[0], uv[1], self.bs[0], self.bs[1])
        self.nuB = _freq_of_tilt(uv[2], uv[3], self.bs[2], self.bs[3])

        self.hgA_phase, self.hgA_amp = HG_mode(COLS, ROWS, uv[4], uv[5], uv[8], uv[9],
                                               uv[12], uv[13])
        self._hgB_cache = {}
        ga = np.array([uv[16], uv[17]], dtype=float)
        ga = ga / ga.max()
        self.ampA0, self.ampB0 = ga
        self.phaseA = (self.tiltA - self.corrA + self.hgA_phase
                       + uv[18] / 100 * 2 * np.pi)
        self.phaseB_base = self.tiltB - self.corrB + uv[19] / 100 * 2 * np.pi

    def _hgB(self, cx=0.0, cy=0.0, waist=0.0):
        key = (round(cx, 3), round(cy, 3), round(waist, 5))
        if key not in self._hgB_cache:
            uv = self.uv
            s = 1.0 + waist
            self._hgB_cache = {key: HG_mode(COLS, ROWS, uv[6], uv[7], uv[10] * s,
                                            uv[11] * s, uv[14] + cx, uv[15] + cy)}
        return self._hgB_cache[key]

    def zernike_phase(self, p):
        out = np.zeros((ROWS, COLS), np.float32)
        for k in self.ZERN:
            if p.get(k, 0.0):
                out += p[k] * self.zbasis[k]
        return out

    def fields(self, p, beta=0.0, rho=1.0):
        hp, ha = self._hgB(p.get("cx", 0.0), p.get("cy", 0.0), p.get("waist", 0.0))
        phB = (self.phaseB_base + hp + beta
               + 2 * np.pi * (p.get("dnu_x", 0.0) * self.X + p.get("dnu_y", 0.0) * self.Y)
               + self.zernike_phase(p))
        EA = self.ampA0 * self.hgA_amp * np.exp(1j * self.phaseA)
        EB = rho * self.ampB0 * ha * np.exp(1j * phB)
        return EA, EB

    def build(self, p, beta=0.0, rho=1.0, kind="both", dump=(0.0, 0.05)):
        """kind: 'both' | 'A' (B dumped) | 'B' (A dumped) | 'blank'.
        Returns plm phase map in [0, 1] (float32), same convention as the GUI."""
        if kind == "blank":
            return np.full((ROWS, COLS), 0.5, np.float32)
        EA, EB = self.fields(p, beta, rho)
        dumpramp = np.exp(2j * np.pi * (dump[0] * self.X + dump[1] * self.Y))
        if kind == "A":
            EB = EB * dumpramp
        elif kind == "B":
            EA = EA * dumpramp
        psi = amp_mod_phase(EA + EB)
        return ((psi + np.pi) / (2 * np.pi)).astype(np.float32)

    def predicted_chromatic_floor(self, bandwidth_nm, wavelength_nm):
        """Small-shift, Gaussian-spectrum estimate of the dispersion-limited null:
            eps_floor ~ pi^2 (sigma_lambda/lambda)^2 |dnu|^2 <x_par^2>
        dnu = carrier frequency difference between channels (cycles/px),
        <x_par^2> = second moment of |E_PLM|^2 along dnu (px^2).  Uses the encoded
        envelope only (uniform illumination assumed), so it is an upper bound if
        the illuminating beam is narrower than the hologram."""
        dnu = np.array(self.nuA) - np.array(self.nuB)
        n = np.linalg.norm(dnu)
        if n == 0:
            return 0.0, dnu, 0.0
        I = self.hgA_amp ** 2
        xp = (self.X * dnu[0] + self.Y * dnu[1]) / n
        m = np.sum(I * xp) / I.sum()
        x2 = np.sum(I * (xp - m) ** 2) / I.sum()
        s = bandwidth_nm / 2.3548 / wavelength_nm
        return float(np.pi ** 2 * s ** 2 * n ** 2 * x2), dnu, float(np.sqrt(x2))


# ------------------------------------------------------------------------ hardware
class Rig:
    def __init__(self, plm, camera, cfg, log):
        self.plm, self.cam, self.cfg, self.log = plm, camera, cfg, log
        self.buf = np.empty((NUM_HOLO, ROWS, COLS), dtype=np.float32)
        self._exp_attr = (camera.ExposureTimeAbs if hasattr(camera, "ExposureTimeAbs")
                          else camera.ExposureTime)
        self.exposure0 = self._exp_attr.GetValue()
        self.full_scale = cfg.full_scale

    # exposure
    @property
    def exposure(self):
        return self._exp_attr.GetValue()

    @exposure.setter
    def exposure(self, v):
        """Clamped to the camera's limits (Basler: 1 s max), so callers must read
        rig.exposure back rather than assume the requested value was set."""
        a = self._exp_attr
        q = getattr(self.cfg, "exposure_quantum_us", None)
        if q:
            v = max(1, round(float(v) / q)) * q
        lo = a.GetMin() if hasattr(a, "GetMin") else 0.0
        hi = a.GetMax() if hasattr(a, "GetMax") else np.inf
        a.SetValue(float(np.clip(v, lo, hi)))

    def start_sw_trigger(self):
        c = self.cam
        if c.IsGrabbing():
            c.StopGrabbing()
        c.TriggerSelector.SetValue("FrameStart")
        c.TriggerMode.SetValue("On")
        c.TriggerSource.SetValue("Software")

    def restore_free_run(self):
        c = self.cam
        if c.IsGrabbing():
            c.StopGrabbing()
        c.TriggerMode.SetValue("Off")
        self.exposure = self.exposure0
        c.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def upload(self, holos):
        slots = self.cfg.work_slots
        if len(holos) > len(slots):
            raise ValueError(f"batch of {len(holos)} > {len(slots)} work slots")
        self.plm.pause_ui()
        try:
            for h, s in zip(holos, slots):
                np.copyto(self.buf, h[None, :, :])
                self.plm.bitpack_and_insert_gpu(self.buf, int(s))
        finally:
            self.plm.resume_ui()
        return slots[:len(holos)]

    def show(self, slot):
        self.plm.set_frame(int(slot))
        time.sleep(self.cfg.settle_s)
        self.plm.play()
        self.plm.play()

    def grab(self, n=None):
        n = n or self.cfg.n_avg
        c = self.cam
        c.StartGrabbingMax(n)
        acc = None
        for _ in range(n):
            if hasattr(c, "WaitForFrameTriggerReady"):
                c.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_ThrowException)
            else:
                time.sleep(0.02)
            c.TriggerSoftware.Execute()
            r = c.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            try:
                if not r.GrabSucceeded():
                    raise RuntimeError(f"grab failed: {r.ErrorDescription}")
                a = r.Array
                if self.full_scale is None:
                    self.full_scale = 255.0 if a.dtype == np.uint8 else 4095.0
                acc = a.astype(np.float64) if acc is None else acc + a
            finally:
                r.Release()
        if c.IsGrabbing():
            c.StopGrabbing()
        return acc / n

    def measure(self, holos):
        """Upload holos (list of 2D maps) and return list of mean raw images."""
        slots = self.upload(holos)
        out = []
        for s in slots:
            self.show(s)
            out.append(self.grab())
        return out


# --------------------------------------------------------------------------- analysis
def fourstep(I, betas):
    """I: sequence of ROI sums (or images, stacked on axis 0), betas: phase steps.
    Returns A, B, phi with I(beta) = A + B cos(beta - phi)."""
    I = np.asarray(I, dtype=float)
    e = np.exp(1j * np.asarray(betas)).reshape((-1,) + (1,) * (I.ndim - 1))
    S = np.sum(I * e, axis=0)
    N = len(betas)
    return I.mean(axis=0), 2 * np.abs(S) / N, np.angle(S)


def centroid(img, frac=0.05):
    """Centroid of the connected spot containing the (smoothed) peak only.  A
    whole-frame threshold lets ghosts, scatter from the dumped channel and
    background structure pull the centroid - at long exposures (small, dim
    beams) that can be several pixels."""
    S = gaussian_filter(img, 1.5)
    pk = np.unravel_index(np.argmax(S), S.shape)
    sig = 1.4826 * np.median(np.abs(S - np.median(S)))
    lab, _ = label(S > max(frac * S[pk], 6 * sig))
    m = lab == lab[pk]
    w = np.where(m, np.clip(img - frac * S[pk], 0, None), 0.0)
    if w.sum() <= 0:
        raise RuntimeError("no signal for centroid")
    y, x = np.indices(img.shape)
    return np.array([np.sum(w * x), np.sum(w * y)]) / w.sum()


def make_roi(IA, IB, thr, dil):
    """Connected region(s) containing the A and B peaks, thresholded on a smoothed
    image at max(thr * peak, 6 sigma_noise).  A plain threshold on raw 8-bit frames
    lets noise pixels in, which biases every sum-over-ROI quantity."""
    rois = []
    for I in (IA, IB):
        S = gaussian_filter(I, 1.0)
        sig = 1.4826 * np.median(np.abs(S - np.median(S)))
        m = S > max(thr * S.max(), 6 * sig)
        lab, _ = label(m)
        pk = np.unravel_index(np.argmax(S), S.shape)
        rois.append(lab == lab[pk])
    m = rois[0] | rois[1]
    return binary_dilation(m, iterations=dil) if dil else m


def error_budget(I4, betas, IA, IB, roi, smooth_px=1.0):
    """Per-pixel decomposition of the global null (see module docstring).

    null = N_amp + N_phase + N_coh exactly, with
      N_phase = sum B_p (1 - cos psi_p)          local relative-phase error
      N_vis   = sum (A_p - B_p)                  local loss of modulation
      N_amp   = sum (sqrt I_A - sqrt I_B)^2      part of N_vis explained by amplitude
      N_coh   = N_vis - N_amp                    the rest: spectral / temporal
    Frames are smoothed (linear, so the 4-step fit is unaffected) and the Rician
    bias of the per-pixel modulation is removed using a noise estimate taken from
    I0 + I2 - I1 - I3, which is zero for a perfect sinusoid."""
    betas = np.asarray(betas)
    sm = (lambda x: gaussian_filter(x, smooth_px)) if smooth_px else (lambda x: x)
    I4 = np.stack([sm(i) for i in I4])
    IAs, IBs = sm(IA), sm(IB)
    Ag, Bg, phig = fourstep([np.sum(i[roi]) for i in I4], betas)
    Ap, Bp, phip = fourstep(I4, betas)
    resid = None
    if len(betas) == 4 and np.allclose(np.diff(betas), np.pi / 2):
        resid = (I4[0] + I4[2] - I4[1] - I4[3]) / 2          # pure noise / drift
        sig2 = (1.4826 * np.median(np.abs(resid - np.median(resid)))) ** 2
        Bp = np.sqrt(np.clip(Bp ** 2 - 4 * sig2 / len(betas), 0, None))
    psi = np.angle(np.exp(1j * (phip - phig)))
    Imax = Ag + Bg
    N_phase = np.sum((Bp * (1 - np.cos(psi)))[roi]) / Imax
    N_vis = np.sum((Ap - Bp)[roi]) / Imax
    N_amp = np.sum(((np.sqrt(np.clip(IAs, 0, None)) - np.sqrt(np.clip(IBs, 0, None))) ** 2)
                   [roi]) / Imax
    N_coh = N_vis - N_amp
    g = np.sqrt(np.clip(IAs, 0, None) * np.clip(IBs, 0, None))
    m = np.clip(np.where(g > 0, Bp / np.maximum(2 * g, 1e-12), 0.0), 0, 1)
    # low-order fit of the relative phase across the spot (camera plane)
    yy, xx = np.indices(IA.shape)
    w = (g * roi).ravel()
    sel = w > 0.05 * w.max()
    cx, cy = np.sum(w * xx.ravel()) / w.sum(), np.sum(w * yy.ravel()) / w.sum()
    X, Y = (xx.ravel() - cx)[sel], (yy.ravel() - cy)[sel]
    M = np.stack([np.ones_like(X), X, Y, X * X + Y * Y, X * X - Y * Y, 2 * X * Y], 1)
    coef = np.linalg.lstsq(M * w[sel, None], psi.ravel()[sel] * w[sel], rcond=None)[0]
    drift = (abs(np.sum(resid[roi])) / Imax) if resid is not None else np.nan
    return dict(eps=(Ag - Bg) / (Ag + Bg), N_amp=N_amp, N_phase=N_phase, N_coh=N_coh,
                fit_residual=drift,
                phase_fit=dict(zip(["piston", "tilt_x", "tilt_y", "defocus",
                                    "astig0", "astig45"], coef.tolist())),
                maps=dict(modulation=m, rel_phase=psi, null=Ap - Bp))


# ---------------------------------------------------------------------- optimiser
class OverlapOptimiser:
    BETAS = np.arange(4) * np.pi / 2

    def __init__(self, gui, plm, camera, cfg):
        self.gui, self.cfg = gui, cfg
        self.log_lines = []
        self.rig = Rig(plm, camera, cfg, self.log)
        self.hb = HologramBuilder(gui)
        self.p = {k: 0.0 for k in cfg.params}
        self.h = {k: cfg.init_step[k] for k in cfg.params}
        self.rho = 1.0
        self.beta_null = np.pi
        self.history = []
        self.dark = None
        self.roi = None

    def log(self, *a):
        s = " ".join(str(x) for x in a)
        self.log_lines.append(s)
        if self.cfg.verbose:
            print(s)

    # -- measurement helpers ---------------------------------------------------
    def holo(self, p=None, beta=0.0, kind="both"):
        return self.hb.build(self.p if p is None else p, beta, self.rho, kind,
                             self.cfg.dump_dnu)

    def run_batch(self, items, check_sat=True):
        """items: list of (p, beta, kind).  A blank and single-channel A/B images at
        the current point are appended automatically.  Returns (imgs, IA, IB) with
        the batch's own dark subtracted."""
        holos = [self.holo(p, b, k) for p, b, k in items]
        holos += [self.holo(kind="A"), self.holo(kind="B"), self.holo(kind="blank")]
        raw = self.rig.measure(holos)
        self.dark = raw[-1]
        sub = [r - self.dark for r in raw[:-1]]
        if check_sat:
            self._check_saturation(raw)
        return sub[:-2], sub[-2], sub[-1]

    def _check_saturation(self, raw):
        fs = self.rig.full_scale
        peak = max(r.max() for r in raw)
        if peak >= 0.98 * fs:
            self.log(f"  ! saturation (peak {peak:.0f}/{fs:.0f}) - reducing exposure")
            self.rig.exposure = self.rig.exposure * 0.6
            self.saturated = True

    def eps_from(self, imgs4):
        A, B, phi = fourstep([np.sum(i[self.roi]) for i in imgs4], self.BETAS)
        return (A - B) / (A + B), A, B, phi

    # -- stages ------------------------------------------------------------------
    def auto_exposure(self):
        for _ in range(10):
            self.saturated = False
            imgs, IA, IB = self.run_batch([(self.p, b, "both") for b in self.BETAS],
                                          check_sat=False)
            peak = max(i.max() + self.dark.max() for i in imgs)
            fs = self.rig.full_scale
            target = self.cfg.target_peak * fs
            self.log(f"  exposure {self.rig.exposure:.0f} us -> peak {peak:.1f}")
            if peak >= 0.98 * fs:
                self.rig.exposure = self.rig.exposure / 4
                continue
            if 0.75 * target < peak < 1.2 * target:
                break
            self.rig.exposure = self.rig.exposure * target / max(peak, 1.0)
        return imgs, IA, IB

    def coarse_align(self):
        """Newton step on centroids using a measured 2x2 Jacobian."""
        hx, hy = 1.0 / COLS, 1.0 / ROWS
        self.p.setdefault("dnu_x", 0.0)
        self.p.setdefault("dnu_y", 0.0)
        p0 = dict(self.p)
        px, py = dict(p0), dict(p0)
        px["dnu_x"] += hx
        py["dnu_y"] += hy
        imgs, IA, IB = self.run_batch([(px, 0, "B"), (py, 0, "B")])
        cA, cB = centroid(IA), centroid(IB)
        J = np.stack([(centroid(imgs[0]) - cB) / hx, (centroid(imgs[1]) - cB) / hy], 1)
        self.J = J
        self.log(f"  centroid A {cA.round(2)}, B {cB.round(2)}; "
                 f"Jacobian (px per cycle/px):\n{J.round(1)}")
        best = (np.linalg.norm(cA - cB), dict(self.p), IA, IB)
        for it in range(4):
            d = np.linalg.solve(J, cA - cB)
            self.p["dnu_x"] += d[0]
            self.p["dnu_y"] += d[1]
            imgs, IA, IB = self.run_batch([])
            cA, cB = centroid(IA), centroid(IB)
            dist = np.linalg.norm(cA - cB)
            self.log(f"  Newton {it}: |A-B| = {dist:.3f} px")
            if dist < best[0]:
                best = (dist, dict(self.p), IA, IB)
            else:
                self.log("  ! Newton step did not help (unreliable centroids?) - "
                         "keeping the best point; the null-based scans take over")
                break
            if dist < 0.2:
                break
        self.p, IA, IB = best[1], best[2], best[3]
        self.roi = make_roi(IA, IB, self.cfg.roi_threshold, self.cfg.roi_dilate_px)
        self.log(f"  ROI: {int(self.roi.sum())} px")

    def balance(self, IA, IB):
        PA, PB = np.sum(IA[self.roi]), np.sum(IB[self.roi])
        self.rho *= np.sqrt(max(PA, 1e-9) / max(PB, 1e-9))
        self.log(f"  P_B/P_A = {PB / PA:.3f} -> amplitude ratio rho = {self.rho:.4f}")
        return PA, PB

    def scan(self, name):
        h = self.h[name]
        offs = np.array([-2, -1, 0, 1, 2]) * h
        items = []
        for o in offs:
            q = dict(self.p)
            q[name] += o
            items += [(q, self.beta_null + b, "both") for b in self.BETAS]
        imgs, IA, IB = self.run_batch(items)
        eps = []
        b0 = self.beta_null
        for k in range(len(offs)):
            e, A, B, phi = self.eps_from(imgs[4 * k:4 * k + 4])
            eps.append(e)
            if k == 2:
                # phi is relative to the steps' centre b0 (eps_from fits against BETAS)
                self.beta_null = (b0 + phi + np.pi) % (2 * np.pi)
        eps = np.array(eps)
        # weight by 1/eps: the null is quadratic near its minimum but flattens far
        # from it, so trust the low points more
        a, b, c = np.polyfit(offs / h, eps, 2, w=1 / np.maximum(eps, 1e-4))
        if a > 0 and abs(-b / (2 * a)) <= 2.5:
            x0 = float(np.clip(-b / (2 * a), -2, 2))
            self.h[name] = h * (0.6 if abs(x0) < 1 else 1.0)
            how = "parabola"
        else:
            x0 = float(offs[np.argmin(eps)] / h)
            self.h[name] = h * 1.5
            how = "best-sample"
        self.p[name] += x0 * h
        self.history.append(dict(param=name, offs=offs.tolist(), eps=eps.tolist(),
                                 move=x0 * h, how=how))
        self.log(f"  {name:10s} eps[-2h..2h] = " + " ".join(f"{e * 100:5.2f}" for e in eps)
                 + f" %  -> move {x0:+.2f} h ({how}), value {self.p[name]:+.4g}")
        return eps[2], IA, IB

    def final_measure(self):
        """Four-step + single-channel at the optimum, error budget, HDR null."""
        imgs, IA, IB = self.run_batch([(self.p, self.beta_null + b, "both")
                                       for b in self.BETAS])
        bud = error_budget(imgs, self.beta_null + self.BETAS, IA, IB, self.roi)
        A, B, phi = fourstep([np.sum(i[self.roi]) for i in imgs], self.beta_null + self.BETAS)
        self.beta_null = phi + np.pi
        Imax_img = imgs[int(np.argmax([np.sum(i[self.roi]) for i in imgs]))]
        # HDR null: same null hologram at k x exposure, with its own dark
        t0 = self.rig.exposure
        null_pk = np.max(np.abs(bud["maps"]["null"])) + self.dark.max()
        k = float(np.clip(0.7 * self.rig.full_scale / max(null_pk, 1e-6), 1.0,
                          self.cfg.hdr_factor))
        self.rig.exposure = t0 * k
        k = self.rig.exposure / t0                  # actual factor after clamping
        raw = self.rig.measure([self.holo(beta=self.beta_null), self.holo(kind="blank")])
        self.rig.exposure = t0
        null_hdr = (raw[0] - raw[1]) / k
        sat = raw[0].max() >= 0.98 * self.rig.full_scale
        eps_hdr = np.sum(null_hdr[self.roi]) / (A + B)
        dark_noise = np.std(raw[1][self.roi]) * np.sqrt(self.roi.sum()) / k / (A + B)
        self.hdr_k = k
        return dict(budget=bud, eps_fit=(A - B) / (A + B), eps_hdr=float(eps_hdr),
                    hdr_saturated=bool(sat), dark_floor=float(dark_noise),
                    PA=float(np.sum(IA[self.roi])), PB=float(np.sum(IB[self.roi])),
                    images=dict(IA=IA, IB=IB, null=null_hdr, max=Imax_img,
                                four=np.stack(imgs)))

    # -- main ------------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        t_start = time.time()
        self.rig.start_sw_trigger()
        try:
            if cfg.bandwidth_nm:
                fl, dnu, xr = self.hb.predicted_chromatic_floor(cfg.bandwidth_nm,
                                                               cfg.wavelength_nm)
                self.log(f"Predicted chromatic floor ({cfg.bandwidth_nm} nm FWHM): "
                         f"{fl * 100:.2f} %  [dnu_AB = {np.round(dnu, 4)} cyc/px, "
                         f"rms hologram extent along dnu = {xr:.0f} px]")
            self.log("Auto-exposure")
            self.auto_exposure()
            self.log("Coarse alignment (centroid Newton)")
            self.coarse_align()
            imgs, IA, IB = self.run_batch([(self.p, b, "both") for b in self.BETAS])
            self.balance(IA, IB)
            imgs, IA, IB = self.run_batch([(self.p, b, "both") for b in self.BETAS])
            e0, A, B, phi = self.eps_from(imgs)
            self.beta_null = phi + np.pi
            self.log(f"Start of fine search: null = {e0 * 100:.2f} %")
            self.initial = dict(eps=e0, budget=error_budget(imgs, self.BETAS, IA, IB,
                                                            self.roi))
            self._log_budget(self.initial["budget"], "Budget before fine search")

            last = e0
            for ps in range(cfg.max_passes):
                self.log(f"Pass {ps + 1}")
                for name in cfg.params:
                    e, IA, IB = self.scan(name)
                self.balance(IA, IB)
                self.log(f"  end of pass: null ~ {e * 100:.2f} %")
                if (last - e) < cfg.tol_rel * last:
                    break
                last = e
            self.log("Final measurement")
            res = self.final_measure()
            self._log_budget(res["budget"], "Budget at optimum")
            self.log(f"  null (4-step fit) = {res['eps_fit'] * 100:.2f} %, "
                     f"null (HDR x{self.hdr_k:.1f}) = {res['eps_hdr'] * 100:.2f} %"
                     + ("  [HDR SATURATED]" if res["hdr_saturated"] else "")
                     + f", dark-noise floor = {res['dark_floor'] * 100:.3f} %")
            res.update(params=dict(self.p), rho=self.rho, beta_null=float(self.beta_null),
                       history=self.history, initial=self.initial,
                       exposure_us=self.rig.exposure, minutes=(time.time() - t_start) / 60)
            out = self.write_back(res)
            res.update(out)
            if cfg.save:
                self.save(res)
            return res
        finally:
            self.rig.restore_free_run()

    def _log_budget(self, b, title):
        self.log(f"{title}: null {b['eps'] * 100:.2f} %  ->  amplitude {b['N_amp'] * 100:.2f}"
                 f" + wavefront {b['N_phase'] * 100:.2f} + decoherence "
                 f"{b['N_coh'] * 100:.2f} %  (4-step drift indicator {b['fit_residual'] * 100:.2f} %:"
                 f" if comparable to the null, drift during the measurement is limiting)")
        pf = b["phase_fit"]
        self.log("  camera-plane relative phase fit (rad, px units): "
                 + ", ".join(f"{k} {v:+.3g}" for k, v in pf.items() if k != "piston"))

    # -- hand results back to the GUI ---------------------------------------------
    def write_back(self, res):
        gui, hb, p = self.gui, self.hb, self.p
        uv = list(hb.uv)
        # new B periods from frequency shift
        nuB = np.array(hb.nuB) + np.array([p.get("dnu_x", 0.0), p.get("dnu_y", 0.0)])
        newBx = (_period_for_freq(uv[2], nuB[0], 0, uv[3], hb.bs[2], hb.bs[3])
                 if not hb.bs[2] else uv[2])
        newBy = (_period_for_freq(uv[3], nuB[1], 1, newBx, hb.bs[2], hb.bs[3])
                 if not hb.bs[3] else uv[3])
        uv[2], uv[3] = newBx, newBy
        # amplitudes (GUI normalises by max anyway)
        aA, aB = hb.ampA0, hb.ampB0 * self.rho
        uv[16], uv[17] = aA / max(aA, aB), aB / max(aA, aB)
        # Zernikes folded into B correction (GUI subtracts the correction)
        corrB_new = hb.corrB - hb.zernike_phase(p)
        if "cx" in p:
            uv[14] += p["cx"]
            uv[15] += p.get("cy", 0.0)
        if "waist" in p:
            uv[10] *= 1 + p["waist"]
            uv[11] *= 1 + p["waist"]

        # Verify with the exact GUI formula (no perturbations) and read off the
        # null phase in the GUI's own convention (Beam A global phase, %).
        hb2 = HologramBuilder(gui, uv=uv, corrB=corrB_new)
        old_hb, old_p = self.hb, self.p
        self.hb, self.p, self.rho = hb2, {}, 1.0
        imgs, IA, IB = self.run_batch([({}, b, "both") for b in self.BETAS])
        e_ver, A, B, phi = self.eps_from(imgs)
        beta_null = phi + np.pi                # B phase increment that gives the null
        uv[18] = (uv[18] - beta_null / (2 * np.pi) * 100) % 100
        # display the verified null on display_slot
        hb3 = HologramBuilder(gui, uv=uv, corrB=corrB_new)
        h = hb3.build({}, 0.0, 1.0, "both")
        self.rig.plm.pause_ui()
        np.copyto(self.rig.buf, h[None])
        self.rig.plm.bitpack_and_insert_gpu(self.rig.buf, self.cfg.display_slot)
        self.rig.plm.resume_ui()
        self.rig.show(self.cfg.display_slot)
        self.hb, self.p = old_hb, old_p
        self.log(f"Verification with GUI-formula hologram: null = {e_ver * 100:.2f} %")
        self.log(f"New B periods: x {newBx:.4f}, y {newBy:.4f}; amplitudes A {uv[16]:.3f}"
                 f" B {uv[17]:.3f}; null phase (Beam A global phase) {uv[18]:.2f} %")

        out = dict(new_uv=uv, newBx=newBx, newBy=newBy, zero_phase=uv[18],
                   eps_verified=float(e_ver), corrB_new=corrB_new)
        if self.cfg.apply_to_gui:
            for i in (2, 3):
                gui.inputs[i].setText(f"{uv[i]:.4f}")
            for i in (10, 11, 14, 15, 16, 17):
                gui.inputs[i].setText(f"{uv[i]:.4g}")
            gui.inputs[18].setText(f"{uv[18]:.2f}")
            gui.beam_B_correction_data = corrB_new
        return out

    def save(self, res):
        now = datetime.now()
        root = self.cfg.save_root or os.path.join(os.getcwd(), "Data", now.strftime("%Y_%m_%d"))
        folder = os.path.join(root, now.strftime("%H_%M_%S") + " overlap v2")
        os.makedirs(folder, exist_ok=True)
        corr_path = os.path.join(folder, "beam_B_correction_overlap_v2.npy")
        np.save(corr_path, res["corrB_new"])
        if self.cfg.apply_to_gui:
            try:
                from savePhaseFile import save_phase_file
                save_phase_file("beam_B_phase_file", corr_path)
            except Exception as e:                       # noqa: BLE001
                self.log(f"  (could not register correction file: {e})")
        im = res["images"]
        b = res["budget"]
        np.savez_compressed(os.path.join(folder, "overlap_v2_data.npz"),
                            roi=self.roi, modulation=b["maps"]["modulation"],
                            rel_phase=b["maps"]["rel_phase"], **im)
        summary = {k: v for k, v in res.items()
                   if k not in ("images", "corrB_new", "budget", "initial")}
        summary["budget"] = {k: v for k, v in b.items() if k != "maps"}
        summary["initial_budget"] = {k: v for k, v in res["initial"]["budget"].items()
                                     if k != "maps"}
        summary["config"] = asdict(self.cfg)
        with open(os.path.join(folder, "overlap_v2_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=float)
        with open(os.path.join(folder, "overlap_v2_log.txt"), "w") as f:
            f.write("\n".join(self.log_lines))
        try:
            self._figure(res, os.path.join(folder, "overlap_v2.png"))
        except Exception as e:                           # noqa: BLE001
            self.log(f"  (figure failed: {e})")
        self.log(f"Saved to {folder}")
        res["folder"] = folder

    def _figure(self, res, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        im, b = res["images"], res["budget"]
        ys, xs = np.where(self.roi)
        sl = (slice(max(ys.min() - 5, 0), ys.max() + 6), slice(max(xs.min() - 5, 0), xs.max() + 6))
        g = np.sqrt(np.clip(im["IA"], 0, None) * np.clip(im["IB"], 0, None))
        sig = self.roi & (gaussian_filter(g, 1) > 0.05 * gaussian_filter(g, 1).max())
        fig, ax = plt.subplots(2, 3, figsize=(12, 7.5))
        panels = [(im["IA"], "Channel A only"), (im["IB"], "Channel B only"),
                  (im["null"], f"Null (HDR), {res['eps_hdr'] * 100:.2f} %"),
                  (np.where(sig, b["maps"]["rel_phase"], np.nan), "Relative phase A-B (rad)"),
                  (np.where(sig, b["maps"]["modulation"], np.nan), "Local fringe visibility")]
        for a, (d, t) in zip(ax.flat, panels):
            h = a.imshow(d[sl], cmap="twilight" if "phase" in t else "viridis")
            a.set_title(t, fontsize=10)
            a.axis("off")
            fig.colorbar(h, ax=a, fraction=0.046)
        a = ax.flat[5]
        e = [self.initial["eps"]] + [min(hh["eps"]) for hh in res["history"]]
        a.semilogy(np.array(e) * 100, "o-")
        a.set_xlabel("scan")
        a.set_ylabel("null (%)")
        a.set_title(f"N_amp {b['N_amp'] * 100:.2f}  N_phase {b['N_phase'] * 100:.2f}  "
                    f"N_coh {b['N_coh'] * 100:.2f} %", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


def overlap_optimiser_v2(gui, plm, camera, cfg=None):
    """Drop-in for overlap_optimiser(): returns (newBxTilt, newByTilt, zero_phase).
    Full result dict stored on gui.last_overlap_result."""
    opt = OverlapOptimiser(gui, plm, camera, cfg or OverlapConfig())
    res = opt.run()
    gui.last_overlap_result = res
    return res["newBx"], res["newBy"], res["zero_phase"]

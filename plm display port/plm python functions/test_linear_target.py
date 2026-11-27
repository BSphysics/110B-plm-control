"""
Simulated rig for linearTarget.py: test_overlap_v2's PLM/beam-displacer optics, with
real polarisation added.  Channels A and B carry elliptical, non-orthogonal Jones
vectors (this morning's calibration: 86.5 deg apart, chi -0.89 / +0.68 deg), B has
hidden aberrations, a power loss and a group delay, and the spectrum is 6.6 nm FWHM.
The camera sits behind a motorised QWP (delta 91.7 deg) + polariser whose true
theta0 is 0.3 deg off the calibration value.
"""
import os
import sys
import json
import types
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_overlap_v2 as T                      # injects the lab-module stand-ins

# --------------------------------------------------------------------- truth
CAL = dict(theta0_deg=39.13, delta_deg=91.69, phi_A_deg=101.14, psiQ_A_deg=140.28,
           psiQ_B_deg=46.76)
TH0_TRUE = np.radians(CAL["theta0_deg"] + 0.3)
DELTA_TRUE = np.radians(91.7)
T.Truth.bw_nm = 6.6
T.Truth.lam0 = 801.0


def jones(psi_deg, chi_deg):
    p, c = np.radians(psi_deg), np.radians(chi_deg)
    v = np.array([np.cos(c), 1j * np.sin(c)])
    R = np.array([[np.cos(p), -np.sin(p)], [np.sin(p), np.cos(p)]])
    return R @ v


JA = jones(CAL["psiQ_A_deg"] - np.degrees(TH0_TRUE), -0.89)     # polariser-stage frame
JB = jones(CAL["psiQ_B_deg"] - np.degrees(TH0_TRUE), +0.68)

MOUNT = {"P": 0.0, "Q": 0.0}


def analyser_row():
    P, Q = np.radians(MOUNT["P"]), np.radians(MOUNT["Q"])
    a = Q - TH0_TRUE
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    W = R @ np.diag([1, np.exp(1j * DELTA_TRUE)]) @ R.T
    return np.array([np.cos(P), np.sin(P)]) @ W


# ------------------------------------------------------------- fake Elliptec
class FakeELL14:
    def __init__(self, port=None, address=None, **_):
        self.port, self.addr = port, address
        self.key = "P" if port == "COM5" else "Q"
        self.info = {"serial": "11400476" if self.key == "P" else "11402101"}

    def move_abs(self, deg):
        MOUNT[self.key] = float(deg) % 360 + np.random.normal(0, 0.002)
        return MOUNT[self.key]

    def release(self):
        pass


ell = types.ModuleType("elliptec")
ell.ELL14 = FakeELL14
sys.modules["elliptec"] = ell


# ------------------------------------------------------- polarisation optics
def channel_fields(H):
    """Per-wavelength camera-plane fields of A and B (copy of T.camera_intensity)."""
    q = np.round(H * 16) / 16
    F = np.fft.fft2(np.exp(2j * np.pi * q)) / (T.ROWS * T.COLS)
    nuA = (1 / T.Truth.p_true_A[0], 1 / T.Truth.p_true_A[1])
    nuB = (1 / T.Truth.p_true_B[0], 1 / T.Truth.p_true_B[1])
    eA = np.fft.ifft2(T._crop(F, nuA))
    eB = np.fft.ifft2(T._crop(F, nuB))
    eB = eB * np.sqrt(T.Truth.T_B) * np.exp(1j * sum(T.Truth.aberr_B[k] * T.ZB[k]
                                                      for k in T.Truth.aberr_B))
    s = T.Truth.bw_nm / 2.3548
    lams = T.Truth.lam0 + np.linspace(-2.5 * s, 2.5 * s, T.Truth.n_lam)
    wts = np.exp(-(lams - T.Truth.lam0) ** 2 / (2 * s ** 2))
    wts /= wts.sum()
    out = []
    W, PAD, CAM = T.W, T.PAD, T.CAM
    c = W * PAD // 2
    sl = (slice(c - CAM // 2, c + CAM // 2), slice(c - CAM // 2, c + CAM // 2))
    for lam, wt in zip(lams, wts):
        r = lam / T.Truth.lam0 - 1
        a = eA * np.exp(2j * np.pi * r * (nuA[0] * T.XS + nuA[1] * T.YS))
        b = eB * np.exp(2j * np.pi * r * (nuB[0] * T.XS + nuB[1] * T.YS))
        b = b * np.exp(2j * np.pi * 300.0 * T.Truth.tau_fs * (1 / lam - 1 / T.Truth.lam0))
        fa = np.fft.fftshift(np.fft.fft2(a, (W * PAD, W * PAD)))[sl]
        fb = np.fft.fftshift(np.fft.fft2(b, (W * PAD, W * PAD)))[sl]
        out.append((wt, fa, fb))
    return out


def intensity(fields):
    row = analyser_row()
    ca, cb = row @ JA, row @ JB
    return sum(wt * np.abs(ca * fa + cb * fb) ** 2 for wt, fa, fb in fields) * 1e6


class PolPLM(T.FakePLM):
    def bitpack_and_insert_gpu(self, phase, offset):
        assert phase.dtype == np.float32 and phase.shape == (24, T.ROWS, T.COLS)
        assert self.paused
        self.frames[offset] = channel_fields(phase[0])
        self.n_bitpacks += 1


class PolCamera(T.FakeCamera):
    def __init__(self, plm, rng):
        super().__init__(plm, rng)
        self.OffsetX, self.OffsetY = T._Node(0), T._Node(0)

    def RetrieveResult(self, *_):
        t = self.ExposureTimeAbs.GetValue()
        I = intensity(self.plm.frames[self.plm.cur]) * T.Truth.K * t + T.Truth.stray * t / 2000
        img = I + T.Truth.dark_dn + self.rng.normal(0, T.Truth.read_dn, I.shape)
        img = np.clip(np.round(img), 0, 255).astype(np.uint8)
        self._left -= 1
        if self._left <= 0:
            self._grabbing = False
        return T._Result(img)


if __name__ == "__main__":
    import time
    work = tempfile.mkdtemp(prefix="lin_")
    os.chdir(work)
    with open("analyser_cal.json", "w") as f:
        json.dump(CAL, f)
    import overlapOptimiser_v2 as ov
    import linearTarget as lt

    rng = np.random.default_rng(3)
    plm = PolPLM()
    cam = PolCamera(plm, rng)
    gui = T.FakeGUI()
    cfg = ov.OverlapConfig(bandwidth_nm=6.6, wavelength_nm=801.0, save_root=work,
                           settle_s=0.0, n_avg=3, verbose=False)
    t = time.time()
    res = lt.linearise(gui, plm, cam, target="D", cfg=cfg,
                       sweep="--sweep" in sys.argv, verify_stokes="--stokes" in sys.argv,
                       sweep_targets=(0, 22.5, 45, 67.5, 90, 135))
    print(f"\nwall {time.time() - t:.0f} s, bitpacks {plm.n_bitpacks}")

    # ---- truth check: Stokes of the displayed state, computed from the simulated fields
    fields = channel_fields(ov.HologramBuilder(gui).build({}, 0.0, 1.0, "both"))
    S = np.zeros(4)
    for wt, fa, fb in fields:
        E = JA[:, None, None] * fa + JB[:, None, None] * fb
        Ex, Ey = E
        S += wt * np.array([np.sum(abs(Ex) ** 2 + abs(Ey) ** 2), np.sum(abs(Ex) ** 2 - abs(Ey) ** 2),
                            np.sum(2 * (Ex * Ey.conj()).real), np.sum(-2 * (Ex * Ey.conj()).imag)])
    az_lab = 0.5 * np.degrees(np.arctan2(S[2], S[1]))
    azA = CAL["psiQ_A_deg"] - np.degrees(TH0_TRUE)
    L = np.linalg.norm(S[1:])
    print(f"TRUTH (whole spot): azimuth rel. A {((az_lab - azA + 90) % 180) - 90:+.2f} deg "
          f"(target {res['target_rel_A_deg']:.2f}), chi {0.5 * np.degrees(np.arcsin(S[3] / L)):+.3f} deg, "
          f"DOP {L / S[0]:.4f}  -> true eps vs target ~ {(1 - L / S[0]) / 2 * 100:.3f} % + angle terms")

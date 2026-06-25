import os
import numpy as np
from ampModPhase import amp_mod_phase

def amp_ramp_frame_generator_v2(beamA_phase, beamB_phase,
                                beamA_HG_amplitude, beamB_HG_amplitude,
                                total_rotation_deg=360,
                                amp_balance=(0.8, 1.0),
                                global_phase_A_pct=50.0,
                                global_phase_B_pct=0.0,
                                txt_path='amp_ramp_settings.txt'):
    H, W = beamA_phase.shape
    numHolograms = 24
    corrA, corrB = amp_balance

    thetas = np.linspace(0.0, np.deg2rad(total_rotation_deg),
                         numHolograms, endpoint=False)

    amp_ramp_scan_frame = np.empty((numHolograms, H, W), dtype=np.float32)
    rows = []

    for idx, theta in enumerate(thetas):
        aV = np.sin(theta)   # Beam A, vertical
        bH = np.cos(theta)   # Beam B, horizontal

        phaseA_extra = np.pi if aV < 0 else 0.0
        phaseB_extra = np.pi if bH < 0 else 0.0
        magA = abs(aV) * corrA
        magB = abs(bH) * corrB

        beamAcomplex = (beamA_HG_amplitude * magA) * np.exp(1j * (beamA_phase + phaseA_extra))
        beamBcomplex = (beamB_HG_amplitude * magB) * np.exp(1j * (beamB_phase + phaseB_extra))

        plm_phase_map = (amp_mod_phase(beamAcomplex + beamBcomplex) + np.pi) / (2 * np.pi)
        amp_ramp_scan_frame[idx, :, :] = plm_phase_map.astype(np.float32)

        # Total phase to type into each GUI box = base + the pi flip, in percent
        phaseA_total = (global_phase_A_pct + phaseA_extra / (2 * np.pi) * 100.0) % 100.0
        phaseB_total = (global_phase_B_pct + phaseB_extra / (2 * np.pi) * 100.0) % 100.0

        rows.append((idx, np.rad2deg(theta), magA, magB,
                     phaseA_total, phaseB_total))

    if txt_path is not None:
        header = (f"# amp_ramp settings  |  total_rotation_deg={total_rotation_deg}  "
                  f"|  amp_balance=(corrA={corrA}, corrB={corrB})  "
                  f"|  base_global_phase=(A={global_phase_A_pct}%, B={global_phase_B_pct}%)\n"
                  f"# {'frame':>5} {'azimuth_deg':>12} {'ampA':>8} {'ampB':>8} "
                  f"{'phaseA_tot_%':>13} {'phaseB_tot_%':>13}\n")
        with open(txt_path, 'w') as f:
            f.write(header)
            for r in rows:
                f.write(f"  {r[0]:5d} {r[1]:12.2f} {r[2]:8.4f} {r[3]:8.4f} "
                        f"{r[4]:13.1f} {r[5]:13.1f}\n")
        print(f"amp_ramp settings written to {os.path.abspath(txt_path)}")

    return amp_ramp_scan_frame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

def simulate_camera_image(finalCombined, poincare_file_path , dynamic_range_db=40, use_log_scale=True):
    """
    Simulate the far-field (Fourier plane) intensity pattern seen on a camera.
    
    Parameters:
        finalCombined   : complex array from your SLM hologram generation
        dynamic_range_db: log scale dynamic range in dB (default 40)
        use_log_scale   : whether to display in log scale (default True)
    """

    # --- 1. FFT to get the Fourier plane field ---
    # Shift zero-frequency to centre, FFT, then shift output to centre
    field_ft = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(finalCombined)
        )
    )

    # --- 2. Intensity (camera measures |E|^2) ---
    intensity = np.abs(field_ft) ** 2

    # --- 3. Normalise ---
    intensity /= intensity.max()

    # --- 4. Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # -- (a) Linear intensity --
    im0 = axes[0].imshow(intensity, cmap='inferno', origin='lower')
    axes[0].set_title('Intensity (linear)', fontsize=12)
    axes[0].set_xlabel('x (pixels)')
    axes[0].set_ylabel('y (pixels)')
    plt.colorbar(im0, ax=axes[0], label='Normalised intensity')

    # -- (b) Log intensity (reveals weak features / side lobes) --
    eps = 10 ** (-dynamic_range_db / 10)          # noise floor
    im1 = axes[1].imshow(
        intensity,
        cmap='inferno',
        origin='lower',
        norm=LogNorm(vmin=eps, vmax=1.0)
    )
    axes[1].set_title(f'Intensity (log, {dynamic_range_db} dB range)', fontsize=12)
    axes[1].set_xlabel('x (pixels)')
    plt.colorbar(im1, ax=axes[1], label='Normalised intensity (log)')

    # -- (c) Phase of the Fourier field (useful for diagnostics) --
    phase = np.angle(field_ft)
    im2 = axes[2].imshow(phase, cmap='hsv', origin='lower')
    axes[2].set_title('Phase in Fourier plane', fontsize=12)
    axes[2].set_xlabel('x (pixels)')
    plt.colorbar(im2, ax=axes[2], label='Phase (rad)')

    plt.suptitle('Simulated camera image — Fourier plane', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(poincare_file_path), 'Poincare sim.png'), dpi=300)
    # plt.show()

    return intensity, field_ft